"""Local web server for the PromoAI interface. Standard library only, no extra packages.

    python server.py                 ->  http://127.0.0.1:8000
    python server.py --port 8080

It serves web/index.html and runs the real pipeline (main.run_pipeline) in the background.
Each run is saved in outputs/web/<run id>/. The run id is a hash of the episode files, the model and the
settings, so pressing Find promos again after a crash or a quota error resumes from the saved AI answers.
"""
import argparse, contextlib, hashlib, json, os, re, sys, threading, time, webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from config import load_config
from main import run_pipeline

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
DEMO = ROOT / "demo"             # a finished run shipped with the app (python make_demo.py run_A); costs no AI calls
N_STEPS = 6                      # read, characters, clips, rank, explain, check
JOBS, LOCK = {}, threading.Lock()
JOB_RE = re.compile(r"^[0-9a-f]{10}$")
DOWNLOADS = {"ranked_promos.csv": "text/csv; charset=utf-8", "ranked_promos.json": "application/json",
             "report.md": "text/markdown; charset=utf-8",
             "posting_plan.md": "text/markdown; charset=utf-8", "PROMOS.md": "text/markdown; charset=utf-8", "dashboard.html": "text/html; charset=utf-8"}


def cfg_now():
    return load_config(CONFIG)


def out_root(cfg):
    return ROOT / cfg["paths"]["outputs_dir"] / "web"


def earlier_logs(cfg, job_id):
    """calls.jsonl of every earlier run (command line and web), so the same episodes are not paid for twice."""
    root = ROOT / cfg["paths"]["outputs_dir"]
    logs = list(root.glob("*/raw/calls.jsonl")) + list((root / "web").glob("*/raw/calls.jsonl"))
    logs += [DEMO / "raw" / "calls.jsonl"] if (DEMO / "raw" / "calls.jsonl").exists() else []
    return [str(p) for p in logs if p.parent.parent.name != job_id]


def new_calls_today(cfg):
    """New (not reused) model calls made by web runs since local midnight: what the daily budget counts."""
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    n = 0
    for log in (out_root(cfg)).glob("*/raw/calls.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            n += "task" in r and not r.get("reused") and r.get("ts", 0) >= midnight
    return n


def budget_left(cfg):
    b = (cfg.get("web") or {}).get("daily_call_budget")
    return None if not b else max(0, b - new_calls_today(cfg))


# ------------------------------------------------------------------ jobs
def new_job(job_id):
    return {"id": job_id, "state": "running", "steps": [{"state": "idle", "note": "", "progress": None} for _ in range(N_STEPS)],
            "log": [], "error": None, "started": time.time(), "finished": None}


def say(job, msg):
    job["log"].append({"t": round(time.time() - job["started"], 1), "msg": msg})


def mark(job, idx):
    for i, s in enumerate(job["steps"]):
        if i < idx:
            s["state"] = "done"
        elif i == idx:
            s["state"] = "active"


def make_progress(job):
    def on(msg):
        if msg.startswith("reading episodes"):
            mark(job, 0); say(job, "Reading the episode files.")
        elif msg.startswith("read "):
            job["steps"][0]["note"] = msg[5:]; say(job, "Read " + msg[5:] + ".")
        elif msg.startswith("building show bible"):
            mark(job, 1); say(job, "Learning who the characters are.")
        elif msg.startswith("episode "):
            m = re.search(r"episode (\d+) \((\d+)/(\d+)\)", msg)
            mark(job, 2)
            if m:
                ep, i, n = m.groups()
                job["steps"][2]["note"] = f"{i} of {n} episodes"
                job["steps"][2]["progress"] = (int(i) - 1) / int(n)
                say(job, f"Episode {ep}: picking, checking and scoring clips ({i} of {n}).")
        elif msg.startswith("stitching"):
            mark(job, 2); job["steps"][2]["note"] = "clips across episodes"; job["steps"][2]["progress"] = None
            say(job, "Looking for clips that run across two episodes (" + msg.split(" ", 1)[1] + ").")
        elif msg.startswith("comparing"):
            mark(job, 3); say(job, "Comparing the top clips against each other.")
        elif msg == "ranking":
            mark(job, 3); say(job, "Ranking the clips and removing overlaps.")
        elif msg.startswith("explaining"):
            mark(job, 4); say(job, "Writing why the top clips rank where they do.")
        elif msg.startswith("checking"):
            mark(job, 5); say(job, "Re-checking every timestamp against your files.")
        elif msg.startswith("WARNING"):
            say(job, msg)
    return on


class Tee:
    """Passes console output through and turns the AI client's retry notices into plain log lines."""
    def __init__(self, job, real):
        self.job, self.real, self.buf = job, real, ""

    def write(self, s):
        self.real.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if "[fallback]" in line:
                say(self.job, line.strip())
            elif "attempt" in line and "failed" in line:
                say(self.job, "The AI provider did not accept a request, trying again. " + line.strip()[:150])
        return len(s)

    def flush(self):
        self.real.flush()


def safe_name(name, i):
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(name)).name) or f"episode_{i}.txt"
    return base if base.lower().endswith(".txt") else base + ".txt"


def work(job, files, stitch):
    try:
        cfg = cfg_now()
        out = out_root(cfg) / job["id"]
        inp = out / "input"
        inp.mkdir(parents=True, exist_ok=True)
        for old in inp.glob("*.txt"):
            old.unlink()
        for i, f in enumerate(files):
            (inp / f"{i:03d}_{safe_name(f['name'], i)}").write_text(f["text"], encoding="utf-8")
        left = budget_left(cfg)
        if left is not None:
            cfg["llm"]["call_budget"] = {"left": left}        # shared by the clip-picking and scoring clients
        cfg["llm"]["cache_from"] = earlier_logs(cfg, job["id"])
        if cfg["llm"]["cache_from"]:
            say(job, f"Reusing saved AI answers from {len(cfg['llm']['cache_from'])} earlier run(s) where the episodes match.")
        with contextlib.redirect_stdout(Tee(job, sys.__stdout__)):
            r = run_pipeline(cfg, str(inp), out, None, False, stitch, make_progress(job))
        for s in job["steps"]:
            s["state"] = "done"
        m = r["meta"]
        if m.get("stopped_reason"):                     # partial: say so in the steps, not just in the results
            job["steps"][2].update(state="stopped", progress=None,
                                   note=f"stopped after {len(m.get('episodes_done', []))} of {len(m.get('episodes', []))} episodes")
            job["steps"][4].update(state="stopped", note="skipped (run stopped early)")
            say(job, "Stopped early: " + m["stopped_reason"][:200] + " Showing what was finished.")
        else:
            say(job, "Finished.")
        job["state"] = "done"
    except Exception as e:                                   # shown to the person, not hidden
        job["state"] = "error"
        job["error"] = str(e) or e.__class__.__name__
        say(job, "Stopped: " + job["error"][:300])
    finally:
        job["finished"] = time.time()


def start_run(body):
    files = body.get("files") or []
    if not files or not all(isinstance(f, dict) and isinstance(f.get("text"), str) for f in files):
        raise ValueError("No episode files were sent.")
    cap = (cfg_now().get("web") or {}).get("max_episodes_per_run", 60)
    if len(files) > cap:
        raise ValueError(f"At most {cap} episodes per run.")
    stitch = bool(body.get("stitch"))
    h = hashlib.sha1()
    for f in sorted(files, key=lambda f: str(f.get("name"))):
        h.update(str(f.get("name")).encode()); h.update(f["text"].encode("utf-8"))
    for p in [CONFIG] + sorted((ROOT / "prompts").glob("*.txt")):     # new settings or prompts -> new run folder
        h.update(p.read_bytes())
    h.update(str(stitch).encode())
    job_id = h.hexdigest()[:10]
    with LOCK:
        running = [j for j in JOBS.values() if j["state"] == "running"]
        if running:
            if running[0]["id"] == job_id:
                return job_id
            raise RuntimeError("Another run is still going. Wait for it to finish first.")
        job = JOBS[job_id] = new_job(job_id)
    threading.Thread(target=work, args=(job, files, stitch), daemon=True).start()
    return job_id


# ------------------------------------------------------------------ results
def run_dir(job_id):
    return DEMO if job_id == "demo" else out_root(cfg_now()) / job_id


def build_result(job_id):
    out = run_dir(job_id)
    rd = lambda n: json.loads((out / n).read_text(encoding="utf-8"))
    ranked, raw, audit = rd("ranked_promos.json"), rd("raw_results.json"), rd("audit.json")
    parsed = {}
    for f in sorted((out / "parsed").glob("ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        parsed[d["ep"]] = d
    promos = []
    for p in ranked["promos"]:
        segs, lines = [], []
        for s in p["segments"]:
            ep = parsed.get(s["ep"], {"events": [], "duration": 0})
            segs.append({"ep": s["ep"], "start": s["start"], "end": s["end"], "start_ts": s["start_ts"], "end_ts": s["end_ts"],
                         "ep_duration": ep["duration"]})
            evs = [e for e in ep["events"] if s["start_id"] <= e["id"] <= s["end_id"] and e["type"] != "CAMERA"]
            for e in sorted(evs, key=lambda e: (e["start"], e["id"])):
                who = ""
                if e["type"] == "DIALOGUE":
                    who = e["speaker"] + (" (" + " ".join(e["modifiers"]) + ")" if e["modifiers"] else "")
                lines.append({"ep": s["ep"], "id": e["id"], "t": e["start"], "who": who, "text": e["text"], "kind": e["type"]})
        notes = ([f"check by hand: {r}" for r in p.get("review") or []]
                 + (audit.get("per_promo_notes") or {}).get(p["id"], []) + p.get("warnings", []))
        promos.append({"rank": p["rank"], "id": p["id"], "segments": segs, "total_duration": p["total_duration"],
                       "final_score": p.get("final_score"), "hook": p.get("hook", ""), "cliffhanger": p.get("cliffhanger", ""),
                       "why": p.get("why", ""), "post_reason": p.get("post_reason", ""), "scores": p.get("scores") or {},
                       "cold_viewer": p.get("cold_viewer") or {}, "hook_event": p.get("hook_event"),
                       "cliffhanger_event": p.get("cliffhanger_event"), "lines": lines, "notes": notes})
    covered = {s["ep"] for p in promos for s in p["segments"]}
    per_ep = {}
    for r in raw.get("rejected", []):
        m = re.match(r"ep(\d+)#", r.get("source", ""))
        if m:
            c = per_ep.setdefault(int(m.group(1)), Counter())
            c["_n"] += 1
            for issue in r.get("issues", []):
                mm = re.match(r"\[\w+\] (\w+)", issue)
                if mm:
                    c[mm.group(1)] += 1
    no_promo = [{"ep": ep, "rejected": per_ep.get(ep, Counter())["_n"],
                 "reasons": [k for k, _ in per_ep.get(ep, Counter()).most_common() if k != "_n"][:3],
                 "not_processed": ep not in set(ranked["meta"].get("episodes_done", parsed))}
                for ep in sorted(parsed) if ep not in covered]
    return {"job": job_id, "meta": ranked["meta"], "promos": promos, "no_promo": no_promo,
            "episode_ranking": ranked["meta"].get("episode_ranking") or [], "audit": audit.get("checks", []),
            "rejected_count": len(raw.get("rejected", [])), "episodes": [{"ep": k, "duration": v["duration"]} for k, v in parsed.items()]}


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype, extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self.send(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/config":
                cfg = cfg_now()
                lc = cfg["llm"]
                self.json({"base_url": lc["base_url"], "model": lc["model"], "api_key_env": lc.get("api_key_env", "LLM_API_KEY"),
                           "key_ready": bool(os.environ.get(lc.get("api_key_env", "LLM_API_KEY"))),
                           "fallbacks": [f.get("model") for f in (lc.get("fallbacks") or [])],
                           "scoring_model": (lc.get("scoring") or {}).get("model"),
                           "demo": (DEMO / "ranked_promos.json").exists(),
                           "demo_episodes": len(list((DEMO / "parsed").glob("ep*.json"))) if DEMO.exists() else 0,
                           "budget_left": budget_left(cfg)})
            elif u.path == "/api/status":
                job = JOBS.get(q.get("job", ""))
                if not job:
                    return self.json({"error": "Unknown run."}, 404)
                self.json({"state": job["state"], "steps": job["steps"], "log": job["log"], "error": job["error"],
                           "elapsed": round((job["finished"] or time.time()) - job["started"], 1)})
            elif u.path == "/api/result":
                job_id = q.get("job", "")
                demo_ok = job_id == "demo" and (DEMO / "ranked_promos.json").exists()
                if not demo_ok and (not JOB_RE.match(job_id) or JOBS.get(job_id, {}).get("state") != "done"):
                    return self.json({"error": "This run has no finished results."}, 404)
                self.json(build_result(job_id))
            elif u.path == "/download":
                job_id, name = q.get("job", ""), q.get("file", "")
                if not (JOB_RE.match(job_id) or job_id == "demo") or name not in DOWNLOADS:
                    return self.json({"error": "Not found."}, 404)
                path = run_dir(job_id) / name
                if not path.exists():
                    return self.json({"error": "Not found."}, 404)
                extra = {} if name == "dashboard.html" else {"Content-Disposition": f'attachment; filename="{name}"'}
                self.send(200, path.read_bytes(), DOWNLOADS[name], extra)
            else:
                self.json({"error": "Not found."}, 404)
        except Exception as e:
            self.json({"error": str(e)}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.json({"error": "Not found."}, 404)
        # Only this page may start runs: another website open in the browser must not be able to spend the API key.
        origin = self.headers.get("Origin")
        own = {h for h in (self.headers.get("Host"), self.headers.get("X-Forwarded-Host")) if h}
        if origin and urlparse(origin).netloc not in own:
            return self.json({"error": "Requests from other websites are not allowed."}, 403)
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.json({"error": "Expected JSON."}, 415)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 200 * 1024 * 1024:
                return self.json({"error": "That is too much text for one run."}, 413)
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            self.json({"job": start_run(body)})
        except ValueError as e:
            self.json({"error": str(e)}, 400)
        except RuntimeError as e:
            self.json({"error": str(e)}, 409)
        except Exception as e:
            self.json({"error": str(e)}, 500)


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    hosted = "PORT" in os.environ                  # set by hosting services (Render, Railway, ...)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0" if hosted else "127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", default=hosted)
    a = ap.parse_args()
    srv = make_server(a.host, a.port)
    url = f"http://127.0.0.1:{a.port}"
    print(f"PromoAI is running at {url}   (press Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")