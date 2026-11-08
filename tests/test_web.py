"""Drives server.py exactly like web/index.html does, against a local stand-in LLM. Run: python tests/test_web.py"""
import json, re, sys, tempfile, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_DATA = Path(__file__).resolve().parent.parent / "data" / "input"
if not any(_DATA.glob("*.txt")):        # these checks use the real episodes, which are not shipped with the code
    print(f"SKIPPED {Path(__file__).name}: put the episode .txt files in data/input to run it")
    sys.exit(0)
import server
from config import load_config


def line_id(prompt, needle):
    m = re.search(r"\[(\d+)\] [\d.]+-[\d.]+ [^\n]*" + re.escape(needle), prompt)
    return int(m.group(1)) if m else None


def reply(p):
    if '"overall"' in p:
        return json.dumps({"overall": "o", "items": [{"rank": 1, "reason": "r"}]})
    if "Order ALL of them" in p:                       # head-to-head: answer worst-listed first
        return json.dumps({"order": list(reversed(re.findall(r"=== CLIP ([A-Z]) ===", p))), "reason": "r"})
    if "SHOW BIBLE:" in p and "SCREENPLAY:" in p:
        ep = int(re.search(r"EPISODE (\d+)", p).group(1))
        if ep == 7:
            a, b = line_id(p, "scarred beggar"), line_id(p, "And your husband?")
            return json.dumps({"candidates": [{"segments": [[a - 1, b + 2]], "hook_event": a, "cliffhanger_event": b,
                                               "hook": "h", "cliffhanger": "c", "why": "w"}]})
        if ep == 10:
            return json.dumps({"candidates": [{"segments": [[3, 4]], "hook": "x", "cliffhanger": "y", "why": "z"}]})
        return json.dumps({"candidates": []})
    if "judging a candidate promo" in p:
        return json.dumps({k: {"score": 4, "reason": "r"} for k in ("hook", "cliffhanger", "standalone", "emotional")} | {"hook_type": "threat"})
    return json.dumps({"premise": "p", "characters": [], "aliases": {}})


HITS = [0]
class LLM(BaseHTTPRequestHandler):
    def do_POST(self):
        HITS[0] += 1
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"choices": [{"message": {"content": reply(b["messages"][-1]["content"])}}]}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass


llm = HTTPServer(("127.0.0.1", 0), LLM); threading.Thread(target=llm.serve_forever, daemon=True).start()
LLM_URL = f"http://127.0.0.1:{llm.server_port}/v1"
TMP = tempfile.mkdtemp()
real_cfg = server.cfg_now


def fake_cfg():
    c = real_cfg()
    c["llm"].update(base_url=LLM_URL, model="stand-in", min_interval_s=0, retries=1, retry_base_s=0, fallbacks=[])
    c["llm"]["scoring"] = {"base_url": LLM_URL, "model": "stand-in-scorer"}
    c["paths"]["outputs_dir"] = TMP
    c["pipeline"]["max_retries"] = 0
    c["web"] = {"daily_call_budget": BUDGET[0], "max_episodes_per_run": 60}
    return c


BUDGET = [None]                                   # no daily cap until the budget check below


server.cfg_now = fake_cfg
server.DEMO = Path(TMP) / "no-demo-yet"
web = server.make_server("127.0.0.1", 0); threading.Thread(target=web.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{web.server_port}"


def get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        body = r.read()
        return (r.status, r.headers.get("Content-Type"), body) if raw else json.loads(body)


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


FAILS = []
def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)
st, ct, body = get("/", raw=True)
ok(st == 200 and b"<title>PromoAI</title>" in body, f"GET / serves web/index.html ({len(body)} bytes)")
cfg = get("/api/config")
ok(all(k in cfg for k in ("base_url", "model", "api_key_env", "key_ready", "fallbacks")), f"/api/config fields: {cfg}")
import urllib.error
try:
    get("/api/library"); ok(False, "the data-folder endpoint is removed")
except urllib.error.HTTPError as e:
    ok(e.code == 404, "the data-folder endpoint is removed (404)")
lib = {"files": [{"name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted(_DATA.glob("*.txt"))]}   # = user uploads
files = [f for f in lib["files"] if re.search(r"episode-0*(7|8|9|10)-", f["name"])]
job = post("/api/run", {"files": files, "stitch": False})["job"]
t0 = time.time()
while True:
    s = get("/api/status?job=" + job)
    if s["state"] != "running" or time.time() - t0 > 120:
        break
    time.sleep(0.5)
ok(s["state"] == "done", f"run finished: state={s['state']} error={s.get('error')}")
ok(len(s["steps"]) == 6 and all(x["state"] == "done" for x in s["steps"]), "all 6 progress steps reached 'done'")
print("     log:", [l["msg"][:60] for l in s["log"]])
R = get("/api/result?job=" + job)
ok(all(k in R for k in ("meta", "promos", "no_promo", "audit")), f"/api/result keys: {list(R)}")
p = R["promos"][0] if R["promos"] else {}
need = ("rank", "segments", "total_duration", "final_score", "hook", "cliffhanger", "why", "scores", "lines", "notes", "hook_event", "cliffhanger_event")
ok(p and all(k in p for k in need), f"promo has every field the page reads ({len(R['promos'])} promos)")
ok(p and all("ep_duration" in sg and sg["ep_duration"] > 0 for sg in p["segments"]), "segments carry ep_duration for the timeline")
ok(p and p["lines"] and all({"ep", "id", "t", "who", "text", "kind"} <= set(l) for l in p["lines"]), f"{len(p.get('lines', []))} script lines with ep/id/t/who/text/kind")
ok(p.get("hook_event") is not None, f"hook_event passed through to the page: {p.get('hook_event')!r}")
ok(all({"name", "status", "detail"} <= set(c) for c in R["audit"]), f"audit rows: {[c['status'] for c in R['audit']]}")
ER = R.get("episode_ranking") or []
ok(sorted(r["ep"] for r in ER) == [7, 8, 9, 10] and ER[0]["status"] == "ok" and ER[0]["potential"] is not None,
   f"episode ranking covers every uploaded episode, best first: {[(r['ep'], r['potential']) for r in ER]}")
print("     no_promo:", R["no_promo"])
print("     meta.model:", R["meta"].get("model"), "| scoring_model:", R["meta"].get("scoring_model"), "| models_used:", R["meta"].get("models_used"))
for f in ("ranked_promos.csv", "dashboard.html"):
    st, ct, body = get(f"/download?job={job}&file={f}", raw=True)
    ok(st == 200 and body, f"/download {f}: {ct}, {len(body)} bytes")
ok(cfg.get("scoring_model") == "stand-in-scorer", f"/api/config reports the scoring model: {cfg.get('scoring_model')}")
ok(R["meta"].get("scoring_model") == "stand-in-scorer", "result meta carries scoring_model (page shows 'scores are comparable')")
import os
before = HITS[0]
renamed = [{"name": "copy_" + f["name"], "text": f["text"]} for f in files]       # same episodes, new run id
job2 = post("/api/run", {"files": renamed, "stitch": False})["job"]
while get("/api/status?job=" + job2)["state"] == "running":
    time.sleep(0.5)
s2 = get("/api/status?job=" + job2)
ok(job2 != job and s2["state"] == "done", f"second run (new id {job2}) finished: {s2['state']} {s2.get('error')}")
ok(HITS[0] == before, f"second run made {HITS[0] - before} new model calls (all answers reused)")
print("     log:", [l["msg"] for l in s2["log"] if "Reusing" in l["msg"]])
logp = Path(TMP) / "web" / job2 / "raw" / "calls.jsonl"
recs = [json.loads(l) for l in logp.read_text(encoding="utf-8").splitlines()]
ok(recs and all(r.get("reused") for r in recs), f"second run's own log has {len(recs)} entries, all marked reused")
import urllib.error
def status_of(headers, body=b'{"files": []}'):
    req = urllib.request.Request(BASE + "/api/run", data=body, headers=headers)
    try:
        urllib.request.urlopen(req, timeout=10); return 200
    except urllib.error.HTTPError as e:
        return e.code
ok(status_of({"Content-Type": "application/json", "Origin": "https://evil.example"}) == 403, "run request from another website is refused (403)")
ok(status_of({"Content-Type": "text/plain"}) == 415, "non-JSON run request is refused (415)")
ok(status_of({"Content-Type": "application/json", "Origin": BASE}) == 400, "the page's own origin is accepted (400 = no files, as sent)")
R2 = get("/api/result?job=" + job2)
ok([p["final_score"] for p in R2["promos"]] == [p["final_score"] for p in R["promos"]], "second run gives the same results")
# deployed host: the page's own origin behind a proxy is accepted, others are still refused
ok(status_of({"Content-Type": "application/json", "Origin": "https://promoai.example.com",
              "X-Forwarded-Host": "promoai.example.com"}) == 400, "deployed origin (X-Forwarded-Host) is accepted")

# finished example: served from saved files, no AI calls
server.DEMO = Path(TMP) / "web" / job
before = HITS[0]
cfg2 = get("/api/config")
ok(cfg2.get("demo") is True and cfg2.get("demo_episodes") == 4, f"/api/config offers the demo ({cfg2.get('demo_episodes')} episodes)")
D = get("/api/result?job=demo")
ok(len(D["promos"]) == len(R["promos"]) and HITS[0] == before, "demo result served with 0 model calls")
st, ct, body = get("/download?job=demo&file=ranked_promos.csv", raw=True)
ok(st == 200 and body, "demo CSV downloads")

# daily budget: new (uncached) episodes stop with a clear message once the budget is used
BUDGET[0] = 1
ok(get("/api/config").get("budget_left") == 0, "budget_left counts today's new calls (budget 1, already used)")
fresh = [f for f in lib["files"] if re.search(r"episode-0*(11|12)-", f["name"])]
job3 = post("/api/run", {"files": fresh, "stitch": False})["job"]
while get("/api/status?job=" + job3)["state"] == "running":
    time.sleep(0.5)
s3 = get("/api/status?job=" + job3)
ok(s3["state"] == "error" and "daily cap" in (s3.get("error") or ""), f"over budget: run stops with: {(s3.get('error') or '')[:70]}")
BUDGET[0] = None
print("outputs written under", TMP)

print("WEB TEST OK" if not FAILS else f"WEB TEST FAILED: {len(FAILS)} check(s)")
sys.exit(1 if FAILS else 0)
