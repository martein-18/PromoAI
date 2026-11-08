"""Runs the FULL pipeline through the real HTTP client against a local stand-in server that acts like an LLM
and misbehaves on purpose (broken JSON, an invented quote, an invalid pick). Proves the real channel path,
the failure handling and the audit all work. Run: python tests/test_real_channel.py"""
import json, os, re, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_DATA = Path(__file__).resolve().parent.parent / "data" / "input"
if not any(_DATA.glob("*.txt")):        # these checks use the real episodes, which are not shipped with the code
    print(f"SKIPPED {Path(__file__).name}: put the episode .txt files in data/input to run it")
    sys.exit(0)
from config import load_config
from main import run_pipeline

def line_id(prompt, needle):
    m = re.search(r"\[(\d+)\] [\d.]+-[\d.]+ [^\n]*" + re.escape(needle), prompt)
    return int(m.group(1)) if m else None

def reply(prompt):
    if "Return ONLY JSON: {\"overall\"" in prompt or '"overall"' in prompt and "Explain the ordering" in prompt:
        return json.dumps({"overall": "Post the Ep 7 clip first: strongest threat and open ending.",
                           "items": [{"rank": i, "reason": f"position {i} reason"} for i in range(1, 6)]})
    if "Order ALL of them" in prompt:                       # head-to-head: answer worst-listed first
        return json.dumps({"order": list(reversed(re.findall(r"=== CLIP ([A-Z]) ===", prompt))), "reason": "r"})
    if "SHOW BIBLE:" in prompt and "SCREENPLAY:" in prompt:
        ep = int(re.search(r"EPISODE (\d+)", prompt).group(1))
        if ep == 7:
            a, b = line_id(prompt, "scarred beggar"), line_id(prompt, "And your husband?")
            return "Here you go:\n```json\n" + json.dumps({"candidates": [{"segments": [[a - 1, b + 2]],
                "hook": "Gerald sneers \"You chose that scarred beggar over a life of comfort.\"",
                "cliffhanger": "Gerald asks \"And your husband?\" as a figure walks out of the fog",
                "why": "Public humiliation then a looming threat"}]}) + "\n```"
        if ep == 9:
            a = line_id(prompt, "rumor going around")
            return json.dumps({"candidates": [{"segments": [[a - 1, max(int(x) for x in re.findall(r"^\[(\d+)\]", prompt, re.M))]],
                "hook": "The Groom whispers \"I know who you really are, Duke\"",   # invented line
                "cliffhanger": "Nobleman whispers Impossible", "why": "Hints a dead Duke is alive"}]})
        if ep == 8:
            return "I think the best moment is when Juliana trims the beard."     # not JSON, every time
        if ep == 10:
            return json.dumps({"candidates": [{"segments": [[3, 4]], "hook": "x", "cliffhanger": "y", "why": "z"}]})  # too short
        return json.dumps({"candidates": []})                 # other episodes: a valid "nothing worth promoting"
    if "You have never seen this show" in prompt:
        return json.dumps({"characters": "A woman and two visitors", "conflict": "They mock her", "next": "who is the husband", "can_answer": True})
    if "judging a candidate promo" in prompt:
        return json.dumps({"hook": {"score": 4, "reason": "sharp insult"}, "cliffhanger": {"score": 5, "reason": "open threat"},
                           "standalone": {"score": 3, "reason": "needs context"}, "emotional": {"score": 4, "reason": "conflict"}, "hook_type": "threat"})
    return json.dumps({"premise": "p", "characters": [], "aliases": {}})

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = reply(body["messages"][-1]["content"])
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": out}}]}).encode())
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
cfg["llm"].update(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="stand-in-llm", min_interval_s=0,
                  retries=1, retry_base_s=0, fallbacks=[])
cfg["llm"]["scoring"] = {"base_url": cfg["llm"]["base_url"], "model": "stand-in-llm"}   # exercise the scoring client
cfg["pipeline"]["max_retries"] = 1
os.environ["LLM_API_KEY"] = "dummy"
out = Path(tempfile.mkdtemp()) / "run"
res = run_pipeline(cfg, Path(__file__).resolve().parent.parent / "data" / "input", out, progress=lambda m: None)
a = {c["name"]: c for c in res["audit"]["checks"]}
assert res["audit"]["status"] == "REAL RUN", res["audit"]["status"]
assert res["meta"]["models_used"] == ["stand-in-llm"]
print("ranked eps:", [p["segments"][0]["ep"] for p in res["ranked"]]); print("rejected:", [(r["source"], r["issues"][:1]) for r in res["rejected"]])
assert res["meta"]["stats"]["json_failures"] >= 2, res["meta"]["stats"]          # Ep 8 broken JSON detected, not hidden
assert a["Model reply format"]["status"] == "warn"
assert a["Quoted lines exist in script"]["status"] == "warn"                      # invented quote flagged
reasons = " ".join(" ".join(r["issues"]) for r in res["rejected"])
assert "model_reply_not_json" in reasons and "duration" in reasons        # broken JSON + too-short pick both rejected
assert any("not inside the clip" in n for notes in res["audit"]["per_promo_notes"].values() for n in notes)   # invented quote caught
assert res["ranked"][0]["post_reason"]
print("REAL-CHANNEL TEST OK")
for c in res["audit"]["checks"]:
    print(f"  [{c['status'].upper():4}] {c['name']}: {c['detail']}")
print("dashboard:", out / "dashboard.html")