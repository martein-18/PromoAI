"""Runs on a clean checkout: small synthetic episodes in tests/fixtures/, no API key, no dataset.
One test per rule or fix a reviewer might check.   Run: python tests/test_fixtures.py"""
import json, shutil, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import load_config
from screenplay_parser import parse_dir, EpisodeFileError
from validator import validate_candidate, is_valid
from llm import BaseClient
from ai_analyzer import generate_candidates

FIX = ROOT / "tests" / "fixtures"
CFG = load_config(str(ROOT / "config.yaml"))
CFG["llm"]["min_interval_s"] = 0                      # offline mock: no need to pace calls
V = CFG["validation"]
EPS = {e.ep: e for e in parse_dir(FIX)}


def eid(ep, text):
    return next(e.id for e in EPS[ep].events if text in e.text)


def check(cand, ep=1):
    segs, issues = validate_candidate(EPS, cand, ep, V)
    return segs, [i.code for i in issues if i.severity == "error"]


class Rules(unittest.TestCase):
    def test_visual_cliffhanger_after_last_line_stays_in_clip(self):
        # last dialogue ends at 33.3s; the cliffhanger is the ACTION "The doors burst open..." at 33.0-36.0s
        a, door = eid(1, "drops her glass"), eid(1, "doors burst open")
        segs, errors = check({"segments": [[a, door]], "hook_event": a, "cliffhanger_event": door})
        self.assertEqual(errors, [])
        self.assertGreaterEqual(segs[-1].end, 36.0)          # old rule cut the clip at 33.6s, before the door

    def test_hook_late_is_rejected(self):
        a, door = eid(1, "drops her glass"), eid(1, "doors burst open")
        late = eid(1, "I came for what you stole")             # starts 6.5s into the clip
        self.assertIn("hook_late", check({"segments": [[a, door]], "hook_event": late, "cliffhanger_event": door})[1])

    def test_hook_outside_clip_is_rejected(self):
        a, door = eid(1, "Then take it"), eid(1, "doors burst open")
        self.assertIn("hook_outside", check({"segments": [[a, door]], "hook_event": eid(1, "drops her glass"),
                                              "cliffhanger_event": door})[1])

    def test_clip_running_past_the_cliffhanger_is_rejected(self):
        a, door, end = eid(1, "drops her glass"), eid(1, "doors burst open"), eid(1, "lifts her veil")
        self.assertIn("cliff_not_at_end", check({"segments": [[a, end]], "hook_event": a, "cliffhanger_event": door})[1])

    def test_length_rule(self):
        a = eid(1, "drops her glass")
        self.assertIn("duration", check({"segments": [[a, a + 1]], "hook_event": a, "cliffhanger_event": a + 1})[1])

    def test_missing_ids_only_warn(self):
        a, door = eid(1, "drops her glass"), eid(1, "doors burst open")
        segs, issues = validate_candidate(EPS, {"segments": [[a, door]]}, 1, V)
        self.assertTrue(is_valid(issues))
        self.assertTrue({"hook_unverified", "cliff_unverified"} <= {i.code for i in issues})


class Files(unittest.TestCase):
    def test_duplicate_episode_number_stops_the_run(self):
        d = Path(tempfile.mkdtemp())
        shutil.copy(FIX / "episode-001-fixture.txt", d / "a.txt")
        shutil.copy(FIX / "episode-001-fixture.txt", d / "b.txt")
        with self.assertRaises(EpisodeFileError) as cm:
            parse_dir(d)
        self.assertIn("a.txt", str(cm.exception))

    def test_missing_episode_number_stops_the_run(self):
        d = Path(tempfile.mkdtemp())
        text = (FIX / "episode-001-fixture.txt").read_text(encoding="utf-8").replace("EPISODE NUMBER: Episode 1\n", "")
        (d / "untitled.txt").write_text(text, encoding="utf-8")
        with self.assertRaises(EpisodeFileError):
            parse_dir(d)


class Cache(unittest.TestCase):
    def test_broken_reply_is_not_served_from_cache_again(self):
        replies = ["{broken json", '{"candidates": []}']

        class Flaky(BaseClient):
            def _call(self, prompt, system, task):
                return replies.pop(0)

        raw = Path(tempfile.mkdtemp())
        c = Flaky({"model": "flaky"}, raw)
        self.assertIsNone(generate_candidates(None, {}, c, ep_text="EPISODE 1", default_ep=1))   # broken -> failure
        self.assertEqual(generate_candidates(None, {}, c, ep_text="EPISODE 1", default_ep=1), [])  # asked again
        self.assertEqual(replies, [])                                    # the model really was called twice
        c2 = Flaky({"model": "flaky"}, raw)                              # a later run reading the same log
        self.assertEqual(generate_candidates(None, {}, c2, ep_text="EPISODE 1", default_ep=1), [])  # good one cached


class Pipeline(unittest.TestCase):
    def test_full_pipeline_offline(self):
        from main import run_pipeline
        out = Path(tempfile.mkdtemp()) / "run"
        r = run_pipeline(CFG, str(FIX), out, mock=True, stitch=True, progress=lambda m: None)
        self.assertTrue(r["ranked"])
        for f in ("ranked_promos.csv", "ranked_promos.json", "audit.json", "posting_plan.md", "PROMOS.md", "dashboard.html"):
            self.assertTrue((out / f).exists(), f)
        raw = json.loads((out / "raw_results.json").read_text(encoding="utf-8"))
        self.assertTrue(any(p["source"] == "assembled" for p in raw["candidates"])
                        or any(x["source"].startswith("assembled") for x in raw["rejected"]), "cross-episode stage ran")
        self.assertIn("## 1. Which to post first", (out / "PROMOS.md").read_text(encoding="utf-8"))
        audit = {c["name"]: c["status"] for c in json.loads((out / "audit.json").read_text(encoding="utf-8"))["checks"]}
        for name in ("Timestamps match screenplay", "Duration rule 20-60s", "Hook opens, cliffhanger closes"):
            self.assertEqual(audit.get(name), "pass", name)


    def test_quota_stop_still_writes_partial_results(self):
        import copy
        from main import run_pipeline
        cfg = copy.deepcopy(CFG)
        cfg["llm"]["call_budget"] = {"left": 5}          # quota "runs out" after 5 new calls, mid-run
        out = Path(tempfile.mkdtemp()) / "run"
        r = run_pipeline(cfg, str(FIX), out, mock=True, stitch=False, progress=lambda m: None)
        self.assertIn("PARTIAL", r["meta"]["status"])
        self.assertTrue(r["ranked"], "promos finished before the stop are kept")
        self.assertIn("PARTIAL", (out / "PROMOS.md").read_text(encoding="utf-8"))
        audit = {c["name"]: c["status"] for c in json.loads((out / "audit.json").read_text(encoding="utf-8"))["checks"]}
        self.assertEqual(audit.get("Run complete"), "warn")
        er = r["meta"]["episode_ranking"]
        self.assertEqual(sorted(x["ep"] for x in er), [1, 2, 3])          # every episode is listed
        self.assertTrue(any(x["status"] == "not_processed" for x in er))   # and unprocessed ones say so

    def test_episode_ranking_follows_final_promo_rank(self):
        import copy
        from main import run_pipeline
        out = Path(tempfile.mkdtemp()) / "run"
        r = run_pipeline(copy.deepcopy(CFG), str(FIX), out, mock=True, stitch=False, progress=lambda m: None)
        er = r["meta"]["episode_ranking"]
        ranks = [x["best_promo_rank"] for x in er if x.get("best_promo_rank")]
        self.assertEqual(ranks, sorted(ranks))                          # ordered by best promo in the final list
        self.assertTrue(all(x["potential"] is None for x in er[len(ranks):] if not x.get("best_promo_rank")
                            and x["status"] != "ok"))                   # episodes without a promo come last
        self.assertIn("Episodes ranked by their best promo", (out / "PROMOS.md").read_text(encoding="utf-8"))


class Failover(unittest.TestCase):
    def test_busy_model_hands_over_quickly(self):
        import threading, time as _t
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from llm import ChainClient
        class Busy(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"])); self.send_response(503); self.end_headers()
                self.wfile.write(b'{"error":"high demand"}')
            def log_message(self, *a): pass
        class Fine(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"])); self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
            def log_message(self, *a): pass
        servers = [HTTPServer(("127.0.0.1", 0), h) for h in (Busy, Fine)]
        for sv in servers:
            threading.Thread(target=sv.serve_forever, daemon=True).start()
        url = lambda sv: f"http://127.0.0.1:{sv.server_port}/v1"
        cfg = {"model": "busy", "base_url": url(servers[0]), "retries": 6, "retry_base_s": 1, "min_interval_s": 0,
               "fallbacks": [{"model": "fine", "base_url": url(servers[1])}]}
        c = ChainClient(cfg, Path(tempfile.mkdtemp()))
        t = _t.time()
        self.assertEqual(c.call("hi"), "ok")
        self.assertEqual(c.last_model, "fine")
        self.assertLess(_t.time() - t, 5)        # 2 quick tries, not 6 tries with growing waits


    def _chain(self, primary_reply):
        """A primary that always fails with primary_reply=(code, body), and a healthy backup. Counts primary hits."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from llm import ChainClient
        hits = [0]
        class Bad(BaseHTTPRequestHandler):
            def do_POST(self):
                hits[0] += 1
                self.rfile.read(int(self.headers["Content-Length"])); self.send_response(primary_reply[0]); self.end_headers()
                self.wfile.write(primary_reply[1].encode())
            def log_message(self, *a): pass
        class Good(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"])); self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
            def log_message(self, *a): pass
        sv = [HTTPServer(("127.0.0.1", 0), h) for h in (Bad, Good)]
        for x in sv:
            threading.Thread(target=x.serve_forever, daemon=True).start()
        cfg = {"model": "primary", "base_url": f"http://127.0.0.1:{sv[0].server_port}/v1", "retries": 6, "retry_base_s": 5,
               "min_interval_s": 0, "fallbacks": [{"model": "backup", "base_url": f"http://127.0.0.1:{sv[1].server_port}/v1"}]}
        make = lambda: ChainClient(cfg, Path(tempfile.mkdtemp()))
        return make, hits

    def test_daily_quota_switches_at_once_and_later_runs_skip_it(self):
        import time as _t
        make, hits = self._chain((429, '{"error":{"message":"You exceeded your current quota","details":'
                                       '[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}'))
        t = _t.time()
        c = make()
        self.assertEqual((c.call("a"), c.last_model, hits[0]), ("ok", "backup", 1))   # one try, then the backup
        self.assertLess(_t.time() - t, 2)
        c2 = make()                                                                      # a later run on the same server
        self.assertEqual((c2.call("b"), c2.last_model, hits[0]), ("ok", "backup", 1))  # primary not even tried

    def test_per_minute_limit_hands_over_without_waiting(self):
        import time as _t
        make, hits = self._chain((429, '{"error":{"message":"Resource exhausted, retry later","details":'
                                       '[{"@type":"RetryInfo","retryDelay":"20s"}]}}'))
        t = _t.time()
        c = make()
        self.assertEqual((c.call("a"), c.last_model, hits[0]), ("ok", "backup", 1))
        self.assertLess(_t.time() - t, 2)                                                # did not sit out the 20s


if __name__ == "__main__":
    unittest.main()
