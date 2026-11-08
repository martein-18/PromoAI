"""Duration-repair guardrails, using real near-miss candidates from run_A. Run: python tests/test_repair.py"""
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_DATA = Path(__file__).resolve().parent.parent / "data" / "input"
if not any(_DATA.glob("*.txt")):        # these checks use the real episodes, which are not shipped with the code
    print(f"SKIPPED {Path(__file__).name}: put the episode .txt files in data/input to run it")
    sys.exit(0)
from config import load_config
from main import load_episodes
from validator import validate_candidate, is_valid
from repair import try_repair

CFG = load_config(str(ROOT / "config.yaml"))
V, RC = CFG["validation"], CFG["pipeline"]["duration_repair"]
EPS = {e.ep: e for e in load_episodes(str(ROOT / CFG["paths"]["data_dir"]))}


def attempt(cand, ep, rc=RC):
    segs, issues = validate_candidate(EPS, cand, ep, V)
    return segs, issues, try_repair(EPS, cand, ep, V, segs, issues, rc)


class T(unittest.TestCase):
    def test_near_miss_short_is_repaired_and_valid(self):
        c = {"segments": [[87, 99]], "hook_event": 87, "cliffhanger_event": 99}      # ep4, 19.8s
        segs, issues, fix = attempt(c, 4)
        self.assertFalse(is_valid(issues))
        self.assertIsNotNone(fix)
        c2, segs2, issues2, info = fix
        self.assertTrue(is_valid(issues2))
        self.assertTrue(V["min_duration_s"] <= sum(s.duration for s in segs2) <= V["max_duration_s"])
        # one line, and both protected events still inside the clip
        self.assertEqual(abs(segs2[0].start_id - 87) + abs(segs2[-1].end_id - 99), 1)
        self.assertTrue(segs2[0].start_id <= 87 and segs2[-1].end_id >= 99)
        self.assertEqual(info["original_segments"], [[87, 99]])

    def test_far_miss_stays_rejected(self):
        c = {"segments": [[1, 47]], "hook_event": 1, "cliffhanger_event": 47}        # ep18, 86.5s
        self.assertIsNone(attempt(c, 18)[2])

    def test_needs_more_than_one_line_stays_rejected(self):
        c = {"segments": [[11, 53]], "hook_event": 13, "cliffhanger_event": 53}      # ep41, 64.1s
        self.assertIsNone(attempt(c, 41)[2])

    def test_no_hook_id_means_no_start_edit(self):
        c = {"segments": [[87, 99]], "cliffhanger_event": 99}                       # hook unknown
        fix = attempt(c, 4, {**RC, "allow_extend_end": False})[2]
        self.assertIsNone(fix)

    def test_other_errors_are_never_repaired(self):
        c = {"segments": [[63, 81]], "hook_event": 63, "cliffhanger_event": 81}      # ep22, slow_open
        segs, issues, fix = attempt(c, 22)
        self.assertTrue(any(i.code == "slow_open" for i in issues))
        self.assertIsNone(fix)

    def test_visual_cliffhanger_now_kept_in_clip(self):
        # ep18 39-49 ends on an action beat after the last line; it used to be cut off (19.8s, rejected)
        segs, issues = validate_candidate(EPS, {"segments": [[39, 49]], "hook_event": 39, "cliffhanger_event": 49}, 18, V)
        self.assertTrue(is_valid(issues), [str(i) for i in issues])

    def test_disabled(self):
        c = {"segments": [[87, 99]], "hook_event": 87, "cliffhanger_event": 99}
        self.assertIsNone(attempt(c, 4, {**RC, "enabled": False})[2])


if __name__ == "__main__":
    unittest.main()
