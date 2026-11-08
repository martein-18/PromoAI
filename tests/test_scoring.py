"""Repeat scoring, head-to-head comparison and reordering. No key or data needed. Run: python tests/test_scoring.py"""
import json, re, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_analyzer import score_clip, compare_top
from promo_scorer import apply_comparison


def S(h, c, s, e, tag="threat"):
    return json.dumps({"hook": {"score": h, "reason": f"h{h}"}, "cliffhanger": {"score": c, "reason": f"c{c}"},
                       "standalone": {"score": s, "reason": f"s{s}"}, "emotional": {"score": e, "reason": f"e{e}"},
                       "hook_type": tag})


class Client:
    def __init__(self, fn):
        self.fn, self.prompts = fn, []

    def call(self, prompt, system=None, task=""):
        self.prompts.append(prompt)
        return self.fn(prompt, len(self.prompts))


class Scoring(unittest.TestCase):
    def test_agreeing_passes_stop_after_two(self):
        c = Client(lambda p, n: S(4, 3, 3, 2))
        out = score_clip("clip", {}, {}, c, samples=3)
        self.assertEqual(len(c.prompts), 2)
        self.assertTrue(out["_agreed"])
        self.assertEqual(out["hook"]["score"], 4)
        self.assertEqual(len(set(c.prompts)), 2)                 # pass 2 is a different prompt: not a cache hit

    def test_disagreeing_passes_take_a_third_and_average(self):
        answers = [S(4, 3, 3, 2), S(2, 3, 3, 2), S(3, 4, 3, 2)]
        c = Client(lambda p, n: answers[n - 1])
        out = score_clip("clip", {}, {}, c, samples=3)
        self.assertEqual(len(c.prompts), 3)
        self.assertFalse(out["_agreed"])
        self.assertEqual(out["hook"]["score"], 3.0)
        self.assertEqual(out["hook"]["passes"], [4, 2, 3])
        self.assertAlmostEqual(out["cliffhanger"]["score"], 3.33, places=2)
        self.assertEqual(out["hook"]["reason"], "h3")           # reason from the pass closest to the mean

    def test_single_sample_is_unchanged(self):
        c = Client(lambda p, n: S(5, 1, 2, 3))
        out = score_clip("clip", {}, {}, c, samples=1)
        self.assertEqual((len(c.prompts), out["hook"]["score"], out["_passes"]), (1, 5, 1))

    def test_one_broken_pass_is_dropped_not_averaged(self):
        answers = ["broken", "broken again", S(4, 4, 4, 4), S(2, 2, 2, 2), S(3, 3, 3, 3)]
        c = Client(lambda p, n: answers[n - 1])
        out = score_clip("clip", {}, {}, c, samples=3)
        self.assertEqual(out["_passes"], 3 - 1)                  # pass 1 failed twice; passes 2 and 3 used
        self.assertEqual(out["hook"]["score"], 3.0)


class Comparison(unittest.TestCase):
    TOP = [{"id": f"p{i}", "clip_text": f"text {i}", "rank": i + 1} for i in range(4)]

    def test_consistent_passes_reorder_and_agree(self):
        # the model always prefers p3 > p2 > p1 > p0, whatever order the list is shown in
        pref = {"text 3": 0, "text 2": 1, "text 1": 2, "text 0": 3}
        def fn(p, n):
            clips = re.findall(r"=== CLIP ([A-Z]) ===\n(text \d)", p)
            return json.dumps({"order": [L for L, t in sorted(clips, key=lambda x: pref[x[1]])]})
        out = compare_top(self.TOP, Client(fn))
        self.assertEqual(out["order"], ["p3", "p2", "p1", "p0"])
        self.assertEqual((out["passes"], out["agreement"]), (2, 1.0))

    def test_position_bias_is_exposed(self):
        # a model that always picks the first-listed clip: forward and reversed passes contradict each other
        def fn(p, n):
            return json.dumps({"order": re.findall(r"=== CLIP ([A-Z]) ===", p)})
        out = compare_top(self.TOP, Client(fn))
        self.assertEqual(out["agreement"], -1.0)
        self.assertEqual(out["order"], ["p0", "p1", "p2", "p3"])   # ties fall back to the score order

    def test_invalid_replies_give_nothing(self):
        out = compare_top(self.TOP, Client(lambda p, n: json.dumps({"order": ["A", "A", "B"]})))
        self.assertEqual(out, {})

    def test_apply_comparison_keeps_score_rank(self):
        ranked = [dict(p) for p in self.TOP] + [{"id": "p9", "rank": 5}]
        out = apply_comparison(ranked, {"order": ["p2", "p0", "p1", "p3"]})
        self.assertEqual([p["id"] for p in out], ["p2", "p0", "p1", "p3", "p9"])
        self.assertEqual([p["rank"] for p in out], [1, 2, 3, 4, 5])
        self.assertEqual(out[0]["score_rank"], 3)
        self.assertEqual(apply_comparison([dict(p) for p in self.TOP], {})[0]["id"], "p0")


if __name__ == "__main__":
    unittest.main()
