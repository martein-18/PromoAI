"""Broken model replies must count as failures, never as a fragment of the answer. Run: python tests/test_json.py
Both broken cases are real replies from run_A."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import extract_json
from ai_analyzer import generate_candidates, score_clip, _scores_ok


class FakeClient:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def call(self, prompt, system=None, task=""):
        self.prompts.append(prompt)
        return self.replies.pop(0)


GOOD_SCORE = ('{"hook": {"score": 4, "reason": "a"}, "cliffhanger": {"score": 3, "reason": "b"}, '
              '"standalone": {"score": 3, "reason": "c"}, "emotional": {"score": 4, "reason": "d"}, "hook_type": "threat"}')


class T(unittest.TestCase):
    def test_fences_and_preamble_still_work(self):
        self.assertEqual(extract_json('Here you go:\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('[{"x": 1}]'), [{"x": 1}])

    def test_broken_candidates_reply_is_a_failure_not_no_promos(self):
        # ep2: missing quote before Juliana. The old parser returned the inner [[1, 30]] -> "no promos".
        raw = '{"candidates": [ { "segments": [[1, 30]], "hook_event": 2, "cliffhanger_event": 30, ' \
              '"hook": "I cannot marry you, Juliana.", "cliffhanger": Juliana\'s father coldly commands her." } ]}'
        with self.assertRaises(ValueError):
            extract_json(raw)
        self.assertIsNone(generate_candidates(None, {}, FakeClient(raw), ep_text="x", default_ep=2))

    def test_empty_list_is_still_a_valid_no_promos(self):
        self.assertEqual(generate_candidates(None, {}, FakeClient('{"candidates": []}'), ep_text="x", default_ep=2), [])

    def test_wrong_shape_is_a_failure(self):
        self.assertIsNone(generate_candidates(None, {}, FakeClient('{"premise": "p"}'), ep_text="x", default_ep=2))
        self.assertIsNone(generate_candidates(None, {}, FakeClient('[[1, 30]]'), ep_text="x", default_ep=2))

    def test_broken_score_is_retried_then_used(self):
        broken = '{"hook": {"score": 3, "reason": "A threat"}, "cliffhanger": {"score": 2, "reason": "x" y}}'
        c = FakeClient(broken, GOOD_SCORE)
        out = score_clip("clip", {}, {}, c)
        self.assertTrue(_scores_ok(out))
        self.assertEqual(len(c.prompts), 2)
        self.assertNotEqual(c.prompts[0], c.prompts[1])          # retry must not be a cache hit

    def test_score_broken_twice_is_marked_not_zeroed(self):
        out = score_clip("clip", {}, {}, FakeClient('{"score": 3, "reason": "only one part"}', "not json"))
        self.assertIn("_parse_error", out)


if __name__ == "__main__":
    unittest.main()
