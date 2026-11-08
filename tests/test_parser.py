import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_DATA = Path(__file__).resolve().parent.parent / "data" / "input"
if not any(_DATA.glob("*.txt")):        # these checks use the real episodes, which are not shipped with the code
    print(f"SKIPPED {Path(__file__).name}: put the episode .txt files in data/input to run it")
    sys.exit(0)
from screenplay_parser import parse_dir
from validator import validate_candidate, is_valid

V = {"min_duration_s": 20, "max_duration_s": 60, "first_beat_max_s": 3.0, "max_segments": 4,
     "snap_end": "last_spoken", "end_pad_s": 0.3}
DATA = Path(__file__).resolve().parent.parent / "data" / "input"


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps = parse_dir(DATA)
        cls.m = {e.ep: e for e in cls.eps}

    def test_loaded_clean(self):
        nums = [e.ep for e in self.eps]
        self.assertEqual(len(nums), len(list(DATA.glob("*.txt"))))   # one episode per screenplay file
        self.assertEqual(nums, sorted(set(nums)))                     # unique and in order
        self.assertTrue({7, 8, 9, 10} <= set(nums))                   # the episodes the other tests rely on
        for e in self.eps:
            self.assertEqual(e.warnings, [], e.warnings)
            self.assertTrue(e.logline and e.continuity)
            self.assertTrue(all(x.end >= x.start for x in e.events))

    def test_ep7_fields(self):
        e = self.m[7]
        d = [x for x in e.events if "scarred beggar" in x.text][0]
        self.assertEqual((d.speaker, d.start, d.end, d.tone), ("Gerald", 51.5, 54.6, "Bitter, Sneering"))
        os_ = [x for x in e.events if x.text == "You look like a servant."][0]
        self.assertEqual(os_.modifiers, ["O.S."])
        vo = [x for x in e.events if x.modifiers == ["V.O."]][0]
        self.assertEqual(vo.speaker, "Gerald")
        self.assertEqual(len(e.scenes), 3)

    def test_validator_accepts_and_rejects(self):
        e = self.m[7]
        a = [x for x in e.events if "scarred beggar" in x.text][0].id
        b = [x for x in e.events if x.text == "And your husband?"][0].id
        segs, iss = validate_candidate(self.m, {"segments": [[a - 1, b + 2]]}, 7, V)
        self.assertTrue(is_valid(iss), [str(i) for i in iss])
        _, iss = validate_candidate(self.m, {"segments": [[a, a + 1]]}, 7, V)
        self.assertFalse(is_valid(iss))                       # too short
        _, iss = validate_candidate(self.m, {"segments": [[9999, 10000]]}, 7, V)
        self.assertFalse(is_valid(iss))                       # bad ids
        _, iss = validate_candidate(self.m, {"segments": [[b, b + 3], [a, a + 2]]}, 7, V)
        self.assertFalse(is_valid(iss))                       # order/overlap


if __name__ == "__main__":
    unittest.main()
