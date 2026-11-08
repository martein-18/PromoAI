"""How consistent is a scoring model? Scores the same clips twice and reports how far the two passes differ.
Use it to choose llm.scoring.model from evidence. Costs 2 calls per clip (plus a retry if a reply is broken).

    python measure_drift.py run_A gemini-3.6-flash 12
    python measure_drift.py run_A meta-llama/llama-3.3-70b-instruct 12 https://openrouter.ai/api/v1 OPENROUTER_API_KEY
"""
import statistics, sys, tempfile
from pathlib import Path
from config import load_config
from llm import ChainClient
from prompting import render
from ai_analyzer import _score_once, SCORE_KEYS
import json

if len(sys.argv) < 3:
    sys.exit(__doc__)
run, model = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
cfg = load_config()
lc = {**cfg["llm"], **(cfg["llm"].get("scoring") or {}), "model": model, "fallbacks": [], "min_interval_s": 3}
if len(sys.argv) > 4:
    lc["base_url"] = sys.argv[4]
if len(sys.argv) > 5:
    lc["api_key_env"] = sys.argv[5]
W = cfg["ranking"]["weights"]
promos = json.loads((Path("outputs") / run / "ranked_promos.json").read_text(encoding="utf-8"))["promos"][:n]
client = ChainClient(lc, Path(tempfile.mkdtemp()))          # fresh cache: both passes really ask the model
gaps, crit = [], {k: [] for k in SCORE_KEYS}
for i, p in enumerate(promos, 1):
    prompt = render("score_prompt", clip=p["clip_text"])
    a = _score_once(prompt, client)
    b = _score_once(prompt + "\n\n(Independent scoring pass 2.)", client)
    if not (a and b):
        print(f"  {p['id']}: unusable reply, skipped")
        continue
    for k in SCORE_KEYS:
        crit[k].append(abs(a[k]["score"] - b[k]["score"]))
    fa, fb = (sum(W[k] * x[k]["score"] for k in W) for x in (a, b))
    gaps.append(abs(fa - fb))
    print(f"  {i:2}/{len(promos)} {p['id']}: final {fa:.2f} vs {fb:.2f}")
if gaps:
    print(f"\n{model} on {len(gaps)} clips:")
    print(f"  final-score gap between two passes: median {statistics.median(gaps):.2f}, max {max(gaps):.2f}, "
          f"identical in {sum(g == 0 for g in gaps)} of {len(gaps)}")
    for k, v in crit.items():
        print(f"  {k:12} same score in {sum(x == 0 for x in v)}/{len(v)}, off by 2+ in {sum(x >= 2 for x in v)}")
    print("  (run_A's gemini-3.5-flash-lite: median gap 0.55, identical in 5 of 50, off by 2+ on some criterion in 26 of 50)")
