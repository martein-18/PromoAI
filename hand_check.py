"""Builds outputs/<run>/hand_check.md, a fill-in sheet for judging a run by hand. Run: python hand_check.py run_A"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import load_config
from main import load_episodes
from validator import validate_candidate
from prompting import clip_text

run = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "run_A")
cfg = load_config(str(ROOT / "config.yaml"))
V = cfg["validation"]
eps = {e.ep: e for e in load_episodes(str(ROOT / cfg["paths"]["data_dir"]))}
rd = json.loads((run / "raw_results.json").read_text(encoding="utf-8"))
rk = json.loads((run / "ranked_promos.json").read_text(encoding="utf-8"))["promos"]

VERDICT = ("- Hook works: [ ] yes  [ ] partly  [ ] no\n- Cliffhanger works: [ ] yes  [ ] partly  [ ] no\n"
           "- Would you post it? [ ] yes  [ ] no\n- Notes: \n")


def segs_str(segs):
    return " + ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']} (lines {s['start_id']}-{s['end_id']})" for s in segs)


def scores(p):
    sc = p.get("scores") or {}
    g = lambda k: (sc.get(k) or {}).get("score", "?") if isinstance(sc.get(k), dict) else "?"
    return f"hook {g('hook')} · cliffhanger {g('cliffhanger')} · standalone {g('standalone')} · emotional {g('emotional')}"


def promo_block(p, extra=""):
    return (f"### #{p['rank']} `{p['id']}` · {p['total_duration']}s · final {p.get('final_score')}{extra}\n\n"
            f"**Clip:** {segs_str(p['segments'])}  \n**Model scores:** {scores(p)}  \n"
            f"**Hook (model):** {p.get('hook')}  \n**Cliffhanger (model):** {p.get('cliffhanger')}  \n"
            f"**Why (model):** {p.get('why')}\n\n<details><summary>Exact screenplay lines</summary>\n\n```\n"
            f"{p.get('clip_text', '').strip()}\n```\n</details>\n\n{VERDICT}\n")


out = [f"# Hand-check sheet: {run.name}\n",
       "Judge each clip as a viewer who has never seen the show. Tick one box per question; a short note on *why* is "
       "the useful part for the write-up. Model scores are shown so you can compare your call with the model's.\n",
       f"Scored by: {rk[0].get('scored_by') if rk else '?'} (all promos). Clip generation used fallbacks; see audit.json.\n"]

out.append("\n## 1. Top 10\n")
for p in rk[:10]:
    out.append(promo_block(p))

rep = [p for p in rk if p.get("repair")]
out.append("\n## 2. Auto-repaired promos (judge separately)\n")
for p in rep:
    r = p["repair"]
    out.append(promo_block(p, f" · **auto-adjusted: {r['action']}, {r['before_s']}s -> {r['after_s']}s**"))
    out.append("- Did the added/removed line hurt the clip? [ ] no  [ ] a bit  [ ] yes\n\n")
if not rep:
    out.append("_none in this run_\n")

out.append("\n## 3. Rejected near-misses (should any of these have been kept?)\n")
lo, hi = V["min_duration_s"], V["max_duration_s"]
picked, units = [], set()
for r in rd["rejected"]:
    c = r.get("candidate")
    if not c or r.get("outcome") or not all(i.startswith("[error] duration") for i in r["issues"] if i.startswith("[error]")):
        continue
    unit = r["source"].split("#")[0]
    m = re.search(r"total ([\d.]+)s", r["issues"][0])
    miss = max(lo - float(m.group(1)), float(m.group(1)) - hi)
    if miss <= 6 and unit not in units:
        units.add(unit)
        picked.append((miss, r, c, unit))
for miss, r, c, unit in sorted(picked, key=lambda x: x[0])[:5]:
    dep = int(re.findall(r"\d+", unit)[0])
    segs, _ = validate_candidate(eps, c, dep, V)
    out.append(f"### `{r['source']}` · rejected: {r['issues'][0]}\n\n**Hook (model):** {c.get('hook')}  \n"
               f"**Cliffhanger (model):** {c.get('cliffhanger')}\n\n<details><summary>Exact screenplay lines</summary>\n\n```\n"
               f"{clip_text(eps, segs).strip()}\n```\n</details>\n\n"
               "- Good promo apart from length? [ ] yes  [ ] no\n- Notes: \n\n")

out.append("\n## 4. Episodes with no promo (is 'none' the right answer?)\n")
covered = {s["ep"] for p in rk for s in p["segments"]}
missing = [e for e in sorted(eps) if e not in covered]
rejected_units = {r["source"].split("#")[0] for r in rd["rejected"]}
chosen = [e for e in missing if f"ep{e}" not in rejected_units][:2] + [e for e in missing if f"ep{e}" in rejected_units][:2]
out.append(f"All episodes with no clip at all: {', '.join(map(str, missing))}. Checking {', '.join(map(str, chosen))}.\n")
for e in chosen:
    ep = eps[e]
    rs = [r for r in rd["rejected"] if r["source"].split("#")[0] == f"ep{e}"]
    why = ("the model suggested no promo" if not rs else
           "every suggestion was rejected: " + "; ".join(f"{r['source']}: {r['issues'][0]}" for r in rs))
    out.append(f"### Episode {e}\n\n**Logline:** {ep.logline}  \n**Why no promo:** {why}\n\n"
               "Skim the episode (`data/input`), then:\n"
               "- Is there a moment that would make a good promo? [ ] yes, missed  [ ] no, 'none' is right\n"
               "- If yes, where (rough time / lines): \n- Notes: \n\n")

out.append("\n## 5. Summary (fill in last)\n\n- Top 10: __ of 10 I would post\n- Repaired: __ of __ as good as the rest\n"
           "- Near-misses: __ of 5 should have been kept\n- No-promo episodes: __ of 4 correctly empty\n"
           "- Does any quiet or tender scene rank too high? \n- Would different weights fix what you saw? \n")
(run / "hand_check.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", run / "hand_check.md")
