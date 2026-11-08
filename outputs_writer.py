import json
from pathlib import Path
from promo_scorer import to_dataframe


def write_all(candidates, ranked, rejected, out_dir, meta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # unedited model-derived results, before ranking/dedupe
    (out / "raw_results.json").write_text(
        json.dumps({"meta": meta, "candidates": candidates, "rejected": rejected}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out / "ranked_promos.json").write_text(
        json.dumps({"meta": meta, "promos": ranked}, indent=2, ensure_ascii=False), encoding="utf-8")
    to_dataframe(ranked).to_csv(out / "ranked_promos.csv", index=False, encoding="utf-8-sig")
    lines = ["# Promo report", "", f"Model: `{meta.get('model')}` | ranked: {len(ranked)} | rejected: {len(rejected)}", ""]
    for p in ranked:
        segs = " | ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        lines += [f"## #{p['rank']}  {segs}  ({p['total_duration']}s, score {p.get('final_score')})",
                  f"- **Hook:** {p.get('hook')}", f"- **Cliffhanger:** {p.get('cliffhanger')}",
                  f"- **Why it works:** {p.get('why')}", f"- **Why this rank:** {p.get('post_reason', '')}",
                  f"- **Scores:** `{json.dumps(p.get('scores'), ensure_ascii=False)}`",
                  f"- **Cold viewer:** `{json.dumps(p.get('cold_viewer'), ensure_ascii=False)}`", "", "```", p.get("clip_text", ""), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")