"""Copies a finished run into demo/, the example the web page shows instantly (no AI calls). Its saved answers are
also reused, so uploading the same episodes again costs nothing.   Run: python make_demo.py run_A"""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "run_A")
need = ["ranked_promos.json", "raw_results.json", "audit.json"]
missing = [n for n in need if not (src / n).exists()]
if missing:
    sys.exit(f"{src} is not a finished run (missing {', '.join(missing)}). Run the pipeline first.")
dst = ROOT / "demo"
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir()
for n in need + ["ranked_promos.csv", "report.md", "dashboard.html", "posting_plan.md", "PROMOS.md", "hand_check.md"]:
    if (src / n).exists():
        shutil.copy2(src / n, dst / n)
shutil.copytree(src / "parsed", dst / "parsed")
(dst / "raw").mkdir()
shutil.copy2(src / "raw" / "calls.jsonl", dst / "raw" / "calls.jsonl")
size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e6
print(f"demo/ ready from {src.name}: {len(list((dst / 'parsed').glob('*.json')))} episodes, {size:.1f} MB")
