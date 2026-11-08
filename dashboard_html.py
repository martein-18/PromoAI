"""Self-contained presentation page (dashboard.html) built from a finished run folder. No server, no internet."""
import json
from html import escape as E
from pathlib import Path

CSS = """
:root{--bg:#0f1220;--card:#181c30;--line:#2a3050;--tx:#e8ebf7;--mut:#9aa3c7;--acc:#7c8cff;--ok:#3ecf8e;--warn:#f5b84a;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px 80px}h1{font-size:28px;margin:0 0 4px}h2{margin:34px 0 12px;font-size:20px}
.sub{color:var(--mut)}.banner{padding:12px 16px;border-radius:10px;margin:14px 0;font-weight:600}
.mock{background:#3a1d24;border:1px solid var(--bad);color:#ffb3b3}.real{background:#13302a;border:1px solid var(--ok);color:#a8f0cf}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}.kpi b{font-size:24px;display:block}.kpi span{color:var(--mut);font-size:13px}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.step{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:13px}
.step i{display:block;color:var(--mut);font-style:normal;font-size:11px}.step.ai{border-color:var(--acc)}.arrow{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}
.card h3{margin:0 0 4px;font-size:18px}.rank{display:inline-block;background:var(--acc);color:#0b0e1c;border-radius:8px;padding:2px 10px;font-weight:700;margin-right:8px}
.meta{color:var(--mut);font-size:13px;margin-bottom:10px}.row{margin:6px 0}.row b{color:var(--acc)}
.tl{position:relative;height:12px;background:#0b0e1c;border-radius:6px;margin:10px 0 4px;border:1px solid var(--line)}
.tl span{position:absolute;top:0;height:100%;background:var(--acc);border-radius:6px;opacity:.95}.tl.s2 span:nth-child(2){background:#ff9f6b}.tl.s3 span:nth-child(3){background:var(--ok)}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px 20px;margin:12px 0}
.bar{height:8px;background:#0b0e1c;border-radius:4px;overflow:hidden;border:1px solid var(--line)}.bar div{height:100%;background:var(--ok)}
.lbl{display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--mut);margin-right:6px}
.tag.warn{color:var(--warn);border-color:var(--warn)}.tag.bad{color:var(--bad);border-color:var(--bad)}.tag.ok{color:var(--ok);border-color:var(--ok)}
details{margin-top:10px}summary{cursor:pointer;color:var(--mut)}pre{white-space:pre-wrap;background:#0b0e1c;border:1px solid var(--line);border-radius:10px;padding:12px;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{color:var(--mut)}
.chk{display:grid;grid-template-columns:70px 220px 1fr;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}
.note{color:var(--mut);font-size:13px}
"""


def _ts(p):
    return " + ".join(f"Ep {s['ep']} · {s['start_ts']}–{s['end_ts']}" for s in p["segments"])


def _bar(label, v, mx=5):
    if not isinstance(v, (int, float)):
        return f'<div><div class="lbl"><span>{E(label)}</span><span>n/a</span></div><div class="bar"></div></div>'
    return (f'<div><div class="lbl"><span>{E(label)}</span><span>{v}/{mx}</span></div>'
            f'<div class="bar"><div style="width:{max(0, min(100, v / mx * 100)):.0f}%"></div></div></div>')


def _sc(p, k):
    v = (p.get("scores") or {}).get(k)
    return v.get("score") if isinstance(v, dict) else None


def _reason(p, k):
    v = (p.get("scores") or {}).get(k)
    return v.get("reason", "") if isinstance(v, dict) else ""


def load_run(out_dir):
    out = Path(out_dir)
    doc = json.loads((out / "ranked_promos.json").read_text(encoding="utf-8"))
    raw = json.loads((out / "raw_results.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8")) if (out / "audit.json").exists() else {}
    durations, events = {}, 0
    for f in sorted((out / "parsed").glob("ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        durations[d["ep"]] = d["duration"]
        events += len(d["events"])
    calls = []
    log = out / "raw" / "calls.jsonl"
    if log.exists():
        calls = [c for c in (json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()) if "task" in c]   # skip "forget" markers
    return {"meta": doc["meta"], "promos": doc["promos"], "rejected": raw.get("rejected", []),
            "candidates": raw.get("candidates", []), "audit": audit, "durations": durations, "events": events, "calls": calls}


def _card(p, durations, audit):
    tl = "".join(f'<span style="left:{s["start"] / max(durations.get(s["ep"], 1), 1) * 100:.1f}%;'
                 f'width:{max((s["end"] - s["start"]) / max(durations.get(s["ep"], 1), 1) * 100, 1.2):.1f}%"></span>'
                 for s in p["segments"])
    cv = p.get("cold_viewer") or {}
    cvtag = ('<span class="tag ok">cold viewer understood it</span>' if cv.get("can_answer") is True else
             '<span class="tag bad">cold viewer confused</span>' if cv.get("can_answer") is False else "")
    notes = (audit.get("per_promo_notes") or {}).get(p["id"], []) + p.get("warnings", [])
    wtag = "".join(f'<span class="tag warn">{E(n[:140])}</span>' for n in notes)
    rp = p.get("repair")
    if rp:
        wtag = (f'<span class="tag bad">auto-adjusted: {E(rp["action"])} ({rp["delta_s"]:+}s)</span>') + wtag
    hook_type = (p.get("scores") or {}).get("hook_type", "")
    return f"""<div class="card"><h3><span class="rank">#{p['rank']}</span>{E(_ts(p))}</h3>
<div class="meta">{p['total_duration']}s · final score {p.get('final_score')} · <span class="tag">{E(str(hook_type))}</span>{cvtag}{wtag}</div>
<div class="tl s{len(p['segments'])}" title="position in the episode">{tl}</div><div class="note">where the clip sits in the episode</div>
<div class="row"><b>Hook:</b> {E(p.get('hook') or '')}</div><div class="row"><b>Cliffhanger:</b> {E(p.get('cliffhanger') or '')}</div>
<div class="row"><b>Why it works:</b> {E(p.get('why') or '')}</div>
{f'<div class="row"><b>Why this rank:</b> {E(p["post_reason"])}</div>' if p.get('post_reason') else ''}
<div class="scores">{_bar('Hook', _sc(p, 'hook'))}{_bar('Cliffhanger', _sc(p, 'cliffhanger'))}{_bar('Standalone', _sc(p, 'standalone'))}{_bar('Emotional', _sc(p, 'emotional'))}</div>
<details><summary>Score reasons and cold-viewer answers</summary><div class="note">
Hook: {E(_reason(p, 'hook'))}<br>Cliffhanger: {E(_reason(p, 'cliffhanger'))}<br>Standalone: {E(_reason(p, 'standalone'))}<br>Emotional: {E(_reason(p, 'emotional'))}<br><br>
Cold viewer, characters: {E(str(cv.get('characters', '')))}<br>Conflict: {E(str(cv.get('conflict', '')))}<br>Wants next: {E(str(cv.get('next', '')))}</div></details>
<details><summary>Exact screenplay lines in this clip</summary><pre>{E(p.get('clip_text', ''))}</pre></details></div>"""


def build_dashboard(out_dir):
    r = load_run(out_dir)
    m, promos, aud = r["meta"], r["promos"], r["audit"]
    is_mock = aud.get("status") == "MOCK RUN" or m.get("provider") == "mock"
    banner = ('<div class="banner mock">MOCK RUN: placeholder output from a fake model. Not real AI results.</div>' if is_mock else
              f'<div class="banner real">REAL RUN: answers produced by {E(", ".join(m.get("models_used") or [m.get("model", "?")]))}</div>')
    tasks = {}
    for c in r["calls"]:
        tasks[c["task"].split(":")[0]] = tasks.get(c["task"].split(":")[0], 0) + 1
    kpis = [("Episodes", len(m.get("episodes", []))), ("Events parsed", r["events"]), ("LLM calls", len(r["calls"])),
            ("Valid candidates", len(r["candidates"])), ("Rejected by rules", len(r["rejected"])), ("Ranked promos", len(promos))]
    flow = [("Parse", "code", ""), ("Show bible", "LLM", f"{tasks.get('bible', 0)} call"), ("Find candidates", "LLM", f"{tasks.get('candidates', 0)} calls"),
            ("Validate", "code", f"{len(r['rejected'])} rejected"), ("Score + cold viewer", "LLM", f"{tasks.get('score', 0) + tasks.get('cold_viewer', 0)} calls"),
            ("Rank + dedupe", "code", ""), ("Explain ranking", "LLM", f"{tasks.get('rank_explain', 0)} call")]
    flow_html = '<span class="arrow">→</span>'.join(f'<div class="step {"ai" if k == "LLM" else ""}"><b>{E(n)}</b><i>{E(k)} {E(x)}</i></div>' for n, k, x in flow)
    top3 = "".join(_card(p, r["durations"], aud) for p in promos[:3])
    rest = "".join(_card(p, r["durations"], aud) for p in promos[3:])
    rej = "".join(f"<tr><td>{E(str(x.get('source')))}</td><td>{E(' · '.join(x.get('issues', [])))}</td></tr>" for x in r["rejected"]) \
        or '<tr><td colspan="2">none</td></tr>'
    chk = "".join(f'<div class="chk"><span class="tag {"ok" if c["status"] == "pass" else "bad" if c["status"] == "fail" else "warn" if c["status"] == "warn" else ""}">{E(c["status"].upper())}</span>'
                  f'<b>{E(c["name"])}</b><span>{E(c["detail"])}</span></div>' for c in aud.get("checks", []))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PromoAI results</title><style>{CSS}</style></head><body><div class="wrap">
<h1>PromoAI · promo finder</h1><div class="sub">Run {E(str(m.get('run_time', '')))} · model {E(str(m.get('model')))} · weights {E(json.dumps(m.get('weights')))}</div>
{banner}<div class="kpis">{''.join(f'<div class="kpi"><b>{v}</b><span>{E(k)}</span></div>' for k, v in kpis)}</div>
<h2>How it works</h2><div class="flow">{flow_html}</div>
<div class="note" style="margin-top:8px">Code does parsing, timestamps and rule checks. The model only makes judgement calls (what is a hook, what is unresolved) and refers to lines by ID, never by time.</div>
{f'<h2>Post first</h2><div class="card"><b>{E(m.get("ranking_summary", ""))}</b><div class="note">Model rationale for the ordering, not a human decision.</div></div>' if m.get('ranking_summary') else ''}
<h2>Top 3</h2>{top3 or '<div class="card">No valid promos in this run.</div>'}
{f'<h2>Remaining ranked promos</h2>{rest}' if rest else ''}
<h2>Where the tool rejected its own picks</h2><div class="card"><table><tr><th>Source</th><th>Why the validators rejected it</th></tr>{rej}</table></div>
<h2>Is this real? Audit of this run</h2><div class="card">{chk}</div>
<h2>Known limits</h2><div class="card note">The tool reads text only, so it cannot see acting, visuals or pacing. Hooks and cliffhangers are inferred from dialogue and action lines.
Scores are a model's opinion until compared with human labels. Weights are proposed, not specified by Kuku.</div>
</div></body></html>"""
    path = Path(out_dir) / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    print(build_dashboard(sys.argv[1]))