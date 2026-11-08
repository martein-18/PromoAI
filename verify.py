"""Audit of a finished run: is it real, is it grounded in the screenplay, did the rules hold?
Writes <run>/audit.json. Nothing here calls a model; it only re-checks files on disk.
  python verify.py outputs/run_A
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
from screenplay_parser import parse_dir
from timeline import resolve_segment

QUOTE = re.compile(r'["“]([^"“”]{4,}?)["”]')


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


def _check(name, status, detail):
    return {"name": name, "status": status, "detail": detail}      # status: pass | warn | fail | info


def audit_run(out_dir, data_dir, cfg):
    out = Path(out_dir)
    ranked_doc = json.loads((out / "ranked_promos.json").read_text(encoding="utf-8"))
    raw_doc = json.loads((out / "raw_results.json").read_text(encoding="utf-8"))
    meta, promos = ranked_doc["meta"], ranked_doc["promos"]
    calls = []
    log = out / "raw" / "calls.jsonl"
    if log.exists():
        calls = [c for c in (json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()) if "task" in c]   # skip "forget" markers
    eps = {e.ep: e for e in parse_dir(data_dir)}
    checks = []

    # 1. real or mock?
    models = Counter(c["model"] for c in calls)
    is_mock = meta.get("provider") == "mock" or "MOCK" in models
    checks.append(_check("Run type", "fail" if is_mock else "pass",
                         "MOCK run: outputs are fake placeholders, not AI results" if is_mock
                         else f"real model run via {meta.get('provider')}"))
    tasks = Counter(c["task"].split(":")[0] for c in calls)
    checks.append(_check("Model calls logged", "pass" if calls else "fail",
                         f"{len(calls)} calls to {dict(models)}; by task {dict(tasks)}"
                         + (f"; {sum(1 for c in calls if c.get('reused'))} of them reused from earlier runs, not new model calls"
                            if any(c.get("reused") for c in calls) else "") if calls
                         else "no calls in raw/calls.jsonl (results may come from cache of an earlier folder or nothing ran)"))
    scorers = Counter(p.get("scored_by") or "unknown" for p in promos)
    if len(scorers) == 1 and "unknown" not in scorers:
        checks.append(_check("Single scoring model", "pass",
                             f"all {len(promos)} promos were scored by {next(iter(scorers))}, so scores are comparable"))
    else:
        checks.append(_check("Single scoring model", "warn",
                             f"promos were scored by more than one model ({dict(scorers)}): scores may not be comparable; "
                             "set llm.scoring in config.yaml and re-run"))
    gen = Counter(c["model"] for c in calls if c["task"].split(":")[0] in ("candidates", "bible"))
    if len(gen) > 1:
        checks.append(_check("Clip generation models", "info",
                             f"candidate clips came from {dict(gen)} (fallbacks); every clip still passed the same code checks"))

    if meta.get("stopped_reason"):
        checks.append(_check("Run complete", "warn", meta.get("status", "partial run")))
    else:
        checks.append(_check("Run complete", "pass", f"all {len(meta.get('episodes', []))} episodes were processed"))

    # 2. reply health
    st = meta.get("stats", {})
    bad = st.get("json_failures", 0) + st.get("score_parse_errors", 0) + (0 if st.get("bible_ok", True) else 1)
    checks.append(_check("Model reply format", "pass" if bad == 0 else "warn",
                         "every reply parsed as JSON" if bad == 0 else
                         f"{st.get('json_failures', 0)} candidate replies and {st.get('score_parse_errors', 0)} score replies were not valid JSON"
                         f"{'; show bible also failed' if not st.get('bible_ok', True) else ''}"))

    # 3. timestamps recomputed from the screenplay files
    v = cfg["validation"]
    mism, rule_bad, grounding = [], [], []
    beats_bad, beats_checked = [], 0
    per_promo = {}
    for p in promos:
        notes = []
        total = 0.0
        for s in p["segments"]:
            ep = eps.get(s["ep"])
            if ep is None:
                mism.append(f"{p['id']}: episode {s['ep']} not in data folder")
                continue
            seg, err = resolve_segment(ep, s["start_id"], s["end_id"], v.get("snap_end", "last_spoken"), v.get("end_pad_s", 0.3))
            if err or abs(seg.start - s["start"]) > 0.01 or abs(seg.end - s["end"]) > 0.01:
                mism.append(f"{p['id']}: stored {s['start']}-{s['end']} vs screenplay {None if err else (seg.start, seg.end)}")
            if s["end"] > ep.duration + 0.5:
                mism.append(f"{p['id']}: ends after episode end")
            total += s["end"] - s["start"]
        if not v["min_duration_s"] <= total <= v["max_duration_s"]:
            rule_bad.append(f"{p['id']}: {total:.1f}s")
        # 4. hook / cliffhanger quotes must exist INSIDE the clip (its own lines, spoken or action), not just anywhere
        #    in the episode: a hook quoted from outside the clip is a viewer promise the clip does not keep
        dlg = " ".join(_norm(e.text) for s in p["segments"] if s["ep"] in eps for e in eps[s["ep"]].events
                       if s["start_id"] <= e.id <= s["end_id"] and e.type != "CAMERA")
        for field in ("hook", "cliffhanger", "why"):
            for q in QUOTE.findall(p.get(field) or ""):
                found = _norm(q) in dlg
                grounding.append(found)
                if not found:
                    notes.append(f"{field} quotes \"{q[:50]}\" which is not inside the clip")
        # 5. hook opens the clip, cliffhanger closes it: recomputed from the raw screenplay times and the STORED
        #    clip times, independent of the validator's code path
        first, last = p["segments"][0], p["segments"][-1]
        hev = eps[first["ep"]].event(p["hook_event"]) if first["ep"] in eps and isinstance(p.get("hook_event"), int) else None
        cev = eps[last["ep"]].event(p["cliffhanger_event"]) if last["ep"] in eps and isinstance(p.get("cliffhanger_event"), int) else None
        if hev is not None and not (first["start_id"] <= hev.id <= first["end_id"] and hev.start - first["start"] <= v.get("first_beat_max_s", 3.0) + 0.01):
            beats_bad.append(f"{p['id']}: hook line {hev.id} is {hev.start - first['start']:.1f}s in or outside the clip")
        if cev is not None and not (last["start_id"] <= cev.id <= last["end_id"] and last["end"] - cev.end <= v.get("cliff_tail_max_s", 4.0) + 0.01):
            beats_bad.append(f"{p['id']}: clip runs {last['end'] - cev.end:.1f}s past the cliffhanger line {cev.id}")
        beats_checked += (hev is not None) + (cev is not None)
        per_promo[p["id"]] = notes
    checks.append(_check("Timestamps match screenplay", "pass" if not mism else "fail",
                         f"all {len(promos)} promos re-derived from the original files match" if not mism else "; ".join(mism[:4])))
    checks.append(_check("Duration rule 20-60s", "pass" if not rule_bad else "fail",
                         "all promos within range" if not rule_bad else "; ".join(rule_bad[:4])))
    checks.append(_check("Hook opens, cliffhanger closes", "pass" if not beats_bad else "fail",
                         f"{beats_checked} hook/cliffhanger positions re-checked against the screenplay: hook within "
                         f"{v.get('first_beat_max_s', 3.0)}s of the start, at most {v.get('cliff_tail_max_s', 4.0)}s after "
                         "the cliffhanger" if not beats_bad else "; ".join(beats_bad[:4])))
    if grounding:
        ok = sum(grounding)
        checks.append(_check("Quoted lines exist in script", "pass" if ok == len(grounding) else "warn",
                             f"{ok} of {len(grounding)} quoted lines found verbatim; the rest may be invented or paraphrased"))
    else:
        checks.append(_check("Quoted lines exist in script", "info", "no direct quotes in hook/cliffhanger text to check"))
    multi = [p for p in promos if ((p.get("scores") or {}).get("_passes") or 1) > 1]
    if multi:
        agreed = sum(1 for p in multi if p["scores"].get("_agreed"))
        third = sum(1 for p in multi if p["scores"].get("_passes", 0) >= 3)
        checks.append(_check("Score stability", "pass" if agreed >= 0.8 * len(multi) else "warn",
                             f"{agreed} of {len(multi)} promos got identical scores on every scoring pass; {third} needed a "
                             f"3rd pass and use the mean (scoring drift is averaged, not hidden: see scores.<criterion>.passes)"))
    cmp_ = meta.get("comparison")
    if cmp_:
        moved = sum(1 for p in promos if p.get("score_rank") and p["score_rank"] != p["rank"])
        ag = cmp_.get("agreement")
        checks.append(_check("Head-to-head ranking", "warn" if ag is not None and ag < 0.5 else "info",
                             f"top {cmp_.get('top_n')} reordered by direct comparison ({cmp_.get('passes')} passes); "
                             f"{moved} promos changed place vs. score order; agreement between the forward and reversed "
                             f"passes: {ag if ag is not None else 'n/a (one pass only)'} (1 = identical, 0 = unrelated, -1 = opposite: the model followed list position, not quality)"))
    cost = meta.get("cost")
    if cost:
        n = max(cost.get("episodes", 1), 1)
        checks.append(_check("Run cost", "info",
                             f"{cost['new_model_calls']} new model calls ({cost['new_model_calls'] / n:.1f} per episode), "
                             f"{cost['model_seconds']:.0f}s waiting for the model, {cost['wall_seconds']:.0f}s in total for "
                             f"{cost['episodes']} episodes; saved answers are reused for free, so a re-run repeats nothing"))
    review = [p for p in promos if p.get("review")]
    checks.append(_check("Needs review", "info", f"{len(review)} of {len(promos)} promos are flagged for a human look "
                         "(see needs_review in the CSV)" if review else "no promo is flagged for review"))
    rep = [p for p in promos if p.get("repair")]
    checks.append(_check("Auto-repaired durations", "info" if not rep else "warn",
                         "no promo needed a length repair" if not rep else
                         f"{len(rep)} of {len(promos)} promos were length-repaired by code (+/-1 line); hand-check them "
                         f"separately: {', '.join(p['id'] for p in rep)}"
                         + (f"; {sum(p['repair'].get('extended_past_cliffhanger', False) for p in rep)} extend past the cliffhanger"
                            if any(p['repair'].get('extended_past_cliffhanger') for p in rep) else "")))
    checks.append(_check("Candidates vs ranked", "info",
                         f"{len(raw_doc.get('candidates', []))} valid candidates, {len(promos)} after overlap removal, "
                         f"{len(raw_doc.get('rejected', []))} rejected by validators"))
    result = {"status": "MOCK RUN" if is_mock else "REAL RUN", "checks": checks, "per_promo_notes": per_promo,
              "models": dict(models)}
    (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    from config import load_config
    cfg = load_config("config.yaml")
    r = audit_run(sys.argv[1], cfg["paths"]["data_dir"], cfg)
    print(r["status"])
    for c in r["checks"]:
        print(f"[{c['status'].upper():4}] {c['name']}: {c['detail']}")