"""Pipeline.  python main.py parse | all  [--mock] [--stitch] [--episodes 7 9] [--out outputs/run_B]
Also importable: run_pipeline(...) is what app.py calls."""
import argparse, json, re, sys, time
from pathlib import Path
from config import load_config
from screenplay_parser import parse_dir, EpisodeFileError
from llm import make_client, make_scoring_client, LLMError
from ai_analyzer import assemble_candidates, build_bible, load_bible, generate_candidates, score_clip, cold_viewer, explain_ranking, compare_top
from validator import validate_candidate, is_valid
from repair import try_repair
from prompting import clip_text
from promo_scorer import rank, apply_comparison, compute_final_score
from timeline import normalize_spec
from outputs_writer import write_all
from stitch import find_links, pair_text
from slate import build_slate, slate_markdown
from deliverable import build as build_deliverable
from verify import audit_run
from dashboard_html import build_dashboard


def load_episodes(data_dir, episodes=None):
    eps = parse_dir(data_dir)
    return [e for e in eps if not episodes or e.ep in episodes]


def pass_gap(sc, weights):
    """How far apart the repeat scoring passes were, as a final score (0-5). 0 when scored once."""
    keys = [k for k in weights if isinstance(sc.get(k), dict) and sc[k].get("passes")]
    if not keys:
        return 0.0
    n = min(len(sc[k]["passes"]) for k in keys)
    totals = [sum(weights[k] * sc[k]["passes"][i] for k in keys) for i in range(n)]
    return round(max(totals) - min(totals), 2)


def rank_episodes(eps, promos, rejected, ranked, done_eps, cfg):
    """Episodes ordered by where their best promo landed in the FINAL ranking (after the head-to-head comparison).
    Potential = the best AI score any of the episode's checked clips received. Pure aggregation, nothing hand-set."""
    rank_of = {p["id"]: p["rank"] for p in ranked}
    rows = []
    for e in eps:
        mine = [p for p in promos if {s["ep"] for s in p["segments"]} == {e.ep}]
        rej = [r for r in rejected if r["source"].split("#")[0] == f"ep{e.ep}"]
        if mine:
            top = max(mine, key=lambda p: compute_final_score(p, cfg))          # highest score -> potential
            in_list = [p for p in mine if p["id"] in rank_of]
            best = min(in_list, key=lambda p: rank_of[p["id"]]) if in_list else top   # highest-ranked promo
            sc = best.get("scores") or {}
            g = lambda k: (sc.get(k) or {}).get("score") if isinstance(sc.get(k), dict) else None
            rows.append({"ep": e.ep, "logline": e.logline, "status": "ok", "potential": round(compute_final_score(top, cfg), 2),
                         "clips_passed": len(mine), "clips_rejected": len(rej), "best_promo": best["id"],
                         "best_promo_rank": rank_of.get(best["id"]), "best_hook": best.get("hook", ""),
                         "hook": g("hook"), "cliffhanger": g("cliffhanger"), "standalone": g("standalone"), "emotional": g("emotional")})
        else:
            if e.ep not in done_eps:
                why = "not processed: the run stopped (AI quota) before this episode"
            elif rej:
                codes = sorted({i.split("]", 1)[1].split(":", 1)[0].strip() for r in rej for i in r["issues"] if "]" in i})
                why = f"all {len(rej)} AI picks broke a rule ({', '.join(codes)})"
            else:
                why = "the AI found no unresolved, promo-worthy moment"
            rows.append({"ep": e.ep, "logline": e.logline, "status": "not_processed" if e.ep not in done_eps else "none",
                         "potential": None, "clips_passed": 0, "clips_rejected": len(rej), "why": why})
    # best final rank first; then episodes whose clips were all dropped as overlaps (by potential); then no promo
    rows.sort(key=lambda r: (r["potential"] is None, r.get("best_promo_rank") is None,
                             r.get("best_promo_rank") or 0, -(r["potential"] or 0), r["ep"]))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    return rows


def run_pipeline(cfg, data_dir, out_dir, episodes=None, mock=False, stitch=False, progress=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    progress("reading episodes")
    eps = load_episodes(data_dir, episodes)
    ep_map = {e.ep: e for e in eps}
    (out / "parsed").mkdir(exist_ok=True)
    for e in eps:
        (out / "parsed" / f"ep{e.ep:03d}.json").write_text(json.dumps(e.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    progress(f"read {len(eps)} episodes, {sum(len(e.events) for e in eps)} script lines")
    client = make_client(cfg, out / "raw", force_mock=mock)
    scorer = make_scoring_client(cfg, out / "raw", client, force_mock=mock)   # one fixed model for all scores
    score_samples = 1 if scorer is client else int((cfg["llm"].get("scoring") or {}).get("samples", 1))
    bible_path = out / "show_bible.json"
    progress("building show bible")
    bible = build_bible(eps, client, bible_path)
    stats = {"json_failures": 0, "score_parse_errors": 0, "auto_repaired": 0, "bible_ok": "_parse_error" not in bible}
    if not stats["bible_ok"]:
        progress("WARNING: show bible reply was not valid JSON; continuing without it")

    pc, vc = cfg["pipeline"], cfg["validation"]
    promos, rejected = [], []

    def locate_strongest(p, segs):
        """Second hook check that does not trust the picker: the BLIND scorer quotes the line that would most stop
        a scroll; if it sits later than first_beat_max_s into the clip, the real hook may be buried."""
        norm = lambda t: " ".join(re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split())
        gl = norm((p.get("scores") or {}).get("grab_line", ""))[:60]
        if len(gl) < 8:
            return
        cum = 0.0
        for s in segs:
            for e in sorted((e for e in ep_map[s.ep].events if s.start_id <= e.id <= s.end_id), key=lambda e: e.start):
                if gl in norm(e.text) or (len(norm(e.text)) >= 8 and norm(e.text) in gl):
                    p["strongest_at_s"] = round(cum + max(0.0, e.start - s.start), 1)
                    return
            cum += s.duration

    def add_promo(c, segs, issues, fix, source):
        """Score a validated clip and add it to promos (shared by every stage)."""
        text = clip_text(ep_map, segs)                     # repaired clips are scored on their FINAL text
        p = {"id": f"{source}-{len(promos)+1}", "source": source, "segments": [s.to_dict() for s in segs],
             "total_duration": round(sum(s.duration for s in segs), 2), "hook": c.get("hook", ""),
             "cliffhanger": c.get("cliffhanger", ""), "why": c.get("why", ""), "confidence": c.get("confidence"),
             "hook_event": c.get("hook_event"), "cliffhanger_event": c.get("cliffhanger_event"),
             "warnings": [str(i) for i in issues], "clip_text": text, "scores": score_clip(text, c, bible, scorer, score_samples)}
        p["scored_by"] = scorer.last_model
        if fix:
            p["repair"] = fix
            p["warnings"].insert(0, f"[warn] auto_adjusted: {fix['action']}, {fix['before_s']}s -> {fix['after_s']}s"
                                    f" ({fix['delta_s']:+}s)")
        locate_strongest(p, segs)
        if pc.get("run_cold_viewer", True):
            p["cold_viewer"] = cold_viewer(text, scorer)
        for name in ("scores", "cold_viewer"):
            if "_parse_error" in (p.get(name) or {}):
                stats["score_parse_errors"] += 1
                p["warnings"].append(f"{name}_parse_error: model reply was not valid JSON, this promo is under-scored")
        promos.append(p)

    def run_unit(ep, default_ep, ep_text, source):
        feedback, accepted, reserve = "", [], []
        for attempt in range(pc["max_retries"] + 1):
            cands = generate_candidates(ep, bible, client, pc["candidates_per_episode"], feedback,
                                        pc["include_camera_in_prompt"], ep_text, default_ep)
            accepted, start_rej = [], len(rejected)
            if cands is None:                                  # unparseable reply: a failure, not "no promos"
                stats["json_failures"] += 1
                rejected.append({"source": f"{source}#try{attempt}", "candidate": None,
                                 "issues": ["[error] model_reply_not_json: reply could not be parsed as JSON"]})
                feedback = ((f"Attempt {attempt + 2}: " if attempt else "")      # 3rd+ attempt must differ from the 2nd
                            + "Your previous reply was not valid JSON. Return ONLY the JSON object, nothing else.")
                continue
            for c in cands:
                segs, issues = validate_candidate(ep_map, c, default_ep, vc)
                if is_valid(issues):
                    accepted.append((c, segs, issues, None))
                    continue
                rec = {"source": f"{source}#try{attempt}", "candidate": c, "issues": [str(i) for i in issues]}
                fix = try_repair(ep_map, c, default_ep, vc, segs, issues, pc.get("duration_repair"))
                if fix:                                # kept in reserve; strict picks from later retries win
                    reserve.append((fix, rec))
                rejected.append(rec)
            if accepted or not cands:
                break   # got valid promos, or the model returned none (legitimate for quiet episodes)
            feedback = ((f"Attempt {attempt + 2}: " if attempt else "")          # 3rd+ attempt must differ from the 2nd
                        + "Your previous attempt was rejected: " + "; ".join(r["issues"][0] for r in rejected[start_rej:][:3]))
        if not accepted and reserve:                   # every retry failed the strict check: fall back to repairs
            seen = set()
            for (c2, segs2, issues2, info), rec in reserve:
                key = tuple((s.ep, s.start_id, s.end_id) for s in segs2)
                if key in seen:
                    rec["outcome"] = "repairable, but the same repaired clip was already used"
                    continue
                seen.add(key)
                rec["outcome"] = f"auto-repaired: {info['action']} ({info['before_s']}s -> {info['after_s']}s)"
                stats["auto_repaired"] += 1
                accepted.append((c2, segs2, issues2, info))
        for c, segs, issues, fix in accepted:
            add_promo(c, segs, issues, fix, source)

    def assemble_stage(asm):
        """Cross-episode promos built ONLY from parts of single-episode clips that already passed every rule, e.g. a
        hook from episode 4 + a cliffhanger from episode 11. New prompt, so no saved answer is invalidated."""
        pool = [p for p in promos if len({s["ep"] for s in p["segments"]}) == 1 and p["source"].startswith("ep")]
        pool = sorted(pool, key=lambda p: compute_final_score(p, cfg), reverse=True)[:asm.get("pool", 15)]
        if len({p["segments"][0]["ep"] for p in pool}) < 2:
            return
        for c in assemble_candidates(pool, ep_map, client, asm.get("max_promos", 3)) or []:
            specs = [normalize_spec(x, 0) for x in c.get("segments") or []]
            first_ep = specs[0][0] if specs and specs[0] else 0
            segs, issues = validate_candidate(ep_map, c, first_ep, vc)
            errs = [str(i) for i in issues if i.severity == "error"]
            eps_used = {sp[0] for sp in specs if sp}
            if len(eps_used) < 2:
                errs.append("[error] not_cross_episode: all parts come from one episode")
            for sp in specs:
                if sp and not any(s["ep"] == sp[0] and s["start_id"] <= sp[1] and sp[2] <= s["end_id"]
                                  for p in pool for s in p["segments"]):
                    errs.append(f"[error] part_not_from_checked_clip: {list(sp)} is not inside a validated clip")
            if errs:
                rejected.append({"source": "assembled#try0", "candidate": c, "issues": errs})
            else:
                add_promo(c, segs, issues, None, "assembled")

    # If the AI quota runs out mid-run, keep everything finished so far and write it, marked partial.
    # Re-running the same command later continues and completes it (answers are cached).
    stopped, done_eps = None, []
    try:
        for i, e in enumerate(eps, 1):
            progress(f"episode {e.ep} ({i}/{len(eps)})")
            run_unit(e, e.ep, None, f"ep{e.ep}")
            done_eps.append(e.ep)
        if stitch:
            for n, m, strength, why in find_links(eps):
                if n in ep_map and m in ep_map:
                    progress(f"stitching {n}->{m}")
                    run_unit(None, n, pair_text(ep_map[n], ep_map[m]), f"stitch{n}-{m}")
        asm = pc.get("assemble") or {}
        if asm.get("enabled"):
            progress("assembling cross-episode promos")
            assemble_stage(asm)
    except LLMError as e:
        stopped = str(e)
        progress(f"WARNING: stopped early ({stopped[:120]}); writing partial results for "
                 f"{len(done_eps)} of {len(eps)} episodes")

    progress("ranking")
    ranked = rank([dict(p) for p in promos], cfg)
    comparison = {}
    cmp_cfg = cfg["ranking"].get("compare") or {}
    if ranked and cmp_cfg.get("enabled") and not stopped:
        progress("comparing the top clips head to head")
        comparison = compare_top(ranked[:cmp_cfg.get("top_n", 12)], scorer)
        ranked = apply_comparison(ranked, comparison)
        if not comparison:
            progress("WARNING: head-to-head comparison reply was unusable; kept the score order")
    # needs-review flag: the honest "editor reviews a shortlist" step. Uncertain promos say why.
    for p in ranked:
        why = []
        sc = p.get("scores") or {}
        gap = pass_gap(sc, cfg["ranking"]["weights"])
        p["score_gap"] = gap
        if gap >= cfg["ranking"].get("review_gap", 0.5):       # small +/-1 wobble is averaged out, not flagged
            why.append(f"scores varied by {gap:.2f} between scoring passes")
        if "_parse_error" in sc:
            why.append("scores missing")
        if p.get("repair"):
            why.append("length auto-adjusted")
        if scorer is not client and p.get("scored_by") and p["scored_by"] != scorer.model:
            why.append(f"scored by backup model {p['scored_by']} (main scorer unavailable)")
        if any("unverified" in w for w in p.get("warnings", [])):
            why.append("hook or cliffhanger position not verified")
        at = p.get("strongest_at_s") or 0
        if at > vc.get("first_beat_max_s", 3.0) and p["total_duration"] - at > 8:   # the ending itself is fine
            why.append(f"strongest line is {p['strongest_at_s']:.0f}s in, not in the first {vc.get('first_beat_max_s', 3.0):.0f}s")
        if p.get("score_rank") and abs(p["score_rank"] - p["rank"]) >= 3:
            why.append(f"head-to-head moved it {p['score_rank'] - p['rank']:+d} places")
        p["review"] = why
    summary = {}
    if ranked and not stopped:
        progress("explaining the ranking")
        summary = explain_ranking(ranked[:10], scorer)
        for p in ranked[:10]:
            p["post_reason"] = (summary.get("items") or {}).get(p["rank"], "")
    meta = {"model": "MOCK" if mock else cfg["llm"]["model"], "provider": "mock" if mock else cfg["llm"]["provider"],
            "episodes": [e.ep for e in eps], "weights": cfg["ranking"]["weights"],
            "models_used": sorted(client.models_used | scorer.models_used), "stats": stats,
            "scoring_model": None if scorer is client else scorer.model,
            "ranking_summary": summary.get("overall", ""),
            "comparison": {k: v for k, v in comparison.items() if k != "order"} | {"top_n": len(comparison.get("order", []))}
                          if comparison else None,
            "score_samples": score_samples,
            "cost": {"new_model_calls": client.new_calls + (0 if scorer is client else scorer.new_calls),
                     "model_seconds": round(client.model_seconds + (0 if scorer is client else scorer.model_seconds), 1),
                     "wall_seconds": round(time.time() - t_start, 1), "episodes": len(eps)},
            "run_time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": ("complete" if not stopped else
                       f"PARTIAL: stopped ({'daily cap in config.yaml' if 'daily cap' in stopped else 'AI quota'}) after {len(done_eps)} of {len(eps)} episodes "
                       f"(head-to-head and explanations skipped). Run it again with the same episodes to complete it: "
                       f"finished work is reused, nothing is repeated."),
            "stopped_reason": stopped,
            "episodes_done": done_eps,
            "episode_ranking": rank_episodes(eps, promos, rejected, ranked, done_eps, cfg)}
    meta["slate"], meta["slate_left_out"] = build_slate(ranked, cfg)
    write_all(promos, ranked, rejected, out, meta)
    (out / "posting_plan.md").write_text(slate_markdown(meta["slate"], meta["slate_left_out"]), encoding="utf-8")
    (out / "PROMOS.md").write_text(build_deliverable(ranked, promos, rejected, meta, cfg, len(eps)), encoding="utf-8")
    progress("checking the work")
    audit = audit_run(out, data_dir, cfg)
    build_dashboard(out)
    return {"ranked": ranked, "rejected": rejected, "episodes": eps, "meta": meta, "audit": audit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--episodes", nargs="*", type=int)
    ap.add_argument("--out")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--stitch", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    data_dir = cfg["paths"]["data_dir"]
    if a.stage == "parse":
        try:
            eps = load_episodes(data_dir, a.episodes)
        except EpisodeFileError as e:
            print(f"stopped: {e}")
            return 1
        bad = [(e.ep, e.path, e.warnings) for e in eps if e.warnings]
        print(f"parsed {len(eps)} episodes, {sum(len(e.events) for e in eps)} events, {len(bad)} with warnings")
        for b in bad:
            print("!", b)
        return
    try:
        r = run_pipeline(cfg, data_dir, a.out or cfg["paths"]["outputs_dir"], a.episodes, a.mock, a.stitch)
    except EpisodeFileError as e:
        print(f"stopped: {e}")
        return 1
    except LLMError as e:
        print(f"\nstopped: no model could answer.\n  {e}\n"
              "Fix the key/quota above and re-run the same command; finished calls are cached and won't repeat.")
        return 1
    fixed = r["meta"].get("stats", {}).get("auto_repaired", 0)
    kept = sum(1 for p in r["ranked"] if p.get("repair"))
    print(f"{len(r['ranked'])} promos ranked, {len(r['rejected'])} rejected by the strict check"
          + (f"; {fixed} near-misses auto-repaired, {kept} of them made the ranking"
             f"{' (the rest overlapped a better promo)' if kept < fixed else ''}, labelled 'auto_adjusted'" if fixed else ""))
    if r["meta"].get("stopped_reason"):
        print("\n" + r["meta"]["status"] + "\n  reason: " + r["meta"]["stopped_reason"][:200])
        return 2


if __name__ == "__main__":
    sys.exit(main())