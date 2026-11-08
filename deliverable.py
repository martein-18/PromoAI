"""PROMOS.md: the one file that answers the brief. Posting order first, then every promo in the requested format,
then a measured "where does it pick badly" section built from the run's own data (no claims without numbers)."""
from collections import Counter
from slate import slate_markdown

REASON = {"duration": "too short or too long", "slow_open": "nothing happens in the first 3s",
          "hook_late": "its own hook line starts after 3s", "hook_outside": "its hook line is not in the clip's opening",
          "cliff_not_at_end": "runs past its cliffhanger (risks resolving it)", "cliff_outside": "cliffhanger line not in the clip",
          "model_reply_not_json": "AI reply unreadable", "bad_ids": "points at lines that do not exist",
          "bad_spec": "malformed clip spec", "order_overlap": "parts out of order", "time_overlap": "parts overlap",
          "not_cross_episode": "join used only one episode", "part_not_from_checked_clip": "join used unchecked lines"}


def _code(issue):
    s = str(issue)
    return s.split("]", 1)[1].split(":", 1)[0].strip() if "]" in s else s


def _seg_lines(p):
    return [f"   {i}. Episode {s['ep']}: {s['start_ts']} -> {s['end_ts']} ({s['duration']:.1f}s)"
            for i, s in enumerate(p["segments"], 1)]


def build(ranked, candidates, rejected, meta, cfg, n_episodes):
    v = cfg["validation"]
    out = ["# Promos", ""]
    if meta.get("stopped_reason"):
        out += [f"> **{meta['status']}**", ""]
    out += [f"{len(ranked)} promos from {n_episodes} episodes. Every timestamp comes from the screenplay files, not from "
            f"the AI. Picked by {', '.join(meta.get('models_used') or []) or '?'}; every promo scored blind by "
            f"{meta.get('scoring_model') or meta.get('model')}.", ""]

    out += ["## 1. Which to post first", ""]
    plan = slate_markdown(meta.get("slate") or [], meta.get("slate_left_out") or []).split("\n", 2)[-1]
    out += [plan.strip(), ""]
    if meta.get("ranking_summary"):
        out += [f"AI's note on the top of the ranking: {meta['ranking_summary']}", ""]

    er = meta.get("episode_ranking") or []
    if er:
        out += ["## Episodes ranked by their best promo", "",
                "Ordered by where each episode's best promo landed in the final ranking. Potential = the highest score the "
                "blind AI scorer gave any of the episode's clips that passed the rules.", "",
                "| # | Episode | Potential (0-5) | Best promo | Clips passed / rejected | Note |", "|---|---|---|---|---|---|"]
        for r in er:
            if r["status"] == "ok":
                out.append(f"| {r['position']} | {r['ep']} | {r['potential']:.2f} | "
                           f"{'#' + str(r['best_promo_rank']) if r.get('best_promo_rank') else '(overlapped)'} | "
                           f"{r['clips_passed']} / {r['clips_rejected']} | {r.get('best_hook', '')[:80]} |")
            else:
                out.append(f"| {r['position']} | {r['ep']} | - | - | 0 / {r['clips_rejected']} | {r.get('why', '')} |")
        out.append("")

    out += ["## 2. All promos, best first", "",
            "Format: episodes, then each part's start -> end in viewing order, then hook, cliffhanger and one-line reason.", ""]
    for p in ranked:
        eps = sorted({s["ep"] for s in p["segments"]})
        flag = f"  _(check by hand: {'; '.join(p['review'])})_" if p.get("review") else ""
        out += [f"**#{p['rank']}** Episode{'s' if len(eps) > 1 else ''} {', '.join(map(str, eps))} | "
                f"{p['total_duration']:.1f}s | score {p.get('final_score')}{flag}"]
        out += _seg_lines(p)
        out += [f"   - Hook: {p.get('hook', '')}", f"   - Cliffhanger: {p.get('cliffhanger', '')}",
                f"   - Why: {p.get('post_reason') or p.get('why', '')}", ""]

    out += ["## 3. Where it picks badly (measured on this run)", ""]
    codes = Counter(_code(i) for r in rejected for i in r.get("issues", []) if str(i).startswith("[error]"))
    if codes:
        out += [f"- **{len(rejected)} AI picks were rejected by the code rules** before scoring:"]
        for c, n in codes.most_common():
            ex = next((r for r in rejected if any(_code(i) == c for i in r.get("issues", []))), None)
            ex_txt = f" (e.g. `{ex['source']}`: {next(str(i) for i in ex['issues'] if _code(i) == c)[8:120]})" if ex else ""
            out += [f"  - {n} x {REASON.get(c, c)}{ex_txt}"]
    timing = sum(codes.get(c, 0) for c in ("hook_late", "hook_outside", "cliff_not_at_end", "cliff_outside"))
    if timing:
        out += [f"- **{timing} picks broke the hook or cliffhanger timing rules**: the AI often names a hook that is not "
                "actually at the start, or ends a beat after the cliffhanger."]
    rep = [p for p in ranked if p.get("repair")]
    if rep:
        out += [f"- **{len(rep)} ranked promos needed a code length repair** (+/-1 line): "
                + ", ".join(f"#{p['rank']}" for p in rep) + "."]
    late = [p for p in ranked if (p.get("strongest_at_s") or 0) > v.get("first_beat_max_s", 3.0)]
    if late:
        out += [f"- **{len(late)} promos bury their strongest line**: the blind scorer's most attention-grabbing line "
                f"comes after the first {v.get('first_beat_max_s', 3.0):.0f}s (e.g. #{late[0]['rank']}: "
                f"{late[0]['strongest_at_s']:.0f}s in). The picker's chosen hook is not always the real hook."]
    multi = [p for p in ranked if ((p.get("scores") or {}).get("_passes") or 1) > 1]
    if multi:
        dis = sum(1 for p in multi if not p["scores"].get("_agreed"))
        out += [f"- **Scoring drift**: {dis} of {len(multi)} promos got different scores on repeat scoring of the same "
                "clip; their scores are averages and they are flagged for review."]
    cmp_ = meta.get("comparison") or {}
    if cmp_.get("agreement") is not None:
        out += [f"- **Head-to-head consistency**: the forward and reversed comparisons agreed at {cmp_['agreement']} "
                "(1 = identical order). Below about 0.5 the comparison follows list position, not quality."]
    covered = {s["ep"] for p in ranked for s in p["segments"]}
    empty = [e for e in meta.get("episodes", []) if e not in covered]
    if empty:
        out += [f"- **{len(empty)} episodes have no promo** ({', '.join(map(str, empty))}): either the AI found no "
                "unresolved moment, or every pick broke a rule. Quiet episodes may be right to skip; check by hand."]
    review = [p for p in ranked if p.get("review")]
    out += [f"- **{len(review)} of {len(ranked)} promos are flagged for a human look** (reasons next to each promo above).",
            "- **Blind spots by design**: the tool reads script text only, so a slap or a look written as one short "
            "action line can be underrated; sarcasm and subtext can be read literally.",
            "- **Hand-check**: judge the top clips by eye in `hand_check.md` (python hand_check.py <run>) and add the "
            "results here.", ""]

    out += ["## 4. Limits", "",
            "- Ranking weights (hook, cliffhanger, standalone, emotional) are proposals, not set by the client.",
            "- Clips are first picked one episode at a time; cross-episode promos come from stitching consecutive "
            "episodes and from joining parts of clips that already passed every rule.",
            "- Free-tier AI models: quality is below the best paid models, and the daily quota limits run size.", ""]
    return "\n".join(out)
