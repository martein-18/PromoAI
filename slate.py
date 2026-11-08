"""Posting plan ("which would you post first, and why"): turns the ranking into a day-by-day slate. Code only.

Rules (config.yaml -> slate):
  * candidates: the best `pool` ranked promos; promos flagged for review are left out unless there are not enough
  * variety: at most `max_per_hook_type` per hook type, and at most one promo per episode
  * spoiler-safe order: promos showing only early episodes go first, so a late twist is not used up on day 1;
    ties go to the better rank
  * platform: clips up to `short_max_s` seconds suit TikTok / Reels; longer ones suit YouTube Shorts
"""


def _eps(p):
    return sorted({s["ep"] for s in p["segments"]})


def build_slate(ranked, cfg):
    sc = cfg.get("slate") or {}
    size, pool = sc.get("size", 7), sc.get("pool", 20)
    cap, short = sc.get("max_per_hook_type", 2), sc.get("short_max_s", 30)
    cands = ranked[:pool]
    ordered = [p for p in cands if not p.get("review")] + [p for p in cands if p.get("review")]
    picks, by_type, used_eps, left_out = [], {}, set(), []
    for p in ordered:
        t = (p.get("scores") or {}).get("hook_type") or "other"
        eps = set(_eps(p))
        if len(picks) < size and (by_type.get(t, 0) >= cap or eps & used_eps):
            if p["rank"] <= size + 3:                   # explain why a top promo is not in the plan
                left_out.append({"rank": p["rank"], "promo_id": p["id"], "why":
                                 f"already {cap} '{t}' hooks in the plan" if by_type.get(t, 0) >= cap
                                 else f"episode {', '.join(map(str, sorted(eps & used_eps)))} already has a promo in the plan"})
            continue
        picks.append(p)
        by_type[t] = by_type.get(t, 0) + 1
        used_eps |= eps
        if len(picks) == size:
            break
    quality = {p["id"]: i for i, p in enumerate(sorted(picks, key=lambda p: p["rank"]), 1)}
    picks.sort(key=lambda p: (max(_eps(p)), p["rank"]))
    days = []
    for d, p in enumerate(picks, 1):
        eps, t = _eps(p), (p.get("scores") or {}).get("hook_type") or "other"
        platform = "TikTok / Reels" if p["total_duration"] <= short else "YouTube Shorts"
        why = (f"rank #{p['rank']}, {t} hook, {p['total_duration']:.0f}s for {platform}; "
               f"shows nothing past episode {max(eps)}")
        q = quality[p["id"]]
        if q > d:
            why += f"; posted earlier than its quality (#{q} of this plan) because it shows only early episodes"
        elif q < d:
            why += (f"; #{q} of this plan in quality, held to day {d} so it does not spoil episode {max(eps)} "
                    "before the earlier promos have run")
        if p.get("review"):
            why += "; check by hand first (" + ", ".join(p["review"]) + ")"
        days.append({"day": d, "promo_id": p["id"], "rank": p["rank"], "episodes": eps, "hook_type": t,
                     "duration_s": p["total_duration"], "platform": platform, "hook": p.get("hook", ""), "why": why})
    return days, left_out


def slate_markdown(days, left_out=()):
    if not days:
        return "# Posting plan\n\nNo promos to schedule.\n"
    out = ["# Posting plan", "",
           "Spoiler-safe order (early episodes first), varied hook types, one promo per episode. Built by code from the ranking.", "",
           "| Day | Promo | Episodes | Hook type | Length | Platform | Why |", "|---|---|---|---|---|---|---|"]
    for d in days:
        out.append(f"| {d['day']} | #{d['rank']} `{d['promo_id']}` | {', '.join(map(str, d['episodes']))} | {d['hook_type']} | "
                   f"{d['duration_s']:.0f}s | {d['platform']} | {d['why']} |")
    if left_out:
        out += ["", "Top-ranked promos not in the plan:", ""]
        out += [f"- #{x['rank']} `{x['promo_id']}`: {x['why']}" for x in left_out]
    return "\n".join(out) + "\n"
