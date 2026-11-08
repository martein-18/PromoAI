"""Scoring and ranking. Weights are PROPOSED (config.yaml -> ranking), not specified by Kuku."""
import pandas as pd


def _s(promo, key):
    v = (promo.get("scores") or {}).get(key)
    if isinstance(v, dict):
        v = v.get("score")
    return float(v) if isinstance(v, (int, float)) else 0.0


def compute_final_score(promo, cfg):
    """Weighted sum of the 1-5 scores. Optional penalty when the cold viewer could not follow the clip."""
    r = cfg["ranking"]
    score = sum(w * _s(promo, k) for k, w in r["weights"].items())
    cv = promo.get("cold_viewer") or {}
    if cv.get("can_answer") is False:
        score -= r.get("cold_viewer_fail_penalty", 0.0)
    return score


def shared_seconds(a, b):
    return sum(max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
               for x in a["segments"] for y in b["segments"] if x["ep"] == y["ep"])


def overlaps(a, b, max_share=0.5):
    """True when the two clips are mostly the same footage: shared time > max_share of the SHORTER clip.
    A one-second touch no longer throws away a different 25s or 55s promo next to a better one."""
    short = min(a["total_duration"], b["total_duration"]) or 1.0
    return shared_seconds(a, b) / short > max_share


def dedupe(ranked, max_share=0.5):
    kept = []
    for p in ranked:
        if not any(overlaps(p, k, max_share) for k in kept):
            kept.append(p)
    return kept


def apply_diversity(ranked, cfg):
    """Cap how many promos of one hook_type reach the top-K (config ranking.diversity). 0/absent = off."""
    d = cfg["ranking"].get("diversity") or {}
    cap, top_k = d.get("max_per_hook_type", 0), d.get("top_k", 10)
    if not cap:
        return ranked
    head, tail, counts = [], [], {}
    for p in ranked:
        t = (p.get("scores") or {}).get("hook_type", "other")
        if len(head) < top_k and counts.get(t, 0) < cap:
            counts[t] = counts.get(t, 0) + 1
            head.append(p)
        else:
            tail.append(p)
    return head + tail


def apply_comparison(ranked, comparison):
    """Reorder the compared top clips by the head-to-head result; the rest keep their score order.
    Every promo keeps score_rank (its place by score alone) so the change stays visible."""
    for p in ranked:
        p["score_rank"] = p["rank"]
    order = (comparison or {}).get("order") or []
    if not order:
        return ranked
    by_id = {p["id"]: p for p in ranked}
    head = [by_id[i] for i in order if i in by_id]
    rest = [p for p in ranked if p["id"] not in set(order)]
    out = head + rest
    for i, p in enumerate(out, 1):
        p["rank"] = i
    return out


def rank(promos, cfg):
    for p in promos:
        p["final_score"] = round(compute_final_score(p, cfg), 3)
    # ties broken by cliffhanger, then hook, then standalone - never by episode order (which favoured episode 1)
    key = lambda p: (p["final_score"], _s(p, "cliffhanger"), _s(p, "hook"), _s(p, "standalone"))
    share = cfg["ranking"].get("max_overlap_share", 0.5)
    ranked = apply_diversity(dedupe(sorted(promos, key=key, reverse=True), share), cfg)
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
    return ranked


def to_dataframe(promos):
    rows = []
    for p in promos:
        sc = p.get("scores") or {}
        g = lambda k: sc[k].get("score") if isinstance(sc.get(k), dict) else None
        cv = p.get("cold_viewer") or {}
        rows.append({"rank": p.get("rank"), "episodes": ",".join(sorted({str(s["ep"]) for s in p["segments"]}, key=int)),
                     "segments": " | ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]),
                     "duration_s": p["total_duration"], "hook": p.get("hook"), "cliffhanger": p.get("cliffhanger"),
                     "why": p.get("why"), "post_reason": p.get("post_reason"), "hook_type": sc.get("hook_type"), "hook_score": g("hook"),
                     "cliff_score": g("cliffhanger"), "standalone_score": g("standalone"),
                     "emotional_score": g("emotional"), "cold_viewer_ok": cv.get("can_answer"),
                     "final_score": p.get("final_score"), "source": p.get("source"),
                     "auto_adjusted": (p.get("repair") or {}).get("action", ""), "scored_by": p.get("scored_by"),
                     "score_rank": p.get("score_rank"), "needs_review": "; ".join(p.get("review") or [])})
    return pd.DataFrame(rows)