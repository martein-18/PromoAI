"""Duration repair: a small, logged fix for near-miss clips. The strict validator is NOT loosened.

Runs only after validate_candidate() rejected a candidate, and only when the duration rule was the ONLY error.
Guardrails (config.yaml -> pipeline.duration_repair):
  * near-misses only: the clip must miss 20-60s by at most max_miss_s
  * one line, at one edge, at most -- a second line starts changing what the model chose
  * the hook event and the cliffhanger event (IDs the model returned) are never removed
  * too long : first drop one line AFTER the cliffhanger, else one setup line BEFORE the hook
  * too short: first add one earlier setup line, but only if the hook still starts within first_beat_max_s
               of the new clip start; extending past the cliffhanger is the last resort and is flagged,
               because the extra line can resolve the cliffhanger
  * the repaired clip must pass the full strict validator again, otherwise the candidate stays rejected
Every repair is returned with a description so it can be stored on the promo and shown, never hidden.
"""
import copy
from timeline import normalize_spec, resolve_segment
from validator import validate_candidate, is_valid


def _ids(ep):
    return sorted(e.id for e in ep.events)


def _neighbour(ep, i, step):
    ids = _ids(ep)
    if i not in ids:
        return None
    j = ids.index(i) + step
    return ids[j] if 0 <= j < len(ids) else None


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def try_repair(ep_map, cand, default_ep, v, segs, issues, rc):
    """Returns (repaired_candidate, segments, issues, info) or None if the candidate must stay rejected."""
    if not rc or not rc.get("enabled", False) or not segs:
        return None
    errors = [i for i in issues if i.severity == "error"]
    if not errors or any(i.code != "duration" for i in errors):
        return None                                    # only pure duration failures are repairable
    specs = [normalize_spec(s, default_ep) for s in cand.get("segments") or []]
    if not specs or None in specs:
        return None

    lo, hi = v.get("min_duration_s", 20), v.get("max_duration_s", 60)
    total = sum(s.duration for s in segs)
    too_short = total < lo
    miss = (lo - total) if too_short else (total - hi)
    if miss > rc.get("max_miss_s", 6.0):
        return None

    hook, cliff = _as_int(cand.get("hook_event")), _as_int(cand.get("cliffhanger_event"))
    (fe, fa, fb), (le, la, lb) = specs[0], specs[-1]
    hook_ok = hook is not None and fa <= hook <= fb      # hook lies in the first segment: we can protect it
    cliff_ok = cliff is not None and la <= cliff <= lb   # cliffhanger lies in the last segment
    options = []                                         # (description, new_specs, flags)

    def with_first(a, b):
        return [(fe, a, b)] + specs[1:]

    def with_last(a, b):
        return specs[:-1] + [(le, a, b)]

    if not too_short:
        if cliff_ok and lb > cliff:
            prev = _neighbour(ep_map[le], lb, -1)
            if prev is not None and prev >= cliff:
                options.append(("-1 line after the cliffhanger at the end", with_last(la, prev), {}))
        if hook_ok and fa < hook:
            nxt = _neighbour(ep_map[fe], fa, +1)
            if nxt is not None and nxt <= hook:
                options.append(("-1 setup line before the hook at the start", with_first(nxt, fb), {}))
    else:
        if hook_ok:
            prev = _neighbour(ep_map[fe], fa, -1)
            if prev is not None:
                options.append(("+1 setup line at the start", with_first(prev, fb), {"check_hook_start": True}))
        if cliff_ok and rc.get("allow_extend_end", True):
            nxt = _neighbour(ep_map[le], lb, +1)
            if nxt is not None:
                options.append(("+1 line after the cliffhanger at the end (may resolve it, check by hand)",
                                with_last(la, nxt), {"extended_past_cliffhanger": True}))

    for desc, new_specs, flags in options:
        c2 = copy.deepcopy(cand)
        c2["segments"] = [list(s) for s in new_specs]
        segs2, issues2 = validate_candidate(ep_map, c2, default_ep, v)
        if not is_valid(issues2):
            continue                                     # must pass the full strict check, unchanged
        if flags.get("check_hook_start"):
            ep = ep_map[fe]
            hook_ev = next((e for e in ep.events if e.id == hook), None)
            if hook_ev is None or hook_ev.start - segs2[0].start > v.get("first_beat_max_s", 3.0):
                continue                                 # the added setup would push the hook too late
        after = sum(s.duration for s in segs2)
        info = {"action": desc, "before_s": round(total, 2), "after_s": round(after, 2),
                "delta_s": round(after - total, 2), "original_segments": cand.get("segments"),
                "original_issue": str(errors[0]),
                "extended_past_cliffhanger": bool(flags.get("extended_past_cliffhanger"))}
        return c2, segs2, issues2, info
    return None
