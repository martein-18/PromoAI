"""LLM-facing steps: show bible, candidate generation, scoring, cold-viewer check.
Prompts live in prompts/*.txt; the model lives in config.yaml (see llm.py)."""
import json
from pathlib import Path
from llm import extract_json
from prompting import render, format_episode


def _safe(raw):
    try:
        return extract_json(raw)
    except ValueError:
        return {"_parse_error": (raw or "")[:200]}


def _forget(client, prompt):
    """Drop an unusable reply from the cache so the next attempt really asks the model again."""
    fn = getattr(client, "forget", None)
    if fn:
        fn(prompt)


def bible_text(bible):
    return json.dumps(bible, ensure_ascii=False, indent=1) if bible else "(none)"


def build_bible(episodes, client, out_path):
    eps = "\n".join(f"Ep {e.ep}: {e.logline}\n   continuity: {e.continuity}" for e in episodes)
    prompt = render("bible_prompt", episodes=eps)
    bible = _safe(client.call(prompt, task="bible"))
    if "_parse_error" in bible:
        _forget(client, prompt)
    Path(out_path).write_text(json.dumps(bible, indent=2, ensure_ascii=False), encoding="utf-8")
    return bible


def load_bible(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def generate_candidates(ep, bible, client, n=3, feedback="", include_camera=True, ep_text=None, default_ep=None):
    prompt = render("promo_prompt", bible=bible_text(bible), episode=ep_text or format_episode(ep, include_camera),
                    n_candidates=n, feedback=feedback)
    raw = client.call(prompt, task=f"candidates:ep{ep.ep if ep else default_ep}")
    data = _safe(raw)
    cands = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(cands, list) or not all(isinstance(c, dict) for c in cands):
        _forget(client, prompt)          # not JSON / wrong shape: a failure (not "no promos"), and never cached
        return None
    return cands


SCORE_KEYS = ("hook", "cliffhanger", "standalone", "emotional")
JSON_RETRY_NOTE = "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object, nothing else."


def _scores_ok(d):
    return isinstance(d, dict) and all(isinstance(d.get(k), dict) and isinstance(d[k].get("score"), (int, float))
                                       for k in SCORE_KEYS)


def _score_once(prompt, client):
    """One scoring pass. A broken reply is retried once (different prompt, so not a cache hit)."""
    for p in (prompt, prompt + JSON_RETRY_NOTE):
        out = _safe(client.call(p, task="score"))
        if _scores_ok(out):
            return out
        _forget(client, p)
    return None


def _vector(d):
    return tuple(d[k]["score"] for k in SCORE_KEYS)


def score_clip(clip, cand, bible, client, samples=1):
    """Scores must contain all four criteria; if every pass is broken the promo gets _parse_error, which main.py
    turns into a visible warning.

    samples > 1: adaptive repeat scoring against run-to-run drift. Pass 2 always runs; pass 3 only when passes 1 and
    2 disagree, so a consistent model costs 2 calls, not 3. Each criterion is the MEAN of the valid passes (finer
    than whole numbers, which also breaks ties). Every pass is kept under "samples" so drift stays visible."""
    prompt = render("score_prompt", bible=bible_text(bible), clip=clip,
                    hook=cand.get("hook", ""), cliffhanger=cand.get("cliffhanger", ""))
    passes = [_score_once(prompt, client)]
    for k in range(2, max(1, samples) + 1):
        valid = [p for p in passes if p]
        if k >= 3 and len(valid) >= 2 and len({_vector(p) for p in valid}) == 1:
            break                                     # first passes agree: a third pass would add nothing
        passes.append(_score_once(prompt + f"\n\n(Independent scoring pass {k}.)", client))
    valid = [p for p in passes if p]
    if not valid:
        return {"_parse_error": "no valid scoring pass"}
    if len(valid) == 1:
        out = dict(valid[0])
    else:
        out = {}
        for key in SCORE_KEYS:
            vals = [p[key]["score"] for p in valid]
            mean = sum(vals) / len(vals)
            closest = min(valid, key=lambda p: abs(p[key]["score"] - mean))   # its reason fits the mean best
            out[key] = {"score": round(mean, 2), "reason": closest[key].get("reason", ""), "passes": vals}
        out["grab_line"] = valid[0].get("grab_line", "")
        tags = [p.get("hook_type") for p in valid if p.get("hook_type")]
        out["hook_type"] = max(set(tags), key=tags.count) if tags else "other"
    out["_passes"] = len(valid)
    out["_agreed"] = len({_vector(p) for p in valid}) == 1
    return out


def compare_top(top, client):
    """Head-to-head ordering of the top clips. Models judge 'which is better' more consistently than absolute
    scores. Two listwise passes, the second with the list REVERSED, cancel position bias; positions are summed
    (Borda). A pairwise round-robin would need n*(n-1)/2 calls (105 for 15 clips); this needs 2.
    Returns {"order": [promo ids best first], "agreement": rank correlation of the two passes, ...} or {}."""
    if len(top) < 2:
        return {}
    letters = [chr(65 + i) for i in range(len(top))]
    by_letter = dict(zip(letters, top))

    def one_pass(seq, tag):
        blocks = [f"=== CLIP {L} ===\n{by_letter[L].get('clip_text', '').strip()}" for L in seq]
        prompt = render("compare_prompt", n=len(seq), clips="\n\n".join(blocks))
        for attempt in (prompt, prompt + JSON_RETRY_NOTE):
            data = _safe(client.call(attempt, task=f"compare:{tag}"))
            order = data.get("order") if isinstance(data, dict) else None
            if isinstance(order, list) and sorted(map(str, order)) == sorted(letters):
                return [str(x) for x in order], data.get("reason", "")
            _forget(client, attempt)
        return None, ""

    fwd, why = one_pass(letters, "forward")
    rev, why2 = one_pass(list(reversed(letters)), "reversed")
    passes = [o for o in (fwd, rev) if o]
    if not passes:
        return {}
    pos = {L: sum(o.index(L) for o in passes) for L in letters}
    score_pos = {L: i for i, L in enumerate(letters)}          # tie-break: the score order
    final = sorted(letters, key=lambda L: (pos[L], score_pos[L]))
    agreement = None
    if fwd and rev:
        n = len(letters)
        d2 = sum((fwd.index(L) - rev.index(L)) ** 2 for L in letters)
        agreement = round(1 - 6 * d2 / (n * (n * n - 1)), 2)
    return {"order": [by_letter[L]["id"] for L in final], "passes": len(passes), "agreement": agreement,
            "reason": why or why2}


def cold_viewer(clip, client):
    """Gets NO bible on purpose: checks the clip stands alone."""
    return _safe(client.call(render("cold_viewer_prompt", clip=clip), task="cold_viewer"))


def explain_ranking(top, client):
    """One extra call: why each of the top promos sits where it does (model rationale, shown as such)."""
    lines = []
    for p in top:
        sc = p.get("scores") or {}
        g = lambda k: sc[k].get("score") if isinstance(sc.get(k), dict) else "?"
        segs = ", ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        moved = (f" | placed here by head-to-head comparison (was #{p['score_rank']} by score)"
                 if p.get("score_rank") and p["score_rank"] != p["rank"] else "")
        lines.append(f"#{p['rank']} {segs} | hook: {p.get('hook')} | cliffhanger: {p.get('cliffhanger')} | "
                     f"hook {g('hook')} cliff {g('cliffhanger')} standalone {g('standalone')} emotional {g('emotional')} | final {p.get('final_score')}{moved}")
    prompt = render("rank_prompt", promos="\n".join(lines))
    data = _safe(client.call(prompt, task="rank_explain"))
    if not isinstance(data, dict) or "_parse_error" in data:
        _forget(client, prompt)
        return {}
    items = {}
    for i in data.get("items", []):
        try:
            items[int(i.get("rank"))] = i.get("reason", "")   # models sometimes return "1" instead of 1
        except (AttributeError, TypeError, ValueError):
            pass
    return {"overall": data.get("overall", ""), "items": items}

def assemble_candidates(pool, ep_map, client, n=3):
    """Cross-episode joins proposed from already-validated clips (see prompts/assemble_prompt.txt).
    Returns a list of candidates (3-number segments) or None if the reply was unusable."""
    blocks = []
    for p in pool:
        lines = []
        for s in p["segments"]:
            ep = ep_map[s["ep"]]
            for e in sorted((e for e in ep.events if s["start_id"] <= e.id <= s["end_id"] and e.type != "CAMERA"),
                            key=lambda e: e.start):
                who = e.speaker if e.type == "DIALOGUE" else e.type
                lines.append(f"[{e.ep}.{e.id}] {e.start:.1f}-{e.end:.1f} {who}: {e.text}")
        blocks.append(f"=== CLIP {p['id']} | Episode {p['segments'][0]['ep']}: {ep_map[p['segments'][0]['ep']].logline} ===\n"
                      + "\n".join(lines))
    prompt = render("assemble_prompt", n=n, clips="\n\n".join(blocks))
    for attempt in (prompt, prompt + JSON_RETRY_NOTE):
        data = _safe(client.call(attempt, task="assemble"))
        promos = data.get("promos") if isinstance(data, dict) else None
        if isinstance(promos, list) and all(isinstance(c, dict) for c in promos):
            return promos[:n]
        _forget(client, attempt)
    return None
