"""Cross-episode helpers. Link detection is static; which links to stitch is your call."""
import re
from prompting import format_event

LINK_PATTERNS = [r"picks? up immediately", r"cutting directly", r"continues? (?:directly|immediately)",
                 r"immediately after", r"follows"]


def find_links(episodes):
    """[(ep_n, ep_n+1, strength, matched_phrase)] using ep n+1's continuity notes.
    strength: 'strong' for immediate pick-ups, 'weak' for plain 'follows'."""
    by = {e.ep: e for e in episodes}
    links = []
    for e in episodes:
        if e.ep - 1 not in by:
            continue
        note = e.continuity.lower()
        for pat in LINK_PATTERNS:
            m = re.search(pat, note)
            if m:
                strength = "weak" if pat == r"follows" else "strong"
                links.append((e.ep - 1, e.ep, strength, m.group(0)))
                break
    return links


def pair_text(ep_a, ep_b, tail_s=30.0, head_s=30.0, include_camera=True):
    """Last tail_s of A + first head_s of B, event IDs kept per episode (use 3-number segments)."""
    def pick(ep, cond):
        return [format_event(e, include_camera) for e in ep.events if cond(e)]
    a = [x for x in pick(ep_a, lambda e: e.end >= ep_a.duration - tail_s) if x]
    b = [x for x in pick(ep_b, lambda e: e.start <= head_s) if x]
    return (f"EPISODE {ep_a.ep} (ending) | logline: {ep_a.logline}\n" + "\n".join(a) +
            f"\n\nEPISODE {ep_b.ep} (opening) | continuity: {ep_b.continuity}\n" + "\n".join(b) +
            "\n\nFor this pair use 3-number segments: [episode, start_id, end_id].")
