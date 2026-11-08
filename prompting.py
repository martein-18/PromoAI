"""Rendering helpers: events -> numbered text, clip text, prompt templates ({{var}} replace)."""
from pathlib import Path
from timeline import fmt_ts

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def render(name, **vars):
    t = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in vars.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


def format_event(e, include_camera=True):
    if e.type == "CAMERA" and not include_camera:
        return None
    who = ""
    if e.type == "DIALOGUE":
        who = f"{e.speaker}{' ' + ' '.join(e.modifiers) if e.modifiers else ''}{f' ({e.tone})' if e.tone else ''}: "
    else:
        who = f"{e.type}: "
    return f"[{e.id}] {e.start:.1f}-{e.end:.1f} {who}{e.text}"


def format_episode(ep, include_camera=True, id_range=None):
    """Numbered event list the model reads. Format: [id] start-end TYPE/SPEAKER: text"""
    out = [f"EPISODE {ep.ep} | logline: {ep.logline} | continuity: {ep.continuity}"]
    for e in ep.events:
        if id_range and not (id_range[0] <= e.id <= id_range[1]):
            continue
        line = format_event(e, include_camera)
        if line:
            out.append(line)
    return "\n".join(out)


def clip_text(ep_map, segments, include_camera=False, include_sfx=True):
    """Plain-text rendering of a candidate clip (what the cold viewer sees)."""
    parts = []
    for s in segments:
        ep = ep_map[s.ep]
        evs = [e for e in ep.events if s.start_id <= e.id <= s.end_id]
        evs.sort(key=lambda e: (e.start, e.id))
        lines = []
        for e in evs:
            if e.type == "CAMERA" and not include_camera:
                continue
            if e.type == "SOUND_FX" and not include_sfx:
                continue
            lines.append(f"{e.speaker}: \"{e.text}\"" if e.spoken else f"({e.type.lower().replace('_', ' ')}) {e.text}")
        parts.append(f"--- Ep {s.ep} {fmt_ts(s.start)}-{fmt_ts(s.end)} ---\n" + "\n".join(lines))
    return "\n".join(parts)
