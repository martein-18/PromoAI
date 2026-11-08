"""ID -> time conversion. The LLM only ever returns event IDs; all timestamps come from here."""
from dataclasses import dataclass, asdict


def fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Segment:
    ep: int
    start_id: int
    end_id: int
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d.update(duration=round(self.duration, 2), start_ts=fmt_ts(self.start), end_ts=fmt_ts(self.end))
        return d


def normalize_spec(spec, default_ep):
    """[a,b] -> (default_ep,a,b); [ep,a,b] -> (ep,a,b). Returns None if malformed."""
    try:
        if len(spec) == 2:
            return int(default_ep), int(spec[0]), int(spec[1])
        if len(spec) == 3:
            return int(spec[0]), int(spec[1]), int(spec[2])
    except (TypeError, ValueError):
        pass
    return None


def resolve_segment(ep, a, b, snap_end="last_spoken", end_pad=0.3):
    """Events with id in [a,b] -> Segment. Returns (Segment|None, error|None).

    snap_end: 'last_spoken' (end = last dialogue end + pad, if any dialogue),
              'last_event' (end = latest event end), 'none' (same as last_event, no pad).
    """
    if a > b:
        return None, f"ep{ep.ep}: range {a}-{b} is reversed"
    evs = [e for e in ep.events if a <= e.id <= b]
    if not evs:
        return None, f"ep{ep.ep}: no events with ids in {a}-{b}"
    start = min(e.start for e in evs)
    end = max(e.end for e in evs)
    if snap_end == "last_spoken":
        # End on the last real beat (dialogue, action, sound), not the last spoken line: a visual cliffhanger such
        # as "The door bursts open" after the last line must stay inside the clip. Only trailing CAMERA lines,
        # which often run long, are trimmed.
        beats = [e for e in evs if e.type != "CAMERA"]
        if beats:
            end = min(end, max(e.end for e in beats) + end_pad)
    return Segment(ep.ep, a, b, start, end), None
