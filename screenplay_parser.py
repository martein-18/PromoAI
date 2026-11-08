"""Episode parser (code only, no LLM). Turns a screenplay .txt into structured events.

Tolerant of format drift: case-insensitive headings, hyphen/en-dash, optional hours in
timestamps, blank lines. Anything odd is recorded in Episode.warnings instead of crashing.
"""
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME = r"(?:(\d+):)?(\d+):(\d{2})(?:\.(\d+))?"
RE_TIME = re.compile(TIME)
RE_RANGE = re.compile(r"start\s*:\s*\(?\s*" + TIME + r"\s*\)?\s*\|\s*end\s*:\s*\(?\s*" + TIME + r"\s*\)?", re.I)
RE_BLOCK = re.compile(r"^\*\*(.+?):?\*\*\s*$")
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_HEADING = re.compile(r"^scene\s+heading\s*:\s*(.+)$", re.I)
RE_SCENE_START = re.compile(r"^scene\s+start\s*:\s*(.+)$", re.I)
RE_SCENE_END = re.compile(r"^scene\s+end\s*:\s*(.+)$", re.I)
RE_SCRIPT_END = re.compile(r"^-+\s*SCRIPT\s+END\s*-+$", re.I)
RE_PAREN = re.compile(r"\(([^)]*)\)")
NON_SPEECH = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SOUND_FX",
              "SOUND EFFECT": "SOUND_FX", "SOUND EFFECTS": "SOUND_FX", "SFX": "SOUND_FX"}
VOICE_MODS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D"}


def to_seconds(s: str):
    m = RE_TIME.search(s or "")
    if not m:
        return None
    h, mi, se, fr = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + (float("0." + fr) if fr else 0.0)


@dataclass
class Event:
    id: int
    ep: int
    scene: int
    start: float
    end: float
    type: str            # CAMERA | ACTION | SOUND_FX | DIALOGUE
    speaker: str = ""
    modifiers: list = field(default_factory=list)   # O.S., V.O. ...
    tone: str = ""
    text: str = ""

    @property
    def spoken(self):
        return self.type == "DIALOGUE"


@dataclass
class Scene:
    number: int
    heading: str = ""
    start: float = None
    end: float = None


@dataclass
class Episode:
    ep: int
    path: str
    logline: str = ""
    continuity: str = ""
    scenes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def duration(self):
        return max((e.end for e in self.events), default=0.0)

    def event(self, eid):
        return self._idx.get(eid)

    def __post_init__(self):
        self._idx = {}

    def reindex(self):
        self._idx = {e.id: e for e in self.events}

    def to_dict(self):
        return {"ep": self.ep, "path": self.path, "logline": self.logline, "continuity": self.continuity,
                "duration": self.duration, "scenes": [asdict(s) for s in self.scenes],
                "events": [asdict(e) for e in self.events], "warnings": self.warnings}


def _parse_speaker(raw: str):
    parens = RE_PAREN.findall(raw)
    name = RE_PAREN.sub("", raw).strip()
    mods = [p.strip() for p in parens if p.strip().upper() in VOICE_MODS]
    tone = ", ".join(p.strip() for p in parens if p.strip().upper() not in VOICE_MODS)
    return name.title(), mods, tone


def _header_field(lines, key):
    pat = re.compile(rf"^{key}\s*:\s*(.*)$", re.I)
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            return m.group(1).strip()
    return ""


def parse_episode(path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    head = lines[:30]
    num_txt = _header_field(head, "EPISODE NUMBER")
    m = re.search(r"(\d+)", num_txt) or re.search(r"episode[-_ ]?0*(\d+)", path.name, re.I)
    ep = Episode(ep=int(m.group(1)) if m else -1, path=path.name,
                 logline=_header_field(head, "EPISODE LOGLINE"),
                 continuity=_header_field(head, "CONTINUITY NOTES"))
    if ep.ep == -1:
        ep.warnings.append("episode number not found")
    if not ep.logline:
        ep.warnings.append("logline missing")

    scene_no, eid, i, n = 0, 0, 0, len(lines)
    cur_scene = None
    while i < n:
        ln = lines[i].strip()
        if RE_SCRIPT_END.match(ln):
            break
        ms = RE_SCENE.match(ln)
        if ms:
            scene_no = int(ms.group(1))
            cur_scene = Scene(number=scene_no)
            ep.scenes.append(cur_scene)
            i += 1
            continue
        if cur_scene is not None:
            for rx, attr, conv in ((RE_HEADING, "heading", str.strip),
                                   (RE_SCENE_START, "start", to_seconds),
                                   (RE_SCENE_END, "end", to_seconds)):
                mm = rx.match(ln)
                if mm:
                    setattr(cur_scene, attr, conv(mm.group(1)))
        mb = RE_BLOCK.match(ln)
        if mb:
            label = mb.group(1).strip()
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            mr = RE_RANGE.search(lines[i]) if i < n else None
            if not mr:
                ep.warnings.append(f"line {i+1}: block '{label}' has no Start/End line")
                continue
            g = mr.groups()
            start = to_seconds(":".join(x for x in (g[0], g[1], g[2]) if x is not None) + (("." + g[3]) if g[3] else ""))
            end = to_seconds(":".join(x for x in (g[4], g[5], g[6]) if x is not None) + (("." + g[7]) if g[7] else ""))
            i += 1
            body = []
            while i < n:
                t = lines[i].strip()
                if RE_BLOCK.match(t) or RE_SCENE.match(t) or RE_SCRIPT_END.match(t):
                    break
                if t:
                    body.append(t)
                i += 1
            text = " ".join(body)
            base = RE_PAREN.sub("", label).strip().upper()
            eid += 1
            if base in NON_SPEECH:
                ev = Event(eid, ep.ep, scene_no, start, end, NON_SPEECH[base], text=text)
            else:
                spk, mods, tone = _parse_speaker(label)
                ev = Event(eid, ep.ep, scene_no, start, end, "DIALOGUE", spk, mods, tone,
                           text.strip().strip('"“”').strip())
            if end < start:
                ep.warnings.append(f"event {eid}: negative duration ({start}->{end})")
            if not text:
                ep.warnings.append(f"event {eid}: empty text")
            ep.events.append(ev)
            continue
        i += 1

    if not ep.scenes:
        ep.warnings.append("no scenes found")
    if not ep.events:
        ep.warnings.append("no events parsed")
    ep.reindex()
    return ep


class EpisodeFileError(ValueError):
    """An episode file problem the person must fix (missing or duplicate episode number)."""


def parse_dir(data_dir, pattern="*.txt"):
    eps = [parse_episode(p) for p in sorted(Path(data_dir).glob(pattern))]
    # Episodes are keyed by number everywhere; a missing or repeated number would make one file silently replace
    # another, so stop with the file names instead.
    missing = [e.path for e in eps if e.ep == -1]
    if missing:
        raise EpisodeFileError("No episode number found in: " + ", ".join(missing)
                         + ' (add a line "EPISODE NUMBER: Episode N" at the top of each file)')
    by = {}
    for e in eps:
        by.setdefault(e.ep, []).append(e.path)
    dupes = {n: ps for n, ps in by.items() if len(ps) > 1}
    if dupes:
        raise EpisodeFileError("Same episode number in more than one file: "
                         + "; ".join(f"episode {n}: {', '.join(ps)}" for n, ps in sorted(dupes.items())))
    eps.sort(key=lambda e: e.ep)
    return eps
