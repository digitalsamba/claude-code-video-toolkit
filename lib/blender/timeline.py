"""Times in a shot spec: shot starts and ends, and narration cues (plain Python, no Blender).

    tl = Timeline(spec["shots"], fps, spec.get("voice"), spec_dir)
    tl.time("S3+0.4")            # seconds, relative to a shot
    tl.time("cue:disk is full")  # seconds, the moment the narration says it

compile.py uses it inside Blender; tests/python/test_blender_spec.py runs it on the host.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class SpecError(ValueError):
    pass


def _words(text: str) -> list[str]:
    """Narration words the way lib/components/cues.ts normalises them."""
    return re.sub(r"[^a-z0-9']+", " ", text.lower().replace("\u2019", "'").replace("\u2018", "'")).split()


class Timeline:
    """Shot starts and ends in seconds, and the narration's words if the spec has a `voice` block.

    No Blender in here: tests/python/test_blender_spec.py runs it on the host.
    """

    def __init__(self, shots, fps, voice=None, base: Path | None = None):
        self.fps, self.shots, t = fps, {}, 0.0
        for s in shots:
            if "id" not in s or "dur" not in s or "type" not in s:
                raise SpecError(f"shot needs id, dur and type: {s}")
            if s["id"] in self.shots or s["id"] in ("end", "cue"):
                raise SpecError(f"shot id {s['id']!r} is taken")
            dur = s["dur"]
            if isinstance(dur, bool) or not isinstance(dur, (int, float)) or dur * fps < 1:
                raise SpecError(f'shot {s["id"]!r}: "dur" is a number of seconds, at least one frame long ({dur!r} given)')
            self.shots[s["id"]] = (t, t + float(dur))
            t += float(dur)
        self.duration = t
        self.frames = self.frame(t)
        self.words: list[tuple[str, float, float]] = []  # word, start s, end s on the film's timeline
        for v in voice or []:
            path = Path(v["captions"])
            path = path if path.is_absolute() else (base or Path(".")) / path
            if not path.exists():
                raise SpecError(f"voice captions not found: {path}")
            at = self.time(v.get("at", 0))
            for w in json.loads(path.read_text(encoding="utf-8")):
                for tok in _words(w["text"]):
                    self.words.append((tok, at + w["startMs"] / 1000, at + w["endMs"] / 1000))
        self.words.sort(key=lambda w: w[1])

    def frame(self, t: float) -> int:
        return int(round(t * self.fps)) + 1

    def cue(self, phrase: str, nth: int = 1) -> float:
        """Seconds at which a word or phrase of the narration starts (its nth occurrence)."""
        want = _words(phrase)
        if not want:
            raise SpecError(f"empty cue: {phrase!r}")
        if not self.words:
            raise SpecError(f'cue "{phrase}" needs a "voice" block with captions in the spec')
        hits = [self.words[i][1] for i in range(len(self.words) - len(want) + 1)
                if [w[0] for w in self.words[i:i + len(want)]] == want]
        if len(hits) < nth:
            said = " ".join(w[0] for w in self.words)
            raise SpecError(f'cue "{phrase}" (occurrence {nth}) is not in the narration: {said[:300]}')
        return hits[nth - 1]

    def time(self, ref) -> float:
        """12.5 | "S3" | "S3+0.4" | "S3.end-0.2" | "end" | "cue:disk is full" | "cue:full#2-0.1" -> seconds."""
        if isinstance(ref, (int, float)):
            return float(ref)
        ref = str(ref)
        if ref.strip().startswith("cue:"):
            m = re.fullmatch(r"\s*cue:(.+?)(#\d+)?\s*([+-]\s*[\d.]+)?\s*", ref)
            if not m:
                raise SpecError(f"bad cue reference: {ref!r}")
            phrase, nth, off = m.groups()
            if off and self.words:  # "cue:top-10" is the words "top 10" if the narration says them, not top minus 10 s
                try:
                    return self.cue(ref.strip()[4:])
                except SpecError:
                    pass
            return self.cue(phrase, int(nth[1:]) if nth else 1) + (float(off.replace(" ", "")) if off else 0.0)
        whole = ref.strip()  # a shot id may itself hold "-2": try the whole string before reading an offset
        if whole in self.shots:
            return self.shots[whole][0]
        if whole.endswith(".end") and whole[:-4] in self.shots:
            return self.shots[whole[:-4]][1]
        m = re.fullmatch(r"\s*([\w-]+?)(\.end)?\s*([+-]\s*[\d.]+)?\s*", ref)
        if not m:
            raise SpecError(f"bad time reference: {ref!r}")
        name, end, off = m.groups()
        if name == "end":
            base = self.duration
        elif name in self.shots:
            base = self.shots[name][1 if end else 0]
        else:
            raise SpecError(f"unknown shot in time reference: {ref!r}")
        return base + (float(off.replace(" ", "")) if off else 0.0)

    def shot_at(self, t: float) -> str:
        for sid, (a, b) in self.shots.items():
            if a <= t < b:
                return sid
        return next(reversed(self.shots))
