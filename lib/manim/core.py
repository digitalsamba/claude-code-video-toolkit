"""The parts of the Manim helpers that need no Manim: narration cues and a brand's colours.

Kept apart from toolkit_manim.py so they can be tested, and used by other tools, without Manim installed."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("MANIM_TOOLKIT_ROOT") or Path(__file__).resolve().parents[2])


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


class Voice:
    """Word timings from a `<name>.captions.json` (tools/captions.py), the same file Remotion's cue() reads.

    `cue("the cache", nth=2)` is the second at which that phrase starts for the second time. A phrase the narration
    does not say raises, so a scene can't be timed to a word that was cut from the script."""

    def __init__(self, captions=None, offset: float = 0.0):
        captions = captions or os.environ.get("MANIM_TOOLKIT_CAPTIONS")
        if not captions:
            raise FileNotFoundError("Voice() needs a .captions.json: pass a path, or render with --captions")
        words = json.loads(Path(captions).read_text())
        self.words = [(_norm(w["text"]), w["startMs"] / 1000, w["endMs"] / 1000) for w in words if _norm(w["text"])]
        self.offset = offset
        self.end = (self.words[-1][2] if self.words else 0.0) + offset

    def cue(self, phrase: str, nth: int = 1, lead: float = 0.0) -> float:
        want = _norm(phrase).split()
        if not want:
            raise KeyError("cue needs a phrase")
        said = [w[0] for w in self.words]
        hits = [self.words[i][1] for i in range(len(said) - len(want) + 1) if said[i:i + len(want)] == want]
        if len(hits) < nth:
            raise KeyError(f"cue {phrase!r} (#{nth}) is not in the narration ({len(hits)} found)")
        return hits[nth - 1] + self.offset - lead


class Brand:
    """A brand's colours and mark, from brands/<name>/brand.json. `brand.c("primary")` is a hex string."""

    def __init__(self, name: str | None = None):
        self.name = name or os.environ.get("MANIM_TOOLKIT_BRAND") or "default"
        self.dir = ROOT / "brands" / self.name
        path = self.dir / "brand.json"
        if not path.exists():
            raise FileNotFoundError(f"no brand at {path}")
        self.data = json.loads(path.read_text())
        self.colors = {k: v for k, v in self.data.get("colors", {}).items() if isinstance(v, str) and v.startswith("#")}

    def c(self, key: str, fallback: str | None = None) -> str:
        if key in self.colors:
            return self.colors[key]
        if fallback is not None:
            return fallback
        raise KeyError(f"brand {self.name!r} has no colour {key!r} (it has {', '.join(sorted(self.colors))})")

    def mark(self, variant: str = "mark") -> Path:
        """brands/<name>/assets/mark.svg, or mark-on-dark / mark-white for the other grounds."""
        path = self.dir / "assets" / f"{variant}.svg"
        if not path.exists():
            raise FileNotFoundError(f"brand {self.name!r} has no {path.name}")
        return path


def frame_size() -> tuple[int, int]:
    """The pixel size the render was asked for (set by tools/manim_render.py), 1920x1080 otherwise."""
    w, _, h = os.environ.get("MANIM_TOOLKIT_SIZE", "1920x1080").partition("x")
    return int(w), int(h)
