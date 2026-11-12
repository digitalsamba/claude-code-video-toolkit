"""Sound from a compiled scene's events.json: effects on the motion, plus a simple score (stdlib only).

Runs on the host, not in Blender:  python lib/blender/sound.py OUT/events.json OUT/sound.wav

Every effect is placed at the frame Blender keyed it on, so nothing needs syncing after the fact. The score is
three moods the spec places in time ({"from": 0, "to": 3.6, "mood": "calm"}): calm (a soft maj7 pad), tension
(a detuned low drone) and resolve (a bright pad). It is synthesis, not a produced track; for a real bed, use
tools/music_gen.py at the same length and mix with tools/addmusic.py.
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import wave

SR = 44100
CLIP_GAIN = 2.5  # effects-only clips: about the level the film's mix puts the same effects at


def note(n):
    return 440 * 2 ** ((n - 69) / 12)


class Mix:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR
        self.buf = [0.0] * self.n

    def add(self, start, dur, fn, gain=1.0):
        i0 = int(start * SR)
        for i in range(max(0, i0), min(self.n, i0 + int(dur * SR))):
            self.buf[i] += gain * fn((i - i0) / SR)

    def write(self, path, seconds, normalise=True):
        # a film's mix is normalised to its peak; a clip of effects gets one fixed gain instead, so every clip
        # sits at the same level and a quiet one (two key clicks) isn't blown up to full scale
        peak = (max(abs(x) for x in self.buf) or 1) if normalise else 1 / CLIP_GAIN
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"".join(struct.pack("<h", int(32000 * max(-1.0, min(1.0, x / peak)) * 0.9)) for x in self.buf[: int(seconds * SR)]))


def tone(freq, decay=6.0, harm=(1, 0.3, 0.1)):
    return lambda t: math.exp(-decay * t) * sum(a * math.sin(2 * math.pi * freq * (k + 1) * t) for k, a in enumerate(harm))


def pad(notes, length, level=0.035):
    return lambda t: sum(level * math.sin(2 * math.pi * note(n) * t) for n in notes) * min(1, t / 0.8) * min(1, (length - t) / 0.8)


def render(events_path, out_path, seed=3, start=0.0, end=None, score=True, room=True):
    """The whole film's sound, or one stretch of it (`start`..`end` seconds: a clip's effects for Remotion,
    usually with `score=False, room=False` so the video's own music and voice carry the bed)."""
    rng = random.Random(seed)
    with open(events_path) as fh:
        ev = json.load(fh)
    end = ev["duration"] if end is None else end
    dur = end - start
    ev = {**ev, "events": [{**e, "t": e["t"] - start} for e in ev["events"] if start - 0.02 <= e["t"] < end],
          "score": [{**s, "from": max(0.0, s["from"] - start), "to": min(dur, s["to"] - start)}
                    for s in ev.get("score", []) if s["to"] > start and s["from"] < end] if score else []}
    mix = Mix(dur)
    b = 0.0
    for i in range(mix.n if room else 0):  # room tone
        b = 0.985 * b + 0.015 * rng.uniform(-1, 1)
        mix.buf[i] += 0.12 * b

    for sec in ev.get("score", []):
        t0, t1, mood = sec["from"], sec["to"], sec["mood"]
        L = t1 - t0
        if mood == "calm":
            mix.add(t0, L, pad((48, 55, 59, 64), L))
        elif mood == "tension":
            mix.add(t0, L, lambda t, L=L: (0.06 * math.sin(2 * math.pi * 98 * t) + 0.05 * math.sin(2 * math.pi * 103.5 * t))
                    * min(1, t) * min(1, (L - t) / 0.6))
        elif mood == "resolve":
            mix.add(t0, L, pad((60, 64, 67, 71), L, 0.03))

    scale = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81]
    for e in ev["events"]:
        t, k = e["t"], e["kind"]
        if k == "key_click":
            mix.add(t + rng.uniform(0, 0.01), 0.06,
                    lambda t: math.exp(-90 * t) * (rng.uniform(-1, 1) * 0.7 + 0.5 * math.sin(2 * math.pi * 190 * t)),
                    rng.uniform(0.18, 0.28))
        elif k == "tap":
            mix.add(t, 0.08, lambda t: math.exp(-60 * t) * math.sin(2 * math.pi * 150 * t), 0.4)
        elif k == "alert":
            mix.add(t, 0.35, tone(note(76), 9), 0.35)
            mix.add(t + 0.16, 0.5, tone(note(71), 7), 0.35)
        elif k == "plip":
            mix.add(t, 0.12, lambda t: math.exp(-25 * t) * math.sin(2 * math.pi * (1400 - 5000 * t) * t), 0.25)
        elif k == "gulp":
            mix.add(t, 0.2, lambda t: math.exp(-14 * t) * math.sin(2 * math.pi * (180 + 300 * t) * t), 0.35)
        elif k == "whoosh":
            mix.add(t, 0.18, lambda t: math.sin(math.pi * t / 0.18) * rng.uniform(-1, 1), 0.06)
        elif k == "progress":
            n = scale[int(e["step"] * len(scale) / max(1, e["of"]))]
            mix.add(t, 0.4, tone(note(n), 10, (1, 0.4, 0.15)), 0.16)
        elif k == "success":
            for j, n in enumerate((72, 76, 79, 84)):
                mix.add(t + 0.03 * j, 1.8, tone(note(n), 2.2, (1, 0.25)), 0.16)
        elif k == "celebrate":
            for j, n in enumerate((67, 72, 76, 79, 84, 79, 84, 88)):
                mix.add(t + 0.11 * j, 0.6, tone(note(n), 5, (1, 0.35, 0.1)), 0.15)
            for j in range(10):
                mix.add(t + 0.9 + 0.36 * j, 0.3, tone(note((48, 55, 52, 55)[j % 4]), 8, (1, 0.5)), 0.22)
        elif k == "title":
            mix.add(t, 0.1, lambda t: math.exp(-40 * t) * math.sin(2 * math.pi * (600 + 2000 * t) * t), 0.3)
    mix.write(out_path, dur, normalise=score or room)
    return out_path


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
