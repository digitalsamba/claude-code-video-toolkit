#!/usr/bin/env python3
"""Lay sound on an ident from the events its scene wrote, and mux it onto the clip.

    uv run examples/manim-ident/sound.py examples/manim-ident/out/typed.mp4

Reads <clip>.events.json (written by tools/manim_render.py): a key click on each `key`, a whoosh across each
`morph:<seconds>`, a longer one across the `fold:<seconds>`, the key at half speed on `land`. Each sound is placed
by its own measured peak (a click) or the point where half its energy has passed (a whoosh), never by ear.
Writes <clip>.wav and <clip>-sound.mp4 and prints how far each click landed from its event."""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
SFX = Path(__file__).resolve().parent / "sfx"


def load(path, rate=1.0):
    af = f"aresample={SR}" + (f",asetrate={int(SR * rate)},aresample={SR}" if rate != 1.0 else "")
    raw = subprocess.run(["ffmpeg", "-v", "0", "-i", str(path), "-ac", "1", "-af", af, "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def peak(a):
    return np.abs(a).argmax() / SR


def centre(a):
    energy = np.cumsum(a ** 2)
    return np.searchsorted(energy, energy[-1] / 2) / SR


def db(x):
    return 10 ** (x / 20)


def build(events, seconds):
    out = np.zeros(int(seconds * SR))

    def put(a, anchor, at, gain):
        a = a / np.abs(a).max() * gain
        i = int(round((at - anchor) * SR))
        j = max(i, 0)
        a = a[j - i:]
        n = min(len(a), len(out) - j)
        if n > 0:
            out[j:j + n] += a[:n]

    keys = 0
    for e in events:
        what, _, dur = e["what"].partition(":")
        dur = float(dur or 0)
        if what == "key":       # one recording, six small changes of pitch and level, so it isn't a machine gun
            rate = [1.0, 1.06, 0.96, 1.03, 0.94, 1.08][keys % 6]
            gain = db([-13, -15, -12.5, -14.5, -13.5, -12][keys % 6])
            keys += 1
            a = load(SFX / "key.mp3", rate)
            put(a, peak(a), e["at"], gain)
        elif what == "morph":
            a = load(SFX / "whoosh-a.mp3")
            put(a, centre(a), e["at"] + dur / 2, db(-14))
        elif what == "fold":
            a = load(SFX / "whoosh-b.mp3")
            put(a, centre(a), e["at"] + dur * 0.6, db(-7))
        elif what == "land":
            a = load(SFX / "key.mp3", 0.5)
            put(a, peak(a), e["at"] + 0.03, db(-5))
    return out / max(np.abs(out).max(), 1e-9) * db(-3)


def click_offsets(out, events):
    """How far the loudest moment near each key event is from the event, in ms."""
    offs = []
    for e in events:
        if e["what"] != "key":
            continue
        i = int(e["at"] * SR)
        lo = max(i - SR // 40, 0)
        seg = np.abs(out[lo:i + SR // 40])
        offs.append(abs((lo + seg.argmax()) - i) / SR * 1000)
    return offs


def main():
    clip = Path(sys.argv[1])
    events = json.loads(clip.with_suffix(".events.json").read_text())
    seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                                   capture_output=True, text=True, check=True).stdout)
    out = build(events, seconds)
    wav = clip.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype("<i2").tobytes())
    voiced = clip.with_name(clip.stem + "-sound.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(voiced)], check=True)
    offs = click_offsets(out, events)
    print(f"{voiced}  {len(events)} sounds" + (f", clicks within {max(offs):.1f} ms of their letters" if offs else ""))


if __name__ == "__main__":
    main()
