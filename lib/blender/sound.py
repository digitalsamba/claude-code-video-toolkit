"""Sound from a compiled scene's events.json: effects on the motion, plus a simple score (stdlib only).

Runs on the host, not in Blender:  python lib/blender/sound.py OUT/events.json OUT/sound.wav

Every effect is placed at the frame Blender keyed it on, so nothing needs syncing after the fact. The score is
three moods the spec places in time ({"from": 0, "to": 3.6, "mood": "calm"}): calm (a soft maj7 pad), tension
(a detuned low drone) and resolve (a bright pad). It is synthesis, not a produced track; for a real bed, use
tools/music_gen.py at the same length and mix with tools/addmusic.py.

A spec can name its own sounds ("sounds": {"clap": [{"file": "sfx/slate.mp3"}, {"tones": [98, 147]}]}): an event
of that kind plays those layers, each recording placed so its loudest moment is on the event's frame. Recordings
are read through ffmpeg. Such a mix is written at one fixed gain ("mix": {"gain": 0.8}), not normalised, so the
same sound is the same level in every clip.
"""
from __future__ import annotations

import json
import math
import random
import struct
import subprocess
import sys
import wave
from array import array

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


    def add_samples(self, start, xs, gain=1.0):
        i0 = int(round(start * SR))
        for k in range(max(0, -i0), min(len(xs), self.n - i0)):
            self.buf[i0 + k] += gain * xs[k]

    def write_fixed(self, path, seconds, gain=0.8, ceiling=0.89, fade=0.03):
        """One fixed gain, so a sound is the same level wherever it is used; only a mix that would clip is turned
        down (and that is said). Returns the peak written, 0-1."""
        n = int(seconds * SR)
        peak = max((abs(x) for x in self.buf[:n]), default=0.0) * gain
        if peak > ceiling:
            print(f"  note  the sound peaks at {20 * math.log10(peak):+.1f} dBFS at this gain: turned down to "
                  f"{20 * math.log10(ceiling):.1f} (lower the layers' gains or the mix's)")
            gain *= ceiling / peak
        k = int(fade * SR)
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"".join(struct.pack("<h", int(32767 * max(-1.0, min(1.0, x * gain * min(1.0, (n - i) / max(1, k))))))
                                   for i, x in enumerate(self.buf[:n])))
        return min(peak, ceiling)


def decode(path) -> list:
    """A recording as mono samples at the mix's rate (through ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    xs = array("f")
    xs.frombytes(raw[: len(raw) // 4 * 4])
    return list(xs)


def loudest(xs, window=0.05) -> int:
    """The sample a recording is placed by: the middle of the energy in its loudest 50 ms. That is the moment
    that goes on the event's frame (tools/sync_check.py finds the same 50 ms, to the nearest window)."""
    w = max(1, int(window * SR))
    if len(xs) <= w:
        at, w = 0, len(xs)
    else:
        acc = sum(x * x for x in xs[:w])
        best, at = acc, 0
        for i in range(w, len(xs)):
            acc += xs[i] * xs[i] - xs[i - w] * xs[i - w]
            if acc > best * (1 + 1e-9):
                best, at = acc, i - w + 1
    energy = sum(x * x for x in xs[at:at + w])
    if energy <= 0:
        return at + w // 2
    return int(round(sum(i * x * x for i, x in enumerate(xs[at:at + w], at)) / energy))


def lowpass(xs, hz):
    a = math.exp(-2 * math.pi * hz / SR)
    out, acc = [], 0.0
    for v in xs:
        acc = (1 - a) * v + a * acc
        out.append(acc)
    return out


def tones(freqs, length, attack=0.005, decay=0.5, hold=0.0, beat=0.0, cut=None, cut_decay=0.08):
    """A chord of sines under one envelope: attack, an optional hold, then a decay, or a quick cut at `cut` s.
    `beat` adds a second copy that much sharp (1.004), so the chord moves a little instead of sitting flat."""
    out = []
    voices = [(f, 0.6 / (1 + 0.6 * i)) for i, f in enumerate(freqs)] + \
             [(f * beat, 0.4 / (1 + 0.6 * i)) for i, f in enumerate(freqs)] if beat else \
             [(f, 1 / (1 + 0.6 * i)) for i, f in enumerate(freqs)]
    for k in range(int(length * SR)):
        t = k / SR
        env = min(t / max(attack, 1e-4), 1.0) * (math.exp(-(t - attack - hold) / decay) if t > attack + hold else 1.0)
        if cut is not None and t > cut:
            env *= math.exp(-(t - cut) / cut_decay)
        if env < 1e-4 and t > attack:
            break
        out.append(env * sum(g * math.sin(2 * math.pi * f * t) for f, g in voices))
    return out


def play_layers(mix, layers, t, until=None, cache={}):
    """One named sound at time t: each recording with its loudest moment on t, each chord starting on t."""
    for layer in layers:
        gain = float(layer.get("gain", 1.0 if "file" in layer else 0.2))
        if "file" in layer:
            key = (layer["file"], layer.get("lowpass"))
            if key not in cache:
                xs = decode(layer["file"])
                cache[key] = (lowpass(xs, float(layer["lowpass"])) if layer.get("lowpass") else xs, loudest(xs))
            xs, peak = cache[key]
            mix.add_samples(t - peak / SR, xs, gain)
        else:
            length = mix.n / SR - t
            xs = tones(layer["tones"], length, float(layer.get("attack", 0.005)), float(layer.get("decay", 0.5)),
                       float(layer.get("hold", 0.0)), float(layer.get("beat", 0.0)),
                       None if until is None else until - t)
            mix.add_samples(t, xs, gain)


def tone(freq, decay=6.0, harm=(1, 0.3, 0.1)):
    return lambda t: math.exp(-decay * t) * sum(a * math.sin(2 * math.pi * freq * (k + 1) * t) for k, a in enumerate(harm))


def pad(notes, length, level=0.035):
    return lambda t: sum(level * math.sin(2 * math.pi * note(n) * t) for n in notes) * min(1, t / 0.8) * min(1, (length - t) / 0.8)


def render(events_path, out_path, seed=3, start=0.0, end=None, score=True, room=True, gains=None):
    """The whole film's sound, or one stretch of it (`start`..`end` seconds: a clip's effects for Remotion,
    usually with `score=False, room=False` so the video's own music and voice carry the bed). `gains` scales one
    kind of event against the rest, e.g. {"step": 0.35} for footsteps 9 dB down under the door."""
    rng = random.Random(seed)
    with open(events_path) as fh:
        ev = json.load(fh)
    end = ev["duration"] if end is None else end
    dur = end - start
    ev = {**ev, "events": [{**e, "t": e["t"] - start} for e in ev["events"] if start - 0.02 <= e["t"] < end],
          "score": [{**s, "from": max(0.0, s["from"] - start), "to": min(dur, s["to"] - start)}
                    for s in ev.get("score", []) if s["to"] > start and s["from"] < end] if score else []}
    named = ev.get("sounds") or {}
    if named and not (ev.get("mix") or {}).get("room"):
        room = False  # a spec with its own sounds is not sitting in a room unless it says so
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
    level = gains or {}
    for e in ev["events"]:
        t, k = e["t"], e["kind"]
        g = level.get(k, 1.0)
        if k in named:
            play_layers(mix, named[k], t, None if e.get("until") is None else e["until"] - start)
        elif k == "key_click":
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
        elif k == "step":  # a soft footfall: a low thud and a little scuff
            mix.add(t, 0.09, lambda t: math.exp(-45 * t) * (math.sin(2 * math.pi * (95 - 200 * t) * t)
                                                            + 0.35 * rng.uniform(-1, 1)), g * rng.uniform(0.2, 0.28))
        elif k in ("door_open", "door_close"):  # the latch, then the hinge
            mix.add(t, 0.05, lambda t: math.exp(-80 * t) * math.sin(2 * math.pi * 900 * t), g * 0.25)
            mix.add(t + 0.08, 0.6, lambda t: math.sin(math.pi * t / 0.6) * 0.5
                    * math.sin(2 * math.pi * (240 + 90 * math.sin(11 * t)) * t), g * 0.05)
        elif k == "switch":  # a wall switch: a short hard click
            mix.add(t, 0.03, lambda t: math.exp(-160 * t) * (math.sin(2 * math.pi * 2400 * t) + rng.uniform(-1, 1)), g * 0.3)
        elif k == "lamp_stutter":  # a tube starting: ticks and a buzz at twice the mains
            for j, dt in enumerate((0.0, 0.09, 0.21, 0.26, 0.38)):
                mix.add(t + dt, 0.025, lambda t: math.exp(-120 * t) * rng.uniform(-1, 1), g * 0.12)
            mix.add(t, 0.45, lambda t: math.sin(2 * math.pi * 100 * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 7 * t)), g * 0.03)
        elif k == "lamp_on":  # the tube catches: a soft low clunk
            mix.add(t, 0.15, lambda t: math.exp(-30 * t) * (math.sin(2 * math.pi * (80 - 120 * t) * t)
                                                            + 0.3 * rng.uniform(-1, 1)), g * 0.3)
        elif k in ("lid_open", "lid_close"):  # a laptop lid: a soft knock when it shuts, a faint creak opening
            if k == "lid_close":
                mix.add(t, 0.12, lambda t: math.exp(-38 * t) * (math.sin(2 * math.pi * (140 - 300 * t) * t)
                                                                + 0.4 * rng.uniform(-1, 1)), g * 0.45)
            else:
                mix.add(t, 0.3, lambda t: math.sin(math.pi * t / 0.3) * rng.uniform(-1, 1), g * 0.03)
        elif k in ("chair", "rustle"):  # cloth and chair legs: filtered noise that swells and goes
            mix.add(t, 0.5, lambda t: math.sin(math.pi * t / 0.5) * rng.uniform(-1, 1), g * (0.05 if k == "rustle" else 0.09))
        elif k == "title":
            mix.add(t, 0.1, lambda t: math.exp(-40 * t) * math.sin(2 * math.pi * (600 + 2000 * t) * t), 0.3)
    if named:
        mix.write_fixed(out_path, dur, float((ev.get("mix") or {}).get("gain", 0.8)))
    else:
        mix.write(out_path, dur, normalise=score or room)
    return out_path


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
