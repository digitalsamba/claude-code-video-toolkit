"""Make the template's three sounds: public/audio/sfx/tick.wav, land.wav and move.wav.

    uv run templates/howto/scripts/sounds.py

They are synthesised here, so they are free to use and the same on every machine. An episode plays them where
things land: a tick as a row or a bar arrives, a low note under a figure, a soft pass as the playhead moves to a
chapter. `sounds: false` in src/episode.ts turns them off. To use other sounds, replace the files (keep the
names, and keep each sound's loudest moment in its first 30 ms: it is placed on the frame the thing lands).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000
OUT = Path(__file__).resolve().parent.parent / "public" / "audio" / "sfx"


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-np.maximum(0, t - attack) / decay)


def lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * hz / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def tick() -> np.ndarray:
    """A soft wooden tap: two short partials and a breath of noise."""
    n = int(SR * 0.09)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    body = np.sin(2 * np.pi * 1150 * t) * env(n, 0.001, 0.012) + 0.5 * np.sin(2 * np.pi * 2300 * t) * env(n, 0.001, 0.006)
    click = lowpass(rng.standard_normal(n), 5000) * env(n, 0.0005, 0.003)
    return 0.5 * body + 0.35 * click


def land() -> np.ndarray:
    """A low, round note that falls a little: something with weight arriving."""
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    f = 62 + 46 * np.exp(-t / 0.035)
    phase = 2 * np.pi * np.cumsum(f) / SR
    rng = np.random.default_rng(11)
    tap = lowpass(rng.standard_normal(n), 1800) * env(n, 0.001, 0.008)
    return 0.8 * np.sin(phase) * env(n, 0.004, 0.11) + 0.25 * tap


def move() -> np.ndarray:
    """A soft pass of air, rising then falling, as long as the playhead takes to travel (18 frames)."""
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    bright = lowpass(noise, 2600) - lowpass(noise, 500)
    shape = np.sin(np.pi * np.minimum(1, t / 0.6)) ** 2
    return 0.6 * bright * shape


def write(name: str, x: np.ndarray, peak_db: float) -> None:
    x = x / np.max(np.abs(x)) * 10 ** (peak_db / 20)
    fade = int(SR * 0.004)
    x[-fade:] *= np.linspace(1, 0, fade)
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    print(f"public/audio/sfx/{name}.wav  {len(x) / SR:.2f} s, peak {peak_db:.0f} dBFS")


def main() -> int:
    write("tick", tick(), -14)
    write("land", land(), -10)
    write("move", move(), -20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
