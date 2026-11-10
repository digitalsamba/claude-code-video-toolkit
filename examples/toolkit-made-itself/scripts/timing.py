"""Lay the voice phrases out on the timeline: src/timing.json (frames at 30fps).

Run after regenerating any phrase in public/audio/scenes (then rerun captions.py). A scene whose
phrase isn't voiced yet uses its ESTIMATE, so the timeline and the music cuts exist before it is.
LEAD frames before a scene's phrase, TAIL after it (the scene's own action lives in that room).
"""
import json, pathlib, subprocess
root = pathlib.Path(__file__).resolve().parent.parent
scenes = root / "public/audio/scenes"
FPS = 30
def frames(f):
    d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)]))
    return round(d * FPS)
# scene: (phrase, lead, tail, estimate in seconds if not voiced)
LAYOUT = {
    "open":     ("01_open",     75, 30, None),   # the sentence types itself before Marilyn speaks
    "narrate":  ("02_narrate",  60, 30, None),   # the voiceover command runs first
    "score":    ("03_score",    55, 45, None),
    "generate": ("04_generate", 30, 45, 9.0),
    "compose":  ("05_compose",  30, 45, None),
    "check":    ("06_check",    75, 45, 11.0),   # qc runs and prints before she speaks
    "close":    ("07_close",    20, 60, None),
}
out = {"phrases": {}, "scenes": {}, "estimated": [], "starts": {}}
t0 = 0
for scene, (phrase, lead, tail, est) in LAYOUT.items():
    f = scenes / f"{phrase}.mp3"
    n = frames(f) if f.exists() else round(est * FPS)
    if not f.exists():
        out["estimated"].append(scene)
    out["phrases"][phrase] = lead
    out["scenes"][scene] = lead + n + tail
    out["starts"][scene] = t0
    t0 += out["scenes"][scene]
out["total"] = t0
(root / "src/timing.json").write_text(json.dumps(out, indent=2))
print(json.dumps({"total_s": round(t0 / FPS, 1), "starts_s": {k: round(v / FPS, 1) for k, v in out["starts"].items()}, "estimated": out["estimated"]}))
