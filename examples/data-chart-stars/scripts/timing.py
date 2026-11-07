"""Lay the voice phrases out on the timeline: src/timing.json (frames at 30fps).

Run after regenerating or re-picking any phrase in public/audio/scenes (then rerun captions.py).
LEAD frames before the first phrase of a scene, GAP between phrases, TAIL after the last.
"""
import json, pathlib, subprocess
root = pathlib.Path(__file__).resolve().parent.parent
scenes = root / "public/audio/scenes"
def frames(f):
    d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)]))
    return round(d * 30)
LEAD, GAP, TAIL = 20, 14, 40
layout = {"stars": ["01a_intro", "01b_slow", "01c_spike", "01d_today"], "months": ["02a_march", "02b_october"]}
out = {"phrases": {}, "scenes": {}}
for scene, files in layout.items():
    t = LEAD
    for f in files:
        out["phrases"][f] = t
        t += frames(scenes / f"{f}.mp3") + GAP
    out["scenes"][scene] = t - GAP + TAIL
(root / "src/timing.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
