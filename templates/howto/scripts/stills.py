"""Stills of an episode before any render, tiled into one sheet: the quick look at every scene's layout.

    uv run scripts/stills.py                                    # a late frame of every segment of the episode
    uv run scripts/stills.py --short 3                          # the same for chapter 3's Short
    uv run scripts/stills.py --at free:200,setup:40             # chosen frames, counted from a scene's start
    uv run scripts/stills.py --at "free@thirty dollars@2"       # 2 frames after the narration says "thirty dollars"
    uv run scripts/stills.py --episode ep1 --name clicks --at ...  # one episode of a project that keeps several

Frames come from the timeline the render plays (scripts/timeline.mjs), and a cue is found in the narration's
word times (src/timing.json) the way cue() finds it, so a still is the frame the render will have. The project
is bundled once (out/bundle) and every still is taken from the bundle. Stills and the sheet go to
out/stills/<composition>-<name>/ and out/stills/<composition>-<name>.jpg.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
LATE = 12  # a late frame: this many frames before a segment ends, when everything in it has landed


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    # npx is npx.cmd on Windows: resolve it so no shell is needed
    exe = shutil.which(cmd[0]) or cmd[0]
    return subprocess.run([exe, *cmd[1:]], cwd=ROOT, capture_output=True, text=True)


def read_timeline(episode: str | None, short: int | None) -> dict:
    cmd = ["node", str(ROOT / "scripts" / "timeline.mjs"), *([episode] if episode else []), *(["--short", str(short)] if short else [])]
    out = run(cmd)
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip() or "scripts/timeline.mjs failed")
    return json.loads(out.stdout)


def words_of(episode: str | None, scene: str) -> list[tuple[str, int]]:
    timing = json.loads((ROOT / "src" / "timing.json").read_text(encoding="utf-8"))
    scenes = timing["episodes"][episode]["scenes"] if episode else timing["scenes"]
    if scene not in scenes:
        raise SystemExit(f"{scene}: no narration in src/timing.json")
    norm = lambda t: re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("’", "'")).split()
    return [(w, p["at"] + round(c["startMs"] * FPS / 1000)) for p in scenes[scene]["phrases"] for c in p["words"] for w in norm(c["text"])]


def cue(episode: str | None, scene: str, phrase: str) -> int:
    """The frame, from the scene's start, that a phrase starts on"""
    words = words_of(episode, scene)
    target = re.sub(r"[^a-z0-9' ]+", " ", phrase.lower()).split()
    for i in range(len(words) - len(target) + 1):
        if [w for w, _ in words[i : i + len(target)]] == target:
            return words[i][1]
    raise SystemExit(f"{scene}: '{phrase}' is not said")


def name_of(seg: dict) -> str:
    if seg["kind"] == "scene":
        return seg["id"]
    if seg["kind"] == "card":
        return f"card{seg['chapter'] + 1}"
    return seg["kind"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", help="In a project that keeps several episodes, the id of one")
    ap.add_argument("--short", type=int, help="A chapter's Short (from 1) instead of the episode")
    ap.add_argument("--at", default="", help="scene:frame or scene@phrase@offset, comma separated; default a late frame of every segment")
    ap.add_argument("--name", default="late", help="Names the folder and the sheet (default late)")
    ap.add_argument("--no-bundle", action="store_true", help="Use out/bundle as it is (after a first run, when only the frames changed)")
    a = ap.parse_args()

    tl = read_timeline(a.episode, a.short)
    starts: dict[str, dict] = {}
    for seg in tl["segments"]:
        starts.setdefault(name_of(seg), seg)  # a scene id appears once; 'ident' can twice: the first is kept

    want: list[tuple[str, int]] = []
    if a.at:
        for item in a.at.split(","):
            if "@" in item:
                scene, phrase, off = item.split("@")
                want.append((scene, cue(a.episode, scene, phrase) + int(off)))
            else:
                scene, frame = item.split(":")
                want.append((scene, int(frame)))
    else:
        want = [(name_of(s), max(0, s["frames"] - LATE)) for s in tl["segments"]]
    for scene, _ in want:
        if scene not in starts:
            raise SystemExit(f"{scene}: not in this timeline; it has {', '.join(starts)}")

    tag = f"-{a.episode}" if a.episode else ""
    comp = f"Short{tag}-{a.short}" if a.short else f"Episode{tag}"
    bundle = ROOT / "out" / "bundle"
    if not a.no_bundle or not bundle.exists():
        print("Bundling the project once ...")
        out = run(["npx", "remotion", "bundle", "src/index.ts", "--out-dir", str(bundle), "--log=error"])
        if out.returncode != 0:
            raise SystemExit(out.stderr.strip() or "remotion bundle failed")

    folder = ROOT / "out" / "stills" / f"{comp}-{a.name}"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for i, (scene, off) in enumerate(want):
        frame = starts[scene]["start"] + off
        out = run(["npx", "remotion", "still", str(bundle), comp, str(folder / f"{i:02d}.png"), f"--frame={frame}", "--log=error"])
        if out.returncode != 0:
            raise SystemExit(out.stderr.strip() or f"remotion still failed at frame {frame}")
        print(f"{i:02d}  {scene}+{off} = frame {frame}")

    cols = 4 if a.short else 2
    rows = -(-len(want) // cols)
    sheet = folder.parent / f"{comp}-{a.name}.jpg"
    out = run([
        "ffmpeg", "-v", "error", "-y", "-i", str(folder / "%02d.png"),
        "-vf", f"scale={405 if a.short else 960}:-1,tile={cols}x{rows}", "-frames:v", "1", "-q:v", "3", str(sheet),
    ])
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip() or "ffmpeg could not tile the stills")
    print(sheet.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
