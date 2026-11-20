#!/usr/bin/env python3
"""
Thai narration with PD Clone Voice (https://app.pd-voiceclone.com).

Built for a small daily quota (the free account allows 20 TTS calls a day):

  * Scenes are joined and sent in as few calls as possible (2,000 characters
    each at most), then the returned WAV is cut back into one file per scene at
    the silences between sentences.
  * Every part is trimmed of silence at the head and tail.
  * English words are rewritten as Thai readings before voicing (Claude becomes
    คล็อด). The spoken text and the original caption text are both kept in the
    manifest, so captions keep the English spelling.
  * Any Latin word left without a reading stops the run before a call is spent.
  * Each part's length is checked against its text: a long line with short
    audio usually means a word was skipped, and is flagged.
  * HTTP 402 or 429 means today's quota is used up. The tool stops at once and
    never retries.

Usage:
    uv run tools/pdclone_tts.py --list-voices
    uv run tools/pdclone_tts.py --text "สวัสดีครับ" --output hello.wav
    uv run tools/pdclone_tts.py --scenes scenes.json --out-dir public/audio/scenes
    uv run tools/pdclone_tts.py --scene-dir public/audio/scenes --out-dir public/audio/scenes
    uv run tools/pdclone_tts.py --scenes scenes.json --out-dir out --dry-run   # no call

Scenes file: a JSON list of {"id": ..., "text": ...} (or "narration"), or an
object with a "scenes" list in that shape. A scene dir holds one .txt per scene.

Key: PDCLONE_API_KEY in .env, or PD_VOICE_KEY in the environment. The key is
never printed or written anywhere by this tool.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

BASE_URL = "https://app.pd-voiceclone.com/api/v1"
VOICES_URL = f"{BASE_URL}/voices"
TTS_URL = f"{BASE_URL}/tts"
USER_AGENT = "video-toolkit-pdclone-tts/1.0"

MAX_CHARS = 2000            # per request, set by the account rules
QUOTA_STATUSES = (402, 429)

# Thai readings for English words. Extend per project with --lexicon.
DEFAULT_LEXICON = {
    "Claude": "คล็อด",
    "Claude Code": "คล็อด โค้ด",
    "ChatGPT": "แชตจีพีที",
    "OpenAI": "โอเพ่นเอไอ",
    "Anthropic": "แอนโทรปิก",
    "Google": "กูเกิล",
    "Gemini": "เจมิไน",
    "AI": "เอไอ",
    "API": "เอพีไอ",
    "YouTube": "ยูทูบ",
    "TikTok": "ติ๊กต็อก",
    "Facebook": "เฟซบุ๊ก",
    "Instagram": "อินสตาแกรม",
    "app": "แอป",
    "online": "ออนไลน์",
    "video": "วิดีโอ",
}

LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'.+-]*")


class QuotaExceeded(RuntimeError):
    pass


# ---------- key and HTTP ----------

def get_api_key() -> str | None:
    """PDCLONE_API_KEY from .env first, then PD_VOICE_KEY from the environment."""
    try:
        from dotenv import dotenv_values
        root = Path(__file__).resolve().parent.parent
        val = dotenv_values(root / ".env").get("PDCLONE_API_KEY")
        if val:
            return val.strip()
    except ImportError:
        pass
    val = os.environ.get("PDCLONE_API_KEY") or os.environ.get("PD_VOICE_KEY")
    return val.strip() if val else None


def _headers(api_key: str) -> dict:
    return {"X-API-Key": api_key, "User-Agent": USER_AGENT}


def _check_quota(resp: requests.Response) -> None:
    if resp.status_code in QUOTA_STATUSES:
        raise QuotaExceeded(
            f"HTTP {resp.status_code}: โควต้าวันนี้หมดแล้ว (daily quota used up). "
            "Stopped without retrying."
        )


def list_voices(api_key: str) -> dict:
    resp = requests.get(VOICES_URL, headers=_headers(api_key), timeout=30)
    _check_quota(resp)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def resolve_voice(api_key: str, voice: str | None) -> tuple[str, str]:
    """Return (reference_id, name). None falls back to PDCLONE_VOICE_ID, then
    the account's default voice."""
    if not voice:
        try:
            from dotenv import dotenv_values
            root = Path(__file__).resolve().parent.parent
            voice = dotenv_values(root / ".env").get("PDCLONE_VOICE_ID")
        except ImportError:
            pass
        voice = voice or os.environ.get("PDCLONE_VOICE_ID")
        if voice == "your_voice_id_here":
            voice = None
    data = list_voices(api_key)
    voices = data.get("voices", [])
    if not voice:
        ref = data.get("default_reference_id")
        for v in voices:
            if v.get("reference_id") == ref or (not ref and v.get("is_default")):
                return v["reference_id"], v.get("name", "")
        if voices:
            return voices[0]["reference_id"], voices[0].get("name", "")
        raise RuntimeError("No voices in this account")
    for v in voices:
        if voice in (v.get("reference_id"), v.get("name")):
            return v["reference_id"], v.get("name", "")
    names = ", ".join(v.get("name", "?") for v in voices)
    raise RuntimeError(f"Voice '{voice}' not found. Available: {names}")


def synthesize(api_key: str, text: str, reference_id: str, speed: float,
               language: str, out_wav: Path, timeout: int = 180) -> None:
    """One TTS call. Writes the returned WAV. Raises QuotaExceeded on 402/429."""
    resp = requests.post(
        TTS_URL,
        headers=_headers(api_key),
        data={"text": text, "language": language,
              "reference_id": reference_id, "speed": str(speed)},
        timeout=timeout,
    )
    _check_quota(resp)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.content
    if body[:4] != b"RIFF":
        raise RuntimeError(
            f"Expected WAV, got {resp.headers.get('content-type')}: {body[:200]!r}")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(body)


# ---------- text ----------

def load_lexicon(path: str | None) -> dict:
    lex = dict(DEFAULT_LEXICON)
    if path:
        lex.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return lex


def to_spoken(text: str, lexicon: dict) -> tuple[str, list[str]]:
    """Replace English words with Thai readings. Returns (spoken, leftover words)."""
    out = text
    # Longest keys first so "Claude Code" wins over "Claude".
    for word in sorted(lexicon, key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                         re.IGNORECASE)
        out = pat.sub(lexicon[word], out)
    leftover = sorted(set(LATIN_WORD.findall(out)))
    return out, leftover


def load_scenes(scenes: str | None, scene_dir: str | None) -> list[dict]:
    if scene_dir:
        files = sorted(Path(scene_dir).glob("*.txt"))
        return [{"id": f.stem, "text": f.read_text(encoding="utf-8").strip()}
                for f in files if f.read_text(encoding="utf-8").strip()]
    data = json.loads(Path(scenes).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("scenes", [])
    out = []
    for i, s in enumerate(data, 1):
        text = (s.get("text") or s.get("narration") or "").strip()
        if text:
            key = str(s.get("id", f"scene{i:02d}"))
            # concept-explainer-short reads audio/scenes/<id>_<slug>.mp3
            sid = f"{key}_{s['slug']}" if s.get("slug") else key
            out.append({"id": sid, "key": key, "text": text})
    return out


def chunk_scenes(scenes: list[dict], limit: int = MAX_CHARS) -> list[list[dict]]:
    """Group scenes in order so each group's joined spoken text fits the limit."""
    groups, cur, size = [], [], 0
    for s in scenes:
        n = len(s["spoken"]) + 1
        if n > limit:
            raise ValueError(f"Scene {s['id']} is {n} characters, over {limit}")
        if cur and size + n > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += n
    if cur:
        groups.append(cur)
    return groups


# ---------- audio ----------

def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def detect_silences(path: Path, noise_db: float, min_sil: float) -> list[tuple[float, float]]:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    total = duration(path)
    if len(ends) < len(starts):
        ends.append(total)
    merged: list[tuple[float, float]] = []
    for a, b in zip(starts, ends):
        if merged and a - merged[-1][1] < 0.05:   # touching pauses are one pause
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def choose_cuts(silences: list[tuple[float, float]], weights: list[int],
                total: float) -> list[float] | None:
    """Pick len(weights)-1 silences in order, closest to where the character
    counts say each boundary should fall. Returns cut times (silence midpoints)."""
    k = len(weights) - 1
    if k == 0:
        return []
    # Ignore leading and trailing silence.
    inner = [(a, b) for a, b in silences if a > 0.05 and b < total - 0.05]
    if len(inner) < k:
        return None
    tot_w = sum(weights)
    acc, targets = 0, []
    for w in weights[:-1]:
        acc += w
        targets.append(total * acc / tot_w)
    mids = [(a + b) / 2 for a, b in inner]
    lens = [b - a for a, b in inner]
    m = len(mids)
    INF = float("inf")
    # cost[i][j]: best cost placing cut i at silence j. Longer silences cost less.
    cost = [[INF] * m for _ in range(k)]
    back = [[-1] * m for _ in range(k)]
    for j in range(m):
        cost[0][j] = abs(mids[j] - targets[0]) - 0.25 * lens[j]
    for i in range(1, k):
        best, best_j = INF, -1
        for j in range(m):
            if j > 0 and cost[i - 1][j - 1] < best:
                best, best_j = cost[i - 1][j - 1], j - 1
            if best_j >= 0:
                cost[i][j] = best + abs(mids[j] - targets[i]) - 0.25 * lens[j]
                back[i][j] = best_j
    j = min(range(m), key=lambda x: cost[k - 1][x])
    if cost[k - 1][j] == INF:
        return None
    picks = [j]
    for i in range(k - 1, 0, -1):
        j = back[i][j]
        picks.append(j)
    return [mids[p] for p in reversed(picks)]


def cut_and_trim(src: Path, start: float, end: float, out: Path, noise_db: float) -> None:
    """Cut [start, end) and trim silence from both ends."""
    trim = (f"silenceremove=start_periods=1:start_threshold={noise_db}dB:start_silence=0.05,"
            f"areverse,silenceremove=start_periods=1:start_threshold={noise_db}dB:start_silence=0.05,"
            "areverse")
    fmt = ["-c:a", "pcm_s16le"] if out.suffix == ".wav" else ["-q:a", "2"]
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ss", f"{start:.3f}",
                        "-to", f"{end:.3f}", "-af", trim, *fmt, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {out.name}: {r.stderr[-300:]}")


def flag_short(parts: list[dict], max_ratio: float = 1.6, max_cps: float = 28.0) -> None:
    """Mark parts whose characters per second is far above the rest."""
    rates = sorted(p["chars_per_sec"] for p in parts)
    median = rates[len(rates) // 2] if rates else 0
    for p in parts:
        reasons = []
        if p["chars_per_sec"] > max_cps:
            reasons.append(f"{p['chars_per_sec']} chars/s is too fast to be real speech")
        if len(parts) >= 3 and median and p["chars_per_sec"] > median * max_ratio:
            reasons.append(f"{p['chars_per_sec']} chars/s vs median {median}")
        if reasons:
            p["suspect"] = "; ".join(reasons) + ": audio may have skipped words"


# ---------- main flows ----------

def run_batch(args, api_key: str, scenes: list[dict], lexicon: dict) -> dict:
    blocked = {}
    for s in scenes:
        s["spoken"], left = to_spoken(s["text"], lexicon)
        if left:
            blocked[s["id"]] = left
    if blocked and not args.allow_latin:
        return {"success": False, "error": "English words without a Thai reading",
                "words": blocked,
                "fix": "Add them to a lexicon JSON ({\"word\": \"คำอ่าน\"}) and pass --lexicon"}

    groups = chunk_scenes(scenes)
    plan = {"requests": len(groups),
            "groups": [[s["id"] for s in g] for g in groups],
            "chars": [sum(len(s["spoken"]) for s in g) for g in groups]}
    if args.dry_run:
        return {"success": True, "dry_run": True, **plan,
                "scenes": [{"id": s["id"], "caption": s["text"], "spoken": s["spoken"]}
                           for s in scenes]}

    ref, name = resolve_voice(api_key, args.voice)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_raw"
    ext = args.format
    parts = []
    calls = 0
    for gi, group in enumerate(groups, 1):
        joined = args.joiner.join(s["spoken"] for s in group)
        raw = raw_dir / f"batch{gi:02d}.wav"
        key_file = raw.with_suffix(".json")
        key = {"text": joined, "ref": ref, "speed": args.speed, "language": args.language}
        reuse = (raw.exists() and key_file.exists()
                 and json.loads(key_file.read_text(encoding="utf-8")) == key)
        if reuse:
            print(f"  batch {gi}/{len(groups)}: same text as the saved WAV, no call",
                  file=sys.stderr)
        else:
            print(f"  call {gi}/{len(groups)}: {len(group)} scene(s), {len(joined)} chars",
                  file=sys.stderr)
            synthesize(api_key, joined, ref, args.speed, args.language, raw)
            key_file.write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
            calls += 1
        total = duration(raw)
        weights = [len(s["spoken"]) for s in group]
        # Every pause is a candidate; position against the character counts
        # decides, length only breaks ties. Long pauses alone can miss a scene
        # break the voice took quickly.
        cuts = choose_cuts(detect_silences(raw, args.noise_db, args.min_silence),
                           weights, total)
        if cuts is None:
            return {"success": False, "partial": parts, "raw": str(raw),
                    "error": f"Batch {gi}: found fewer pauses than scene breaks. "
                             "The raw WAV is kept; lower --min-silence or end each "
                             "scene with a full stop before calling again."}
        bounds = [0.0, *cuts, total]
        for s, a, b in zip(group, bounds[:-1], bounds[1:]):
            out = out_dir / f"{s['id']}.{ext}"
            cut_and_trim(raw, a, b, out, args.noise_db)
            d = duration(out)
            parts.append({"id": s["id"], "key": s.get("key", s["id"]), "file": str(out), "duration_seconds": round(d, 2),
                          "frames_30fps": int(round(d * 30)), "caption": s["text"],
                          "spoken": s["spoken"], "chars": len(s["spoken"]),
                          "chars_per_sec": round(len(s["spoken"]) / d, 1) if d else 0})
    flag_short(parts)
    manifest = {"success": True, "provider": "pdclone", "voice": name,
                "reference_id": ref, "speed": args.speed, "requests": calls,
                "scenes": parts}
    (out_dir / "pdclone-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.vo_durations:
        # Same shape gen_vo.py writes: {scene id: seconds}, read by build.py.
        Path(args.vo_durations).write_text(json.dumps(
            {p["key"]: p["duration_seconds"] for p in parts}, indent=2))
    return manifest


def parse_args():
    p = argparse.ArgumentParser(
        description="Thai narration with PD Clone Voice, batched to save quota",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--list-voices", action="store_true", help="List voices and exit")
    p.add_argument("--text", help="Single line to voice")
    p.add_argument("--output", help="Output file for --text (.wav or .mp3)")
    p.add_argument("--scenes", help="Scenes JSON (list of {id, text})")
    p.add_argument("--scene-dir", help="Directory of one .txt per scene")
    p.add_argument("--out-dir", help="Where per-scene files and the manifest go")
    p.add_argument("--voice", help="Voice name or reference_id (default: the account's ⭐ voice)")
    p.add_argument("--speed", type=float, default=1.0, help="0.5 to 2.0 (default 1.0)")
    p.add_argument("--language", default="th", help="Language code (default th)")
    p.add_argument("--lexicon", help="JSON of English word to Thai reading, added to the defaults")
    p.add_argument("--allow-latin", action="store_true",
                   help="Voice text even if English words are left without a reading")
    p.add_argument("--format", choices=["wav", "mp3"], default="mp3", help="Per-scene format")
    p.add_argument("--noise-db", type=float, default=-38.0, help="Silence threshold (default -38)")
    p.add_argument("--min-silence", type=float, default=0.12,
                   help="Shortest pause that can be a scene break, seconds (default 0.12)")
    p.add_argument("--vo-durations",
                   help="Also write {scene id: seconds} here (concept-explainer-short's vo_durations.json)")
    p.add_argument("--joiner", default=" ... ",
                   help="Text put between scenes so the voice pauses (default ' ... ')")
    p.add_argument("--dry-run", action="store_true", help="Show the plan, spend no call")
    p.add_argument("--json", action="store_true", help="Print the result as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0.5 <= args.speed <= 2.0:
        sys.exit("Error: --speed must be between 0.5 and 2.0")
    api_key = get_api_key()
    if not api_key and not args.dry_run:
        sys.exit("Error: no key. Add PDCLONE_API_KEY to .env (or set PD_VOICE_KEY).")
    lexicon = load_lexicon(args.lexicon)

    try:
        if args.list_voices:
            data = list_voices(api_key)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(f"Voices ({data.get('total', len(data.get('voices', [])))}):")
                for v in data.get("voices", []):
                    star = " ⭐" if v.get("is_default") else ""
                    emo = ", ".join(v.get("emotions") or [])
                    print(f"  {v.get('name')}{star}  {v.get('reference_id')}  {emo}")
            return

        if args.text:
            if not args.output:
                sys.exit("Error: --output is required with --text")
            out = Path(args.output)
            args.out_dir = str(out.parent)
            scenes = [{"id": out.stem, "text": args.text}]
            args.format = out.suffix.lstrip(".") or "mp3"
        elif args.scenes or args.scene_dir:
            if not args.out_dir:
                sys.exit("Error: --out-dir is required with --scenes or --scene-dir")
            scenes = load_scenes(args.scenes, args.scene_dir)
            if not scenes:
                sys.exit("Error: no scene text found")
        else:
            sys.exit("Error: give --text, --scenes, --scene-dir or --list-voices")

        result = run_batch(args, api_key, scenes, lexicon)
    except QuotaExceeded as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(3)
    except (RuntimeError, ValueError, requests.RequestException) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json or result.get("dry_run") or not result.get("success"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Voice: {result['voice']}  calls used: {result['requests']}", file=sys.stderr)
        for p in result["scenes"]:
            mark = f"  !! {p['suspect']}" if p.get("suspect") else ""
            print(f"  {p['id']}: {p['duration_seconds']}s ({p['frames_30fps']} frames), "
                  f"{p['chars_per_sec']} chars/s{mark}", file=sys.stderr)
    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
