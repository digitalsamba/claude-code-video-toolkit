#!/usr/bin/env python3
"""
Align caption timestamps to actual TTS audio via ElevenLabs Scribe.

After voice generation, the hand-estimated cap(text, fromSec, durSec) timings
in sprint-config.ts drift from the real speech. This tool:

  1. Transcribes each scene MP3 with word-level timestamps (Scribe)
  2. Fuzzy-matches each caption phrase to the word stream
  3. In --apply mode, patches cap() lines in sprint-config.ts with the
     actual fromSec / durSec derived from word boundaries

Phrase detection anchors on audioFile / introAudioFile / outroAudioFile
declarations — whichever captions block follows each declaration is the
one aligned against that scene's MP3.

Usage:
    # Dry run — print proposed changes
    python3 tools/align_captions.py \\
        --config templates/ai-engineering-review/src/config/sprint-config.ts \\
        --scene-dir templates/ai-engineering-review/public/audio/scenes

    # Apply changes to config file (with 0.3s trailing hold per phrase)
    python3 tools/align_captions.py --apply --tail-pad 0.3

    # JSON output
    python3 tools/align_captions.py --json
"""

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path


# ─── Caption parsing ────────────────────────────────────────

CAP_RE = re.compile(
    r"cap\(\s*(['\"])(?P<text>.+?)\1\s*,\s*(?P<from>[\d.]+)\s*,\s*(?P<dur>[\d.]+)\s*\)"
)


def _find_captions_block(text: str, start_pos: int, key: str):
    """Find `{key}: [ ... ]` at or after start_pos using bracket counting.

    Returns (abs_start_of_bracket, abs_end_inclusive_of_bracket) or None.
    """
    m = re.search(rf"\b{re.escape(key)}\s*:\s*\[", text[start_pos:])
    if not m:
        return None
    abs_bracket_start = start_pos + m.end() - 1  # position of '['
    depth = 0
    i = abs_bracket_start
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return (abs_bracket_start, i)
        i += 1
    return None


def _parse_captions_in_span(text: str, span_start: int, span_end: int):
    """Return list of cap() dicts found within text[span_start:span_end+1]."""
    block = text[span_start : span_end + 1]
    captions = []
    for cm in CAP_RE.finditer(block):
        captions.append(
            {
                "text": cm.group("text"),
                "fromSec": float(cm.group("from")),
                "durSec": float(cm.group("dur")),
                "match_span": (span_start + cm.start(), span_start + cm.end()),
            }
        )
    return captions


def parse_scenes(config_text: str):
    """Extract scene groupings anchored on audioFile declarations.

    Returns list of dicts with:
        audio_stem      : e.g. '00-intro', 'samba-8623'
        captions        : [ {text, fromSec, durSec, match_span} ]
        captions_span   : (abs_bracket_start, abs_bracket_end)
    """
    scenes = []

    patterns = [
        (r"introAudioFile:\s*['\"]audio/scenes/([^.'\"]+)\.mp3['\"]", "introCaptions"),
        (r"outroAudioFile:\s*['\"]audio/scenes/([^.'\"]+)\.mp3['\"]", "outroCaptions"),
        (r"(?<!intro)(?<!outro)\baudioFile:\s*['\"]audio/scenes/([^.'\"]+)\.mp3['\"]", "captions"),
    ]

    for regex, key in patterns:
        for m in re.finditer(regex, config_text):
            stem = m.group(1)
            block = _find_captions_block(config_text, m.end(), key)
            if not block:
                continue
            span_start, span_end = block
            captions = _parse_captions_in_span(config_text, span_start, span_end)
            if not captions:
                continue
            scenes.append(
                {
                    "audio_stem": stem,
                    "captions": captions,
                    "captions_span": (span_start, span_end),
                }
            )

    # Order by first caption's position in file (keeps diff output readable)
    scenes.sort(key=lambda s: s["captions_span"][0])
    return scenes


# ─── Scribe transcription ───────────────────────────────────

def transcribe_words(audio_path: str, api_key: str, model_id: str = "scribe_v1"):
    """Return list of word dicts: {text, start, end}."""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id=model_id,
            tag_audio_events=False,
        )
    return [
        {"text": w.text, "start": w.start, "end": w.end}
        for w in result.words
        if w.type == "word"
    ]


# ─── Fuzzy matching ─────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    # Convert em-dash / en-dash / ellipsis to whitespace
    s = re.sub(r"[—–…]", " ", s)
    # Keep alphanumerics, apostrophes (for contractions), whitespace
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str):
    return _normalize(s).split()


def _match_phrase(phrase_tokens, words, start_idx):
    """Find best contiguous match for phrase_tokens in words starting at or after start_idx.

    Returns ((first_word_idx, last_word_idx_inclusive), score) or None.
    """
    if not phrase_tokens or start_idx >= len(words):
        return None

    n = len(phrase_tokens)
    best_score = 0.0
    best_range = None

    # Limit search horizon to keep it fast
    horizon = min(len(words), start_idx + n * 4 + 10)

    for i in range(start_idx, horizon):
        for length in range(max(1, n - 2), n + 4):
            j = i + length
            if j > len(words):
                break
            candidate_text = " ".join(w["text"] for w in words[i:j])
            cand_tokens = _tokens(candidate_text)
            if not cand_tokens:
                continue
            ratio = difflib.SequenceMatcher(
                None, phrase_tokens, cand_tokens, autojunk=False
            ).ratio()
            if ratio > best_score:
                best_score = ratio
                best_range = (i, j - 1)
                if ratio == 1.0:
                    return (best_range, 1.0)

    if best_score < 0.45:
        return None
    return (best_range, best_score)


def align_scene(scene, words, tail_pad: float = 0.0):
    aligned = []
    word_ptr = 0
    for cap in scene["captions"]:
        phrase_tokens = _tokens(cap["text"])
        hit = _match_phrase(phrase_tokens, words, word_ptr)
        if not hit:
            aligned.append({**cap, "aligned": False})
            continue
        (ws, we), score = hit
        start_sec = round(words[ws]["start"], 2)
        end_sec = round(words[we]["end"] + tail_pad, 2)
        aligned.append(
            {
                **cap,
                "aligned": True,
                "score": round(score, 3),
                "fromSec_new": start_sec,
                "durSec_new": round(max(0.01, end_sec - start_sec), 2),
                "word_start": ws,
                "word_end": we,
            }
        )
        word_ptr = we + 1
    return aligned


# ─── Patching ───────────────────────────────────────────────

def _format_cap(text: str, from_sec: float, dur_sec: float) -> str:
    """Re-emit a cap() call with minimal quote-escaping hassle."""
    # If the text contains a single quote, use double-quotes; otherwise single.
    if "'" in text and '"' not in text:
        quoted = f'"{text}"'
    else:
        # escape single-quotes if we must
        safe = text.replace("\\", "\\\\").replace("'", "\\'")
        quoted = f"'{safe}'"
    # Use 1 decimal place if the value is already a round tenth; else 2
    def fmt(v):
        if abs(v * 10 - round(v * 10)) < 1e-6:
            return f"{v:.1f}"
        return f"{v:.2f}"
    return f"cap({quoted}, {fmt(from_sec)}, {fmt(dur_sec)})"


def apply_patches(config_text: str, scenes_aligned):
    """Replace cap() spans with updated timings. Patch from the end forward
    so earlier match spans aren't shifted by later edits."""
    patches = []
    for scene in scenes_aligned:
        for cap in scene["captions_aligned"]:
            if not cap.get("aligned"):
                continue
            patches.append(
                (
                    cap["match_span"],
                    _format_cap(cap["text"], cap["fromSec_new"], cap["durSec_new"]),
                )
            )
    patches.sort(key=lambda p: p[0][0], reverse=True)
    out = config_text
    for (start, end), new_text in patches:
        out = out[:start] + new_text + out[end:]
    return out


# ─── Main ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Align caption timestamps to TTS audio via ElevenLabs Scribe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", required=True, help="Path to sprint-config.ts")
    ap.add_argument("--scene-dir", required=True, help="Directory containing scene MP3s")
    ap.add_argument("--apply", action="store_true", help="Write changes back to config")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    ap.add_argument(
        "--tail-pad",
        type=float,
        default=0.0,
        help="Seconds to extend each caption past its last spoken word (e.g. 0.3 for a gentle hold)",
    )
    ap.add_argument(
        "--model",
        default="scribe_v1",
        choices=["scribe_v1", "scribe_v1_experimental"],
        help="Scribe STT model",
    )
    args = ap.parse_args()

    # Load env via the toolkit's own loader if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    scene_dir = Path(args.scene_dir)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not scene_dir.is_dir():
        print(f"Error: scene-dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    config_text = config_path.read_text()
    scenes = parse_scenes(config_text)
    if not scenes:
        print("No captioned scenes found in config", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenes)} captioned scene(s)", file=sys.stderr)

    total_matched = 0
    total_caps = 0
    for scene in scenes:
        stem = scene["audio_stem"]
        audio_path = None
        for ext in (".mp3", ".wav", ".m4a"):
            candidate = scene_dir / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if not audio_path:
            print(f"  [skip] {stem}: no audio found in {scene_dir}", file=sys.stderr)
            scene["captions_aligned"] = [{**c, "aligned": False} for c in scene["captions"]]
            continue

        print(f"  Aligning {stem} ({len(scene['captions'])} caps)...", file=sys.stderr)
        try:
            words = transcribe_words(str(audio_path), api_key, model_id=args.model)
        except Exception as e:
            print(f"    [error] transcription failed: {e}", file=sys.stderr)
            scene["captions_aligned"] = [{**c, "aligned": False} for c in scene["captions"]]
            continue

        if not words:
            print(f"    [error] no words in transcript", file=sys.stderr)
            scene["captions_aligned"] = [{**c, "aligned": False} for c in scene["captions"]]
            continue

        scene["captions_aligned"] = align_scene(scene, words, tail_pad=args.tail_pad)
        scene["audio_seconds"] = round(words[-1]["end"], 2)

        for cap in scene["captions_aligned"]:
            total_caps += 1
            if cap.get("aligned"):
                total_matched += 1
                drift_from = cap["fromSec_new"] - cap["fromSec"]
                drift_dur = cap["durSec_new"] - cap["durSec"]
                label = cap["text"][:48]
                if len(cap["text"]) > 48:
                    label = label[:45] + "..."
                print(
                    f"    {label:<48}  "
                    f"from {cap['fromSec']:>5.2f}→{cap['fromSec_new']:>5.2f} ({drift_from:+.2f}s)  "
                    f"dur {cap['durSec']:>4.2f}→{cap['durSec_new']:>4.2f} ({drift_dur:+.2f}s)  "
                    f"[score {cap['score']:.2f}]",
                    file=sys.stderr,
                )
            else:
                label = cap["text"][:60]
                print(f"    [UNMATCHED] {label}", file=sys.stderr)

    print(
        f"\nAligned {total_matched}/{total_caps} captions across {len(scenes)} scenes",
        file=sys.stderr,
    )

    if args.apply:
        patched = apply_patches(config_text, scenes)
        if patched == config_text:
            print("No changes to apply.", file=sys.stderr)
        else:
            config_path.write_text(patched)
            print(f"Wrote {config_path}", file=sys.stderr)
    else:
        print("Dry run — use --apply to write changes.", file=sys.stderr)

    if args.json:
        summary = {
            "scenes": [
                {
                    "audio_stem": s["audio_stem"],
                    "audio_seconds": s.get("audio_seconds"),
                    "captions": [
                        {
                            "text": c["text"],
                            "fromSec": c["fromSec"],
                            "durSec": c["durSec"],
                            "fromSec_new": c.get("fromSec_new"),
                            "durSec_new": c.get("durSec_new"),
                            "aligned": c.get("aligned", False),
                            "score": c.get("score"),
                        }
                        for c in s.get("captions_aligned", [])
                    ],
                }
                for s in scenes
            ],
            "total_matched": total_matched,
            "total_captions": total_caps,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
