#!/usr/bin/env python3
"""Resolve named narration phrases to timestamps for clicks, zooms, and highlights.

Uses the word-level Caption[] files produced by tools/captions.py. No audio is
generated or transcribed. Re-run after regenerating narration and its captions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import tempfile


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)*", text.replace("\u2019", "'").casefold())


def resolve_cues(
    captions: object,
    spec: object,
    *,
    audio_offset_ms: float = 0,
    fps: float | None = None,
) -> dict:
    """Resolve whole-word phrases; reject missing or ambiguous matches.

    Occurrences are one-based. Offsets are added after choosing the start or end
    of a phrase. Output timestamps include audio_offset_ms and are independent
    of FPS; optional frames use nearest-frame rounding, with halves rounded up.
    """
    audio_offset_ms = _number(audio_offset_ms, "audio offset")
    if fps is not None:
        fps = _number(fps, "fps")
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
    if not isinstance(captions, list) or not captions:
        raise ValueError("captions must be a non-empty Caption[] array")

    words: list[tuple[str, float, float]] = []
    previous_start = -1.0
    for i, caption in enumerate(captions):
        label = f"caption {i + 1}"
        if not isinstance(caption, dict) or not isinstance(caption.get("text"), str):
            raise ValueError(f"{label} must have text, startMs, and endMs")
        start = _number(caption.get("startMs"), f"{label} startMs")
        end = _number(caption.get("endMs"), f"{label} endMs")
        if start < 0 or end < start:
            raise ValueError(f"{label} requires 0 <= startMs <= endMs")
        if start < previous_start:
            raise ValueError(f"{label} startMs is out of order")
        previous_start = start
        words.extend((word, start, end) for word in _tokens(caption["text"]))
    if not words:
        raise ValueError("captions contain no words")

    if not isinstance(spec, dict) or set(spec) - {"version", "cues"}:
        raise ValueError("spec must contain cues and optionally version")
    version = spec.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("unsupported cue spec version; expected 1")
    definitions = spec.get("cues")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("spec.cues must be a non-empty object of named phrases")

    tokens = [word for word, _, _ in words]
    resolved = {}
    for name, cue in definitions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("cue names must be non-empty strings")
        label = f"cue {name!r}"
        allowed = {"phrase", "occurrence", "anchor", "offsetMs"}
        if not isinstance(cue, dict) or set(cue) - allowed:
            raise ValueError(f"{label} accepts phrase, occurrence, anchor, and offsetMs")
        phrase = cue.get("phrase")
        if not isinstance(phrase, str) or not _tokens(phrase):
            raise ValueError(f"{label} phrase must contain words")
        needle = _tokens(phrase)
        matches = [i for i in range(len(tokens) - len(needle) + 1)
                   if tokens[i:i + len(needle)] == needle]
        if not matches:
            raise ValueError(f"{label}: phrase {phrase!r} was not found; check the script and captions")
        occurrence = cue.get("occurrence", 1)
        if "occurrence" not in cue:
            if len(matches) > 1:
                raise ValueError(f"{label}: phrase {phrase!r} matches {len(matches)} times; set occurrence")
            occurrence = 1
        if type(occurrence) is not int or occurrence < 1 or occurrence > len(matches):
            raise ValueError(f"{label}: occurrence must be an integer from 1 to {len(matches)}")
        anchor = cue.get("anchor", "start")
        if anchor not in ("start", "end"):
            raise ValueError(f"{label} anchor must be 'start' or 'end'")
        offset = _number(cue.get("offsetMs", 0), f"{label} offsetMs")
        first = matches[occurrence - 1]
        start = _number(words[first][1] + audio_offset_ms, f"{label} shifted startMs")
        end = _number(words[first + len(needle) - 1][2] + audio_offset_ms, f"{label} shifted endMs")
        at = _number((start if anchor == "start" else end) + offset, f"{label} timeMs")
        if at < 0:
            raise ValueError(f"{label} resolves before frame zero; adjust the offsets")
        item = {
            "phrase": phrase,
            "occurrence": occurrence,
            "anchor": anchor,
            "startMs": start,
            "endMs": end,
            "timeMs": at,
        }
        if fps is not None:
            item["frame"] = math.floor(_number(at / 1000 * fps, f"{label} frame") + 0.5)
        resolved[name] = item

    result = {"version": 1, "audioOffsetMs": audio_offset_ms, "cues": resolved}
    if fps is not None:
        result["fps"] = fps
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def main(argv: list[str] | None = None) -> int:
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True, help="Word-level .captions.json input")
    parser.add_argument("--spec", type=Path, required=True, help="Named phrase definitions JSON")
    parser.add_argument("--output", type=Path, help="Write cue JSON atomically; otherwise print it")
    parser.add_argument("--audio-offset-ms", type=float, default=0,
                        help="Where the audio starts in its scene, in milliseconds")
    parser.add_argument("--fps", type=float, help="Also include frame numbers at this FPS")
    args = parser.parse_args(argv)
    temporary = None
    try:
        if args.output and args.output.resolve() in (args.captions.resolve(), args.spec.resolve()):
            raise ValueError("output must be different from the captions and spec files")
        result = resolve_cues(_read_json(args.captions), _read_json(args.spec),
                              audio_offset_ms=args.audio_offset_ms, fps=args.fps)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=args.output.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(text)
            temporary.replace(args.output)
            print(f"Wrote {len(result['cues'])} cues to {args.output}", file=sys.stderr)
        else:
            print(text, end="")
        return 0
    except (OSError, ValueError) as error:
        print(f"cues: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    sys.exit(main())
