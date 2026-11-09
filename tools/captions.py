#!/usr/bin/env python3
"""Word-level caption files for burned captions in the Remotion templates.

Transcribes narration audio to word timestamps and writes a sibling
`<name>.captions.json` next to each audio file:

    public/audio/scenes/01-title.mp3  ->  public/audio/scenes/01-title.captions.json

The JSON is an array in the @remotion/captions `Caption` shape
({text, startMs, endMs, timestampMs, confidence}, one entry per word), which
the shared <Captions> component in lib/components renders. The templates pick
the file up automatically once it exists and captions are enabled in config.

Script-aligned by default: when a sibling `<name>.txt` exists (voiceover.py
--scene-dir leaves one beside every MP3), the script's words are mapped onto
the transcript's timing, so captions spell product names and punctuation the
way the script does, never the way the recogniser misheard them. Pass
--no-script to burn the raw transcript, or --script FILE for a single input.

Engines:
  scribe   ElevenLabs Scribe (hosted, needs ELEVENLABS_API_KEY)
  whisper  openai-whisper (local, free; `uv sync --extra whisper`, pulls in torch)
  auto     scribe when ELEVENLABS_API_KEY is set, else whisper (default)

Usage:
    # Every scene MP3 in a project (run from the toolkit root)
    uv run tools/captions.py projects/my-video/public/audio/scenes

    # A single voiceover file, aligned to its script
    uv run tools/captions.py projects/my-video/public/audio/voiceover.mp3 \\
        --script projects/my-video/VOICEOVER-SCRIPT.txt

    # Local and free
    uv run tools/captions.py projects/my-video/public/audio/scenes --engine whisper --json
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

from pacing import spoken_text
from win_encoding import ensure_utf8_output

AUDIO_EXTS = (".mp3", ".wav", ".m4a")
SUFFIX = ".captions.json"

# Below this share of script words found in the transcript, the .txt is
# probably stale (script edited after the audio was generated).
MIN_MATCH_RATIO = 0.6


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


# ─── Transcription ──────────────────────────────────────────

def transcribe_scribe(audio: Path, api_key: str, model_id: str, language: str | None) -> list[dict]:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    kwargs = {"language_code": language} if language else {}
    with open(audio, "rb") as f:
        result = client.speech_to_text.convert(
            file=f, model_id=model_id, tag_audio_events=False, **kwargs,
        )
    return [{"w": w.text.strip(), "s": w.start, "e": w.end}
            for w in result.words if w.type == "word" and w.text.strip()]


def load_whisper(model_name: str):
    try:
        import whisper
    except ImportError:
        sys.exit("openai-whisper is not installed:\n"
                 "    uv sync --extra whisper\n"
                 "or use --engine scribe (needs ELEVENLABS_API_KEY)")
    if model_name == "auto":
        import torch  # whisper already depends on it
        model_name = "turbo" if torch.cuda.is_available() else "base"
    print(f"whisper model: {model_name}", file=sys.stderr)
    return whisper.load_model(model_name)


def transcribe_whisper(model, audio: Path, language: str | None) -> list[dict]:
    r = model.transcribe(str(audio), word_timestamps=True, language=language)
    return [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]}
            for seg in r["segments"] for w in seg.get("words", []) if w["word"].strip()]


# ─── Script alignment ───────────────────────────────────────

def align(script_words: list[str], heard: list[dict]) -> tuple[list[dict], float]:
    """Map script words onto heard-word timings; interpolate unmatched runs.

    Returns the timed script words and the share of them matched directly.
    """
    sm = difflib.SequenceMatcher(
        a=[norm(w) for w in script_words],
        b=[norm(h["w"]) for h in heard],
        autojunk=False,
    )
    starts: list[float | None] = [None] * len(script_words)
    ends: list[float | None] = [None] * len(script_words)
    matched = 0
    for a, b, n in sm.get_matching_blocks():
        matched += n
        for k in range(n):
            starts[a + k] = heard[b + k]["s"]
            ends[a + k] = heard[b + k]["e"]
    total_end = heard[-1]["e"] if heard else 0.0
    i = 0
    while i < len(script_words):
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < len(script_words) and starts[j] is None:
            j += 1
        lo = ends[i - 1] if i > 0 else 0.0
        hi = starts[j] if j < len(script_words) else total_end
        span = max(hi - lo, 0.12 * (j - i))
        for k in range(i, j):
            starts[k] = lo + span * (k - i) / (j - i)
            ends[k] = lo + span * (k - i + 1) / (j - i)
        i = j
    words = [{"w": w, "s": starts[k], "e": ends[k]} for k, w in enumerate(script_words)]
    return words, matched / max(len(script_words), 1)


def to_remotion(words: list[dict]) -> list[dict]:
    """@remotion/captions Caption[]: one token per word, leading space after the first."""
    out = []
    for k, w in enumerate(words):
        start = round(w["s"] * 1000)
        end = max(round(w["e"] * 1000), start + 1)
        out.append({
            "text": w["w"] if k == 0 else f" {w['w']}",
            "startMs": start,
            "endMs": end,
            "timestampMs": (start + end) // 2,
            "confidence": None,
        })
    return out


def export_subtitles(words: object, *, format: str = 'srt', max_words: int = 7) -> str:
    """Group Caption[] at sentence ends, pauses over 600ms, or the word limit."""
    if format not in ('srt', 'vtt') or type(max_words) is not int or not 1 <= max_words <= 30 or not isinstance(words, list):
        raise ValueError('Use srt/vtt, a word limit of 1-30 and a caption array')
    previous = 0
    validated = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get('text'), str) or not word['text'].strip():
            raise ValueError('Caption words require nonempty text')
        start, end = word.get('startMs'), word.get('endMs')
        try:
            valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (start, end))
        except OverflowError:
            valid = False
        if not valid or start < 0 or end < start:
            raise ValueError('Caption times must be finite, nonnegative and end after they start')
        # Our own pipeline can overlap neighbours (to_remotion gives a zero-length word 1 ms; align's
        # minimum span can run an interpolated word past the next one), so clamp onto the previous word.
        start = max(start, previous)
        end = max(end, start)
        previous = end
        validated.append({**word, 'text': ' '.join(word['text'].split()), 'startMs': start, 'endMs': end})

    def timestamp(value):
        seconds, millis = divmod(round(value), 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}{"," if format == "srt" else "."}{millis:03}'

    groups, pending = [], []
    for index, word in enumerate(validated):
        pending.append(word)
        sentence_end = re.search(r'[.?!。！？]["\'”’\)\]]*$', word['text'])
        pause = index + 1 < len(validated) and validated[index + 1]['startMs'] - word['endMs'] > 600
        if sentence_end or pause or len(pending) >= max_words:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    blocks = []
    for group in groups:
        content = ' '.join(word['text'] for word in group).replace('-->', '→')
        if format == 'vtt':
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        start = group[0]['startMs']
        end = max(start + 1, group[-1]['endMs'])
        blocks.append(f'{len(blocks) + 1}\n{timestamp(start)} --> {timestamp(end)}\n{content}')
    return ('WEBVTT\n\n' if format == 'vtt' else '') + '\n\n'.join(blocks) + ('\n' if blocks else '')


def write_text_atomic(path: Path, content: str) -> None:
    """Leave the previous output intact if the replacement cannot be written."""
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(content, encoding='utf-8', newline='\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_caption_outputs(out: Path, captions: list, *, srt=False, vtt=False, max_words=7, write_json=True) -> dict:
    # The caption JSON is what the paid transcription bought, so it is written first and kept even if a
    # subtitle can't be built. The subtitles are serialized before either is replaced.
    if write_json:
        write_text_atomic(out, json.dumps(captions, ensure_ascii=False, allow_nan=False, indent=1) + '\n')
    outputs = {}
    paths = {'captions': str(out)}
    for format, enabled in [('srt', srt), ('vtt', vtt)]:
        if enabled:
            target = out.with_name(out.name[:-len(SUFFIX)] + '.' + format)
            outputs[target] = export_subtitles(captions, format=format, max_words=max_words)
            paths[format] = str(target)
    for path, content in outputs.items():
        write_text_atomic(path, content)
    return paths


# ─── CLI ────────────────────────────────────────────────────

def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Error: not found: {p}")
    if not files:
        sys.exit("Error: no audio files (.mp3/.wav/.m4a) found")
    return files


def main(argv=None) -> None:
    ensure_utf8_output()  # Windows: piped stdout defaults to cp1252
    from voiceover import ELEVENLABS_STT_MODELS, UrlSafeHelpFormatter, resolve_model_id

    ap = argparse.ArgumentParser(
        description="Write word-level <name>.captions.json files for burned Remotion captions.",
        formatter_class=UrlSafeHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("inputs", nargs="+", help="Audio files/directories, or existing *.captions.json for offline subtitle export")
    ap.add_argument("--engine", choices=["auto", "scribe", "whisper"], default="auto",
                    help="Transcription engine (default: auto = scribe if ELEVENLABS_API_KEY is set, else whisper)")
    ap.add_argument("--script", help="Script text to align to (single input only; default: sibling <name>.txt)")
    ap.add_argument("--no-script", action="store_true", help="Use the raw transcript, ignore any .txt")
    ap.add_argument("--language", help="Language code, e.g. en (default: auto-detect)")
    ap.add_argument("--whisper-model", default="auto",
                    help="whisper model name (default: auto = turbo on CUDA, base on CPU)")
    ap.add_argument("--scribe-model", default="scribe_v2", choices=ELEVENLABS_STT_MODELS,
                    help="Scribe STT model (default: scribe_v2)")
    ap.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary on stdout")
    ap.add_argument('--srt', action='store_true', help='Also write a sibling <name>.srt subtitle file')
    ap.add_argument('--vtt', action='store_true', help='Also write a sibling <name>.vtt subtitle file')
    ap.add_argument('--max-words', type=int, default=7, help='Maximum words per subtitle cue (1-30, default 7); sentences and pauses also split cues')
    args = ap.parse_args(argv)
    if not 1 <= args.max_words <= 30:
        ap.error('--max-words must be between 1 and 30')

    files = collect_inputs(args.inputs)
    audio_files = [p for p in files if not p.name.endswith(SUFFIX)]
    if len(audio_files) != len(files) and not (args.srt or args.vtt):
        ap.error('Existing caption JSON needs --srt or --vtt')
    if args.script and (len(files) != 1 or not audio_files):
        sys.exit("Error: --script needs exactly one audio input")
    if args.script and args.no_script:
        sys.exit("Error: --script and --no-script are mutually exclusive")
    if args.script and not Path(args.script).is_file():
        sys.exit(f"Error: script not found: {args.script}")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("ELEVENLABS_API_KEY")
    engine = args.engine
    if not audio_files:
        engine = None
    if engine == "auto":
        engine = "scribe" if api_key else "whisper"
    if engine == "scribe" and not api_key:
        sys.exit("Error: ELEVENLABS_API_KEY not set (or use --engine whisper)")
    whisper_model = load_whisper(args.whisper_model) if engine == "whisper" else None
    scribe_model = resolve_model_id(args.scribe_model)
    if engine:
        print(f"engine: {engine}", file=sys.stderr)

    results = []
    for audio in files:
        if audio.name.endswith(SUFFIX):
            try:
                captions = json.loads(audio.read_text(encoding='utf-8-sig'))
                paths = write_caption_outputs(audio, captions, srt=args.srt, vtt=args.vtt,
                                              max_words=args.max_words, write_json=False)
                results.append({**paths, 'words': len(captions)})
            except (ValueError, OSError) as error:
                print(f'  [error] {audio.name}: {error}', file=sys.stderr)
                results.append({'captions': str(audio), 'error': str(error)})
            continue
        out = audio.with_name(audio.stem + SUFFIX)
        script_path = None
        if not args.no_script:
            script_path = Path(args.script) if args.script else audio.with_suffix(".txt")
            if not script_path.is_file():
                script_path = None

        try:
            heard = (transcribe_scribe(audio, api_key, scribe_model, args.language)
                     if engine == "scribe"
                     else transcribe_whisper(whisper_model, audio, args.language))
        except Exception as e:  # one bad file shouldn't sink the batch
            print(f"  [error] {audio.name}: {e}", file=sys.stderr)
            results.append({"audio": str(audio), "error": str(e)})
            continue
        if not heard:
            print(f"  [skip] {audio.name}: no speech found", file=sys.stderr)
            results.append({"audio": str(audio), "error": "no speech found"})
            continue

        match_ratio = None
        words = heard
        if script_path:
            script_words = spoken_text(script_path.read_text(encoding="utf-8")).split()
            if script_words:
                words, match_ratio = align(script_words, heard)

        try:
            paths = write_caption_outputs(out, to_remotion(words), srt=args.srt, vtt=args.vtt, max_words=args.max_words)
        except (ValueError, OSError) as error:
            print(f'  [error] {audio.name}: {error}', file=sys.stderr)
            results.append({'audio': str(audio), 'error': str(error)})
            continue
        source = f"aligned to {script_path.name}" if match_ratio is not None else "raw transcript"
        line = f"  {out.name}: {len(words)} words, {source}"
        if match_ratio is not None:
            line += f" ({match_ratio:.0%} matched)"
            if match_ratio < MIN_MATCH_RATIO:
                line += " — WARNING: script and audio disagree; is the .txt stale?"
        print(line, file=sys.stderr)
        results.append({
            "audio": str(audio),
            **paths,
            "words": len(words),
            "script": str(script_path) if match_ratio is not None else None,
            "matchRatio": round(match_ratio, 3) if match_ratio is not None else None,
            "durationSeconds": round(words[-1]["e"], 2),
        })

    if args.json:
        print(json.dumps({"engine": engine, "files": results}, indent=2))
    if any("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
