#!/usr/bin/env python3
"""Diarized multi-speaker dubbing using WhisperX, existing TTS tools and FFmpeg."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from media_ops import (
    audio_stream,
    load_json,
    number,
    output_path,
    probe,
    run,
    sha256,
    staged_output,
    video_stream,
    write_json,
)
from transcript import segments

ROOT = Path(__file__).resolve().parents[1]


def import_transcript(value):
    if not isinstance(value, dict) or not isinstance(value.get('segments'), list):
        raise ValueError('Import expects diarized WhisperX JSON')
    if any(not isinstance(s, dict) or not s.get('speaker') for s in value['segments']):
        raise ValueError('Every segment needs a speaker ID. Run diarization before importing.')
    utterances = segments(value)
    return {'version': 1, 'speakers': {s['speaker']: {'provider': 'elevenlabs', 'voiceId': 'SET_VOICE_ID'} for s in utterances},
            'segments': [{'id': f'line-{i + 1:04}', 'speaker': s['speaker'], 'startSeconds': s['start'],
                          'endSeconds': s['end'], 'text': s['text']} for i, s in enumerate(utterances)]}


def diarize(video, output, *, model='base', device='cpu'):
    output_path(output, video)
    audio_stream(probe(video))
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('Set HF_TOKEN and accept the selected pyannote model conditions before diarization')
    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    except ImportError as error:
        raise ValueError('Install WhisperX in a separate environment; see docs/advanced-production.md') from error
    audio = whisperx.load_audio(str(video))
    engine = whisperx.load_model(model, device, compute_type='int8' if device == 'cpu' else 'float16')
    result = engine.transcribe(audio, batch_size=4)
    del engine
    gc.collect()
    aligner, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
    aligned = whisperx.align(result['segments'], aligner, metadata, audio, device, return_char_alignments=False)
    del aligner
    gc.collect()
    pipeline = DiarizationPipeline(token=token, device=device)
    labeled = whisperx.assign_word_speakers(pipeline(audio), aligned)
    # Also validate that all segments actually received labels before writing.
    import_transcript(labeled)
    write_json(output, labeled)


def resolve(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Audio and reference paths must be nonempty strings')
    path = (root / value).resolve()
    if not path.is_file():
        raise ValueError(f'Input file does not exist: {path}')
    return path


def prepare(spec, root, duration):
    if not isinstance(spec, dict) or type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Dubbing spec version must be 1')
    speakers = spec.get('speakers')
    lines = spec.get('segments')
    if not isinstance(speakers, dict) or not speakers or not isinstance(lines, list) or not 1 <= len(lines) <= 128:
        raise ValueError('Dubbing needs speakers and 1 to 128 segments per job')
    identities, normalized, previous = set(), [], -1
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get('id'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', line['id']):
            raise ValueError('Each dubbing segment needs a safe id')
        if line['id'] in identities:
            raise ValueError('Duplicate dubbing segment id')
        identities.add(line['id'])
        speaker = line.get('speaker')
        if not isinstance(speaker, str) or speaker not in speakers or not isinstance(speakers[speaker], dict):
            raise ValueError(f'Unknown speaker: {speaker}')
        start = number(line.get('startSeconds'), 'segment start', 0, duration)
        end = number(line.get('endSeconds'), 'segment end', start + .001, duration + .001)
        if start < previous:
            raise ValueError('Dubbing segments must be ordered by start')
        previous = start
        text = line.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Every dubbing segment needs replacement text')
        voice = speakers[speaker]
        item = {**line, 'startSeconds': start, 'endSeconds': end, 'gain': number(line.get('gain', 1), 'voice gain', 0, 4)}
        if line.get('audio'):
            item['audio'] = str(resolve(root, line['audio']))
            audio_stream(probe(item['audio']))
        else:
            if voice.get('provider') not in ('elevenlabs', 'qwen3', '60db'):
                raise ValueError('Speaker provider must be elevenlabs, qwen3 or 60db')
            identity = voice.get('speaker') if voice['provider'] == 'qwen3' else voice.get('voiceId')
            if not isinstance(identity, str) or not identity.strip() or identity.startswith('SET_'):
                raise ValueError(f'Configure a voice identity for {speaker}')
            if voice.get('refAudio'):
                resolve(root, voice['refAudio'])
                if not isinstance(voice.get('refText'), str) or not voice['refText'].strip():
                    raise ValueError('Cloned references need refText')
        normalized.append(item)
    return normalized


def synthesize(line, voice, root, directory):
    script = directory / (line['id'] + '.txt')
    target = directory / (line['id'] + '.mp3')
    script.write_text(line['text'], encoding='utf-8')
    command = [sys.executable, ROOT / 'tools/voiceover.py', '--provider', voice['provider'],
               '--script', script, '--output', target, '--json']
    if voice['provider'] == 'qwen3':
        command += ['--speaker', voice['speaker']]
        if voice.get('refAudio'):
            command += ['--ref-audio', resolve(root, voice['refAudio']), '--ref-text', voice['refText']]
    else:
        command += ['--voice-id', voice['voiceId']]
    run(command)
    audio_stream(probe(target))
    return target


def dub(video, spec_file, output, *, execute=False, min_tempo=.75, max_tempo=1.5):
    spec = load_json(spec_file)
    root = Path(spec_file).resolve().parent
    info = probe(video)
    video_stream(info)
    lines = prepare(spec, root, info['duration'])
    output_path(output, video, spec_file, *[line.get('audio') for line in lines])
    number(min_tempo, 'minimum tempo', .5, 1)
    number(max_tempo, 'maximum tempo', 1, 2)
    background = resolve(root, spec['background']) if spec.get('background') else None
    if background:
        output_path(output, background)
        audio_stream(probe(background))
    result = {'version': 1, 'executed': execute, 'output': str(Path(output).resolve()),
              'sourceSha256': sha256(video), 'segments': lines, 'backgroundPreserved': bool(background),
              'synthesizedSegments': sum(not line.get('audio') for line in lines), 'timing': []}
    if not execute:
        return result
    with tempfile.TemporaryDirectory(prefix='video-dub-') as folder:
        directory = Path(folder)
        command = ['ffmpeg', '-y', '-v', 'error', '-i', str(Path(video).resolve()),
                   '-f', 'lavfi', '-i', f'anullsrc=r=48000:cl=stereo:d={info["duration"]}']
        filters, labels = [], ['[1:a:0]']
        index = 2
        if background:
            command += ['-i', str(background)]
            filters.append(f'[{index}:a:0]aresample=48000,apad,atrim=duration={info["duration"]}[background]')
            labels.append('[background]')
            index += 1
        for line in lines:
            audio = Path(line['audio']) if line.get('audio') else synthesize(line, spec['speakers'][line['speaker']], root, directory)
            length = probe(audio)['duration']
            slot = line['endSeconds'] - line['startSeconds']
            tempo = length / slot
            if not min_tempo <= tempo <= max_tempo:
                raise ValueError(f'{line["id"]} requires tempo {tempo:.3f}, outside {min_tempo}–{max_tempo}. Revise the text or timing.')
            command += ['-i', str(audio)]
            label = f'line{index}'
            filters.append(f'[{index}:a:0]aresample=48000,atempo={tempo},apad,atrim=duration={slot},'
                           f'volume={line["gain"]},adelay={round(line["startSeconds"] * 1000)}:all=1[{label}]')
            labels.append(f'[{label}]')
            result['timing'].append({'id': line['id'], 'speaker': line['speaker'], 'tempo': tempo,
                                     'startSeconds': line['startSeconds'], 'durationSeconds': slot})
            index += 1
        filters.append(''.join(labels) + f'amix=inputs={len(labels)}:normalize=0:duration=first,alimiter=limit=.95:latency=1[out]')
        with staged_output(output) as target:
            run([*command, '-filter_complex', ';'.join(filters), '-map', '0:v:0', '-map', '[out]',
                 '-c:v', 'copy', '-c:a', 'aac', '-t', str(info['duration']), target])
    return result


def separate(video, output, *, execute=False, python=sys.executable):
    output_path(output, video)
    audio_stream(probe(video))
    if execute:
        with tempfile.TemporaryDirectory(prefix='video-stems-') as folder:
            source = Path(folder) / 'source.wav'
            run(['ffmpeg', '-v', 'error', '-i', video, '-vn', '-c:a', 'pcm_s16le', source])
            run([python, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs', '-o', folder, source], timeout=7200)
            stem = Path(folder) / 'htdemucs/source/no_vocals.wav'
            audio_stream(probe(stem))
            with staged_output(output) as target:
                shutil.copyfile(stem, target)
    return {'version': 1, 'executed': execute, 'output': str(Path(output).resolve()), 'engine': 'demucs/htdemucs'}


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ingest = commands.add_parser('import', help='Write an editable dubbing spec from diarized WhisperX JSON')
    ingest.add_argument('transcript', type=Path)
    ingest.add_argument('--output', type=Path, required=True)
    analyze = commands.add_parser('diarize', help='Run installed WhisperX and pyannote locally; downloads model weights')
    analyze.add_argument('--video', type=Path, required=True)
    analyze.add_argument('--output', type=Path, required=True)
    analyze.add_argument('--model', default='base')
    analyze.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    render = commands.add_parser('render')
    render.add_argument('--video', type=Path, required=True)
    render.add_argument('--spec', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    render.add_argument('--min-tempo', type=float, default=.75)
    render.add_argument('--max-tempo', type=float, default=1.5)
    render.add_argument('--run', action='store_true')
    stems = commands.add_parser('separate', help='Extract a background stem with an installed Demucs environment')
    stems.add_argument('--video', type=Path, required=True)
    stems.add_argument('--output', type=Path, required=True)
    stems.add_argument('--python', default=sys.executable, help='Python executable with Demucs installed')
    stems.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'import':
            output_path(args.output, args.transcript)
            result = import_transcript(load_json(args.transcript))
            write_json(args.output, result)
        elif args.command == 'diarize':
            diarize(args.video, args.output, model=args.model, device=args.device)
            result = {'version': 1, 'output': str(args.output.resolve()), 'executed': True}
        elif args.command == 'separate':
            result = separate(args.video, args.output, execute=args.run, python=args.python)
        else:
            result = dub(args.video, args.spec, args.output, execute=args.run, min_tempo=args.min_tempo, max_tempo=args.max_tempo)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
