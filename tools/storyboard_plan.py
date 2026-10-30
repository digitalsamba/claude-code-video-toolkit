#!/usr/bin/env python3
"""Production adapter for manifest-driven storyboard projects."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from production import BuildError, atomic_write, project_path
from project_manifest import ProjectValidationError, load_project, resolve_project_path, validate_project


def scene_paths(scene: dict) -> tuple[str, str]:
    key = re.sub(r'[^A-Za-z0-9_-]', '-', scene['id'])[:40] + '-' + hashlib.sha256(scene['id'].encode()).hexdigest()[:6]
    return f'public/audio/scenes/{key}.txt', f'audio/scenes/{key}.mp3'


def storyboard_composition(format: object) -> str:
    if isinstance(format, dict):
        ratio = format.get('aspectRatio')
        if ratio is None and format.get('width') and format.get('height'):
            ratio = '1:1' if format['width'] == format['height'] else '9:16' if format['height'] > format['width'] else '16:9'
        format = ratio or 'landscape'
    aliases = {'portrait': 'StoryboardPortrait', '9:16': 'StoryboardPortrait', 'square': 'StoryboardSquare', '1:1': 'StoryboardSquare',
               'landscape': 'Storyboard', '16:9': 'Storyboard', None: 'Storyboard'}
    if not isinstance(format, (str, type(None))) or format not in aliases:
        raise BuildError('Storyboard format must be landscape/16:9, portrait/9:16 or square/1:1')
    return aliases[format]


def prepare_props(project: Path, *, captions: bool = False) -> dict:
    data = copy.deepcopy(load_project(project))
    for scene in data['scenes']:
        audio = scene.get('audioFile') or scene.get('audio', {}).get('file')
        if scene.get('narration') and not audio:
            _, audio = scene_paths(scene)
            scene['audioFile'] = audio
            if captions:
                scene['captionsFile'] = audio[:-4] + '.captions.json'
        if audio:
            path = resolve_project_path(project / 'public', audio)
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], capture_output=True, text=True, check=True)
            scene['durationSeconds'] = float(result.stdout) + .6
        else:
            scene.setdefault('durationSeconds', 4)
        visual = scene.get('visual', {})
        assets = [visual.get('asset'), scene.get('audioFile'), scene.get('captionsFile'), scene.get('narratorFile')]
        assets += [scene.get('audio', {}).get('file'), scene.get('audio', {}).get('captions')]
        for asset in filter(None, assets):
            if not resolve_project_path(project / 'public', asset).is_file():
                raise BuildError(f"{scene['id']}: missing storyboard asset {asset}")
    return {'project': validate_project(data), 'safeZones': False}


def storyboard_plan(project: Path, *, provider: str, cloud: str, captions: bool, caption_engine: str) -> dict:
    data = load_project(project)
    tasks, dependencies = [], []
    for scene in data['scenes']:
        if not scene.get('narration') or scene.get('audioFile') or scene.get('audio', {}).get('file'):
            continue
        script, audio = scene_paths(scene)
        key = Path(script).stem
        script_id, audio_id = 'script-' + key, 'voice-' + key
        writer = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(sys.argv[2], encoding='utf-8')"
        tasks.append({'id': script_id, 'command': ['{python}', '-c', writer, '{project}/' + script, scene['narration']],
            'inputs': [], 'outputs': [script], 'estimatedCostUsd': 0})
        voice = data.get('voice', {})
        command = ['{python}', '{toolkit}/tools/voiceover.py', '--script', '{project}/' + script,
            '--output', '{project}/public/' + audio, '--provider', voice.get('provider', provider),
            '--cloud', voice.get('cloud', cloud), '--json']
        if voice.get('voiceId'):
            command += ['--voice-id', voice['voiceId']]
        if voice.get('language'):
            command += ['--language', voice['language']]
        if data.get('brand'):
            command += ['--brand', data['brand']]
        tasks.append({'id': audio_id, 'dependsOn': [script_id], 'command': command, 'inputs': [script],
            'toolkitInputs': ['tools/voiceover.py', 'tools/pacing.py', 'tools/config.py', 'tools/cloud_gpu.py', '.env'] +
                ([f"brands/{data['brand']}/voice.json"] if data.get('brand') else []),
            'outputs': ['public/' + audio]})
        dependencies.append(audio_id)
        if captions:
            caption_id = 'captions-' + key
            tasks.append({'id': caption_id, 'dependsOn': [audio_id],
                'command': ['{python}', '{toolkit}/tools/captions.py', '{project}/public/' + audio,
                            '--script', '{project}/' + script, '--engine', caption_engine],
                'inputs': ['public/' + audio, script], 'outputs': ['public/' + audio[:-4] + '.captions.json']})
            dependencies.append(caption_id)
    props = '.video-toolkit/storyboard-props.json'
    prepare_command = ['{python}', '{toolkit}/tools/storyboard_plan.py', '--project', '{project}', '--prepare']
    if captions:
        prepare_command.append('--captions')
    tasks.append({'id': 'prepare-storyboard', 'dependsOn': dependencies, 'command': prepare_command,
        'inputs': ['project.json', 'public'], 'outputs': [props], 'estimatedCostUsd': 0})
    composition = storyboard_composition(data.get('format'))
    output = data.get('render', {}).get('file') or 'out/video.mp4'
    tasks.append({'id': 'render', 'dependsOn': ['prepare-storyboard'], 'command': ['node',
        '{project}/node_modules/@remotion/cli/remotion-cli.js', 'render', composition, '{project}/' + output,
        '--props', '{project}/' + props], 'inputs': ['src', 'public', props, 'package.json'],
        'toolkitInputs': ['lib'], 'outputs': [output], 'timeoutSeconds': 7200, 'estimatedCostUsd': 0})
    qc = ['{python}', '{toolkit}/tools/qc.py', '--render', '{project}/' + output, '--json']
    if not any(scene.get('narration') or scene.get('audioFile') or scene.get('audio', {}).get('file') for scene in data['scenes']):
        qc.append('--allow-silent')
    tasks.append({'id': 'qc-after', 'dependsOn': ['render'], 'command': qc, 'inputs': [output],
        'outputs': [], 'stdoutFile': '.video-toolkit/reports/qc-after.json', 'estimatedCostUsd': 0})
    return {'version': 1, 'tasks': tasks}


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--captions', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not args.prepare:
            parser.error('--prepare is required; use tools/build.py to inspect the plan')
        props = prepare_props(args.project, captions=args.captions)
        atomic_write(project_path(args.project, '.video-toolkit/storyboard-props.json'), json.dumps(props, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'prepared': True, 'scenes': len(props['project']['scenes'])}))
        return 0
    except (OSError, ValueError, ProjectValidationError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
