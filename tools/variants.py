#!/usr/bin/env python3
"""Validate and materialize language/campaign versions without changing a source project."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

from production import BuildError, atomic_write, project_path
from project_manifest import ProjectValidationError, load_project, validate_project

TRANSLATABLE = ('title', 'onScreenText', 'narration')


def derive_variants(project: dict, specification: object) -> list[dict]:
    project = validate_project(project)
    if not isinstance(specification, dict) or specification.get('version') != 1 or isinstance(specification.get('version'), bool):
        raise BuildError('Variant specification needs version 1')
    variants = specification.get('variants')
    if not isinstance(variants, list) or not variants:
        raise BuildError('variants must be a nonempty array')
    glossary = specification.get('glossary', {})
    if not isinstance(glossary, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in glossary.items()):
        raise BuildError('glossary must map source terms to required translated terms')
    known = {scene['id'] for scene in project['scenes']}
    seen = set()
    result = []
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(variant.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', variant['id']):
            raise BuildError('Each variant needs a portable id of up to 64 letters, digits, underscores or hyphens')
        identifier = variant['id']
        if re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]', identifier, re.IGNORECASE):
            raise BuildError(f'Reserved filesystem name: {identifier}')
        if identifier.casefold() in seen:
            raise BuildError(f'Duplicate variant id: {identifier}')
        seen.add(identifier.casefold())
        locale = variant.get('locale', project.get('locale', 'en'))
        if not isinstance(locale, str) or not re.fullmatch(r'[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*', locale):
            raise BuildError(f'{identifier}: invalid locale')
        kind = variant.get('kind', 'locale')
        if kind not in ('locale', 'campaign'):
            raise BuildError('kind must be locale or campaign')
        overrides = variant.get('scenes', {})
        if not isinstance(overrides, dict) or set(overrides) - known:
            raise BuildError(f'{identifier}: scene overrides contain unknown IDs')
        derived = copy.deepcopy(project)
        # A variation needs its own execution/review/output state.
        for key in ('production', 'sessions', 'publish', 'output'):
            derived.pop(key, None)
        derived['name'] = f"{project['name']} - {identifier}"
        derived['variant'] = {'id': identifier, 'kind': kind, 'sourceName': project['name']}
        derived['locale'] = locale
        derived['phase'] = 'planning'
        derived['render'] = {**derived.get('render', {}), 'file': f'out/{identifier}/video.mp4'}
        if 'format' in variant:
            derived['format'] = copy.deepcopy(variant['format'])
        if 'voice' in variant:
            if not isinstance(variant['voice'], dict):
                raise BuildError(f'{identifier}: voice must be an object')
            derived['voice'] = copy.deepcopy(variant['voice'])
        for scene in derived['scenes']:
            update = overrides.get(scene['id'], {})
            if not isinstance(update, dict) or set(update) - set(TRANSLATABLE) - {'durationSeconds', 'audioFile', 'captionsFile', 'narratorFile'}:
                raise BuildError(f"{identifier}/{scene['id']}: unsupported scene override")
            if kind == 'locale':
                for field in TRANSLATABLE:
                    original = scene.get(field)
                    if isinstance(original, str) and original.strip() and field not in update:
                        raise BuildError(f"{identifier}/{scene['id']}: missing translated {field}")
            for field in TRANSLATABLE:
                if field in update and update[field] is not None and not isinstance(update[field], str):
                    raise BuildError(f"{identifier}/{scene['id']}/{field}: expected text or null")
                target = update.get(field, scene.get(field))
                if kind == 'locale' and isinstance(target, str):
                    for source_term, target_term in glossary.items():
                        if source_term in (scene.get(field) or '') and target_term not in target:
                            raise BuildError(f"{identifier}/{scene['id']}: translation must preserve glossary term {target_term!r}")
            changed = update.get('narration', scene.get('narration')) != scene.get('narration') or kind == 'locale'
            if changed:
                for field in ('audio', 'audioFile', 'captionsFile', 'scriptFile', 'narratorFile'):
                    scene.pop(field, None)
            scene.update(update)
            scene.pop('reviewStatus', None)
        result.append(validate_project(derived))
    return result


def materialize(project_root: Path, derived: list[dict], *, write: bool = False, with_audio: bool = False) -> dict:
    paths = []
    prepared = []
    tasks = []
    for project in derived:
        identifier = project['variant']['id']
        directory = project_path(project_root, f'variants/{identifier}')
        if write and directory.exists():
            raise BuildError(f'Variant directory already exists: {directory}. Use a new variant ID to preserve prior work.')
        normalized = copy.deepcopy(project)
        scripts = []
        audio_tasks = []
        for scene in normalized['scenes']:
            if scene.get('narration'):
                slug = re.sub(r'[^A-Za-z0-9_-]', '-', scene['id']) + '-' + str(len(scripts) + 1)
                script = f'variants/{identifier}/scripts/{slug}.txt'
                scene['scriptFile'] = script
                scripts.append((script, scene['narration']))
                if with_audio and not scene.get('audioFile'):
                    voice = normalized.get('voice', {})
                    audio = f'variants/{identifier}/audio/{slug}.mp3'
                    scene['audioFile'] = audio
                    scene['captionsFile'] = audio[:-4] + '.captions.json'
                    task_id = f'vo-{identifier}-{slug}'
                    audio_tasks.append(task_id)
                    command = ['{python}', '{toolkit}/tools/voiceover.py', '--script', '{project}/' + script,
                        '--output', '{project}/public/' + audio, '--provider', voice.get('provider', 'qwen3'),
                        '--cloud', voice.get('cloud', 'modal'), '--json']
                    if voice.get('voiceId'):
                        command += ['--voice-id', voice['voiceId']]
                    if voice.get('language'):
                        command += ['--language', voice['language']]
                    tasks.append({'id': task_id, 'command': command, 'inputs': [script],
                        'outputs': ['public/' + audio], 'toolkitInputs': ['tools/voiceover.py', 'tools/pacing.py']})
                    caption_id = 'cap-' + task_id
                    tasks.append({'id': caption_id, 'dependsOn': [task_id],
                        'command': ['{python}', '{toolkit}/tools/captions.py', '{project}/public/' + audio,
                                    '--script', '{project}/' + script, '--engine', 'auto'],
                        'inputs': ['public/' + audio, script], 'outputs': ['public/' + scene['captionsFile']]})
                    audio_tasks.append(caption_id)
        prepared.append((directory, normalized, scripts))
        paths.append({'id': identifier, 'project': str(directory / 'project.json'),
                      'props': str(directory / 'render-props.json'), 'scripts': len(scripts)})
        if project['template'] == 'storyboard':
            props_file = f'variants/{identifier}/render-props.json'
            deps = audio_tasks
            if with_audio and audio_tasks:
                retime_id = f'retime-{identifier}'
                tasks.append({'id': retime_id, 'dependsOn': audio_tasks, 'estimatedCostUsd': 0,
                    'command': ['{python}', '{toolkit}/tools/variants.py', '--project', '{project}', '--retime-props', props_file, '--write'],
                    'inputs': [props_file, f'public/variants/{identifier}/audio'], 'outputs': [props_file]})
                deps = [retime_id]
            from storyboard_plan import storyboard_composition
            composition = storyboard_composition(project.get('format'))
            tasks.append({'id': f'render-{identifier}', 'dependsOn': deps, 'estimatedCostUsd': 0,
                'command': ['node', '{project}/node_modules/@remotion/cli/remotion-cli.js', 'render', composition,
                    '{project}/' + project['render']['file'], '--props', '{project}/' + props_file],
                'inputs': ['src', 'public', props_file], 'toolkitInputs': ['lib'], 'outputs': [project['render']['file']]})
    if write:
        for directory, project, scripts in prepared:
            # All variants have passed validation and collision checks first.
            atomic_write(directory / 'project.json', json.dumps(project, ensure_ascii=False, indent=2) + '\n')
            atomic_write(directory / 'render-props.json', json.dumps({'project': project, 'safeZones': False}, ensure_ascii=False, indent=2) + '\n')
            for filename, text in scripts:
                atomic_write(project_path(project_root, filename), text + '\n')
        if tasks:
            atomic_write(project_path(project_root, 'variants/production.json'), json.dumps({'version': 1, 'tasks': tasks}, indent=2) + '\n')
    return {'written': write, 'variants': paths, 'production': {'version': 1, 'tasks': tasks}}


def retime_props(root: Path, filename: str) -> dict:
    target = project_path(root, filename)
    props = json.loads(target.read_text(encoding='utf-8'))
    project = validate_project(props['project'])
    for scene in project['scenes']:
        audio = scene.get('audioFile') or scene.get('audio', {}).get('file')
        if not audio:
            continue
        file = project_path(root, 'public/' + audio)
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'default=noprint_wrappers=1:nokey=1', str(file)],
                                capture_output=True, text=True, check=True)
        duration = float(result.stdout)
        if duration <= 0:
            raise BuildError(f'Invalid audio duration: {audio}')
        scene['durationSeconds'] = duration + .6
    project = validate_project(project)
    atomic_write(target, json.dumps({**props, 'project': project}, ensure_ascii=False, indent=2) + '\n')
    return project


def main(argv=None) -> int:
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--spec', type=Path, help='Translation/campaign variant JSON')
    parser.add_argument('--write', action='store_true', help='Create files; preview is the default')
    parser.add_argument('--with-audio', action='store_true', help='Plan per-variant narration, captions and audio-driven timing')
    parser.add_argument('--retime-props', help='Update scene lengths from declared narration audio')
    args = parser.parse_args(argv)
    try:
        if args.retime_props:
            if not args.write:
                raise BuildError('--retime-props requires --write')
            result = retime_props(args.project, args.retime_props)
        else:
            if not args.spec:
                raise BuildError('--spec is required')
            project = load_project(args.project)
            derived = derive_variants(project, json.loads(args.spec.read_text(encoding='utf-8')))
            result = materialize(args.project.resolve(), derived, write=args.write, with_audio=args.with_audio)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BuildError, ProjectValidationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
