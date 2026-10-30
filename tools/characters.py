#!/usr/bin/env python3
"""Lock character references and plan consistent FLUX stills and LTX video shots."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

from asset_history import record_asset
from media_ops import (
    load_json,
    number,
    output_path,
    probe,
    sha256,
    staged_output,
    video_stream,
    write_json,
)
from production import project_path, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', value):
        raise ValueError('Character and shot IDs must use lowercase letters, numbers, hyphens or underscores')
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be nonempty text')
    return value.strip()


def image_info(path):
    from PIL import Image
    with Image.open(path) as image:
        size = list(image.size)
        image.verify()
    return {'path': str(path), 'sha256': sha256(path), 'dimensions': size}


def lock(spec, project):
    if not isinstance(spec, dict) or type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Character spec version must be 1')
    style = text(spec.get('style'), 'Visual style and lighting')
    definitions = spec.get('characters')
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError('Define at least one character')
    characters = {}
    for name, character in definitions.items():
        identity(name)
        if not isinstance(character, dict):
            raise ValueError('Character definitions must be objects')
        references = character.get('references')
        if not isinstance(references, list) or not 1 <= len(references) <= 3:
            raise ValueError('Each character needs one to three reference images')
        images = []
        for reference in references:
            path = project_path(project, reference)
            images.append({**image_info(path), 'path': reference})
        characters[name] = {'description': text(character.get('description'), 'Character description'),
                            'wardrobe': text(character.get('wardrobe'), 'Wardrobe'), 'references': images}
    shots, seen = [], set()
    if not isinstance(spec.get('shots'), list) or not spec['shots']:
        raise ValueError('Define at least one shot')
    for shot in spec['shots']:
        if not isinstance(shot, dict):
            raise ValueError('Shots must be objects')
        name = identity(shot.get('id'))
        if name in seen:
            raise ValueError('Duplicate shot ID')
        seen.add(name)
        cast = shot.get('characters')
        if not isinstance(cast, list) or not cast or any(not isinstance(c, str) or c not in characters for c in cast) or len(set(cast)) != len(cast):
            raise ValueError('Shots must list known, unique character IDs')
        refs = [ref for c in cast for ref in characters[c]['references']]
        if len(refs) > 3:
            raise ValueError('The existing FLUX endpoint supports at most three references per shot')
        seed = shot.get('seed', int(hashlib.sha256((str(spec.get('seed', 0)) + ':' + name).encode()).hexdigest()[:8], 16))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Shot seed must be an unsigned 32-bit integer')
        item = {'id': name, 'characters': cast, 'prompt': text(shot.get('prompt'), 'Shot prompt'), 'seed': seed}
        if shot.get('motionPrompt'):
            item['motionPrompt'] = text(shot['motionPrompt'], 'Motion prompt')
        for key in ('imageCostUsd', 'videoCostUsd'):
            if shot.get(key) is not None:
                item[key] = number(shot[key], key, 0)
        shots.append(item)
    return {'version': 1, 'kind': 'character-lock', 'style': style, 'characters': characters, 'shots': shots}


def checked_lock(value, project):
    if not isinstance(value, dict) or value.get('kind') != 'character-lock':
        raise ValueError('Use a character lock created by the lock command')
    definitions = value.get('characters')
    if not isinstance(definitions, dict) or any(
        not isinstance(c, dict) or not isinstance(c.get('references'), list)
        or any(not isinstance(r, dict) or not isinstance(r.get('path'), str)
               or not isinstance(r.get('sha256'), str) for r in c['references'])
        for c in definitions.values()
    ):
        raise ValueError('Malformed character reference lock')
    # Run the same structural validation, then compare every recorded reference.
    raw = {**value, 'characters': {name: {**character, 'references': [r['path'] for r in character['references']]}
                                  for name, character in value.get('characters', {}).items()}}
    refreshed = lock(raw, project)
    for name, character in refreshed['characters'].items():
        if character['references'] != value['characters'][name]['references']:
            raise ValueError(f'Reference images changed for {name}. Review them and create a new lock.')
    return refreshed


def prompt_for(locked, shot):
    cast = '; '.join(f'{name}: {locked["characters"][name]["description"]}. Wardrobe: {locked["characters"][name]["wardrobe"]}' for name in shot['characters'])
    return f'Preserve the identity and appearance of the characters in the reference images. {cast}. Visual style and lighting: {locked["style"]}. Shot: {shot["prompt"]}. Keep the same faces, clothing and style.'


def production_plan(project, lock_file, output_dir='public/characters', cloud='runpod'):
    locked = checked_lock(load_json(lock_file), project)
    relative_lock = Path(lock_file).resolve().relative_to(Path(project).resolve()).as_posix()
    project_path(project, output_dir)
    if cloud not in ('runpod', 'modal'):
        raise ValueError('FLUX cloud must be runpod or modal')
    tasks = []
    for shot in locked['shots']:
        image = f'{output_dir}/{shot["id"]}.png'
        references = [r['path'] for name in shot['characters'] for r in locked['characters'][name]['references']]
        for kind in ('image', 'video') if shot.get('motionPrompt') else ('image',):
            target = image if kind == 'image' else f'{output_dir}/{shot["id"]}.mp4'
            task = {'id': f'{shot["id"]}-{kind}', 'command': ['{python}', '{toolkit}/tools/characters.py', 'shot',
                    '--project', '{project}', '--lock', '{project}/' + relative_lock, '--id', shot['id'], '--kind', kind,
                    '--output', target, '--cloud', cloud, '--run'],
                    'inputs': [relative_lock, *references], 'outputs': [target],
                    'toolkitInputs': ['tools/characters.py', 'tools/flux2.py', 'tools/ltx2.py', 'tools/cloud_gpu.py'],
                    'environmentInputs': ['RUNPOD_FLUX2_ENDPOINT_ID', 'MODAL_FLUX2_ENDPOINT_URL', 'MODAL_LTX2_ENDPOINT_URL'],
                    'estimatedCostUsd': shot.get('imageCostUsd' if kind == 'image' else 'videoCostUsd'), 'timeoutSeconds': 1800}
            if kind == 'video':
                task['dependsOn'] = [f'{shot["id"]}-image']
                task['inputs'].append(image)
                task['command'] += ['--image', image]
            tasks.append(task)
    return validate_plan({'version': 1, 'tasks': tasks}, Path(project), ROOT)


def render_shot(project, lock_file, shot_id, output, *, kind='image', image=None, cloud='runpod', execute=False):
    locked = checked_lock(load_json(lock_file), project)
    shot = next((s for s in locked['shots'] if s['id'] == shot_id), None)
    if shot is None or kind not in ('image', 'video') or cloud not in ('runpod', 'modal'):
        raise ValueError('Unknown shot, media kind or cloud')
    target = project_path(project, output)
    if target.suffix.lower() != ('.png' if kind == 'image' else '.mp4') or '.video-toolkit' in Path(output).parts:
        raise ValueError('Use .png for image shots or .mp4 for video, outside toolkit state')
    references = [project_path(project, r['path']) for name in shot['characters'] for r in locked['characters'][name]['references']]
    output_path(target, lock_file, *references)
    conditioning = None
    if kind == 'video':
        if not shot.get('motionPrompt') or not image:
            raise ValueError('Video shots need motionPrompt and a generated --image')
        conditioning = project_path(project, image)
        image_info(conditioning)
        output_path(target, conditioning)
    prompt = prompt_for(locked, shot)
    result = {'version': 1, 'executed': execute, 'shot': shot_id, 'kind': kind, 'output': str(target),
              'seed': shot['seed'], 'prompt': prompt, 'referenceSha256': [sha256(r) for r in references],
              'lockSha256': sha256(lock_file)}
    if not execute:
        return result
    with staged_output(target) as temporary, redirect_stdout(sys.stderr):
        if kind == 'image':
            from flux2 import edit_image
            response = edit_image([str(r) for r in references], prompt, str(temporary), seed=shot['seed'], cloud=cloud, open_result=False)
            if not response:
                raise ValueError('FLUX generation failed')
            image_info(temporary)
        else:
            from ltx2 import generate_video
            response = generate_video(prompt=prompt + ' Motion: ' + shot['motionPrompt'], input_path=str(conditioning),
                                      output_path=str(temporary), seed=shot['seed'], cloud='modal', open_result=False)
            if not response:
                raise ValueError('LTX generation failed')
            video_stream(probe(temporary))
    record = record_asset(Path(project), target, kind=kind, metadata={**result, 'provider': 'flux2' if kind == 'image' else 'ltx2'})
    return {**result, 'assetVersion': record['id']}


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('lock')
    create.add_argument('--spec', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--output-dir', default='public/characters')
    plan.add_argument('--save-plan', type=Path)
    shot = commands.add_parser('shot')
    shot.add_argument('--id', required=True)
    shot.add_argument('--kind', choices=['image', 'video'], default='image')
    shot.add_argument('--image')
    shot.add_argument('--output', required=True, help='Project-relative path')
    shot.add_argument('--run', action='store_true')
    for command in (create, plan, shot):
        command.add_argument('--project', type=Path, required=True)
        if command is not create:
            command.add_argument('--lock', type=Path, required=True)
            command.add_argument('--cloud', choices=['runpod', 'modal'], default='runpod')
    args = parser.parse_args(argv)
    try:
        if args.command == 'lock':
            output_path(args.output, args.spec)
            result = lock(load_json(args.spec), args.project)
            write_json(args.output, result)
        elif args.command == 'plan':
            result = production_plan(args.project, args.lock, args.output_dir, args.cloud)
            if args.save_plan:
                output_path(args.save_plan, args.lock)
                write_json(args.save_plan, result)
        else:
            result = render_shot(args.project, args.lock, args.id, args.output, kind=args.kind,
                                 image=args.image, cloud=args.cloud, execute=args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
