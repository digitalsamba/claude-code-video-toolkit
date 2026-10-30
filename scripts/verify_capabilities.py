#!/usr/bin/env python3
"""Run the local production example, inspect real outputs and prove checkpoint resume."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from media_ops import load_json, probe, run, sha256, write_json


def verify(project):
    module_spec = importlib.util.spec_from_file_location('capability_example', ROOT / 'examples/production-capabilities/create.py')
    example = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(example)
    plan = example.create(project)
    cli = [sys.executable, ROOT / 'tools/build.py', '--project', project, '--plan', project / 'production.json', '--budget-usd', '0', '--json']
    preview = json.loads(run(cli))
    if (project / '.video-toolkit').exists() or not preview['budget']['allowed']:
        raise ValueError('Preview wrote state or blocked the free example')
    built = json.loads(run([*cli, '--run']))
    if not built['success'] or len(built['completed']) != len(plan['tasks']):
        raise ValueError(f'Example did not complete: {built}')
    outputs = {}
    for name, dimensions in [('chart', (320, 180)), ('dubbed', (320, 180)), ('mixed', (320, 180)), ('highlights', (320, 180)), ('portrait', (90, 160)), ('composite', (320, 180))]:
        path = project / f'out/{name}.mp4'
        info = probe(path)
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        if (video['width'], video['height']) != dimensions:
            raise ValueError(f'Incorrect {name} dimensions')
        if not any(s['codec_type'] == 'audio' for s in info['streams']):
            raise ValueError(f'{name} lost its audio')
        outputs[name] = {'path': str(path), 'sha256': sha256(path), 'durationSeconds': info['duration'], 'dimensions': list(dimensions)}
    edits = load_json(project / 'out/edits.json')
    if abs(outputs['highlights']['durationSeconds'] - edits['durationSeconds']) > .15:
        raise ValueError('Rendered highlights differ from edit decisions')
    evidence = load_json(project / 'out/evidence.json')
    if not evidence['success']:
        raise ValueError('Evidence check failed')
    if any(item['level'] == 'error' for item in load_json(project / 'out/qc.json')['findings']):
        raise ValueError('Final QC contains errors')
    characters = load_json(project / 'out/characters.plan.json')
    if len(characters['tasks']) != 2 or characters['tasks'][1]['dependsOn'] != ['wave-image']:
        raise ValueError('Character image-to-video dependency missing')
    resumed = json.loads(run([*cli, '--run']))
    if resumed['completed'] or len(resumed['resumed']) != len(plan['tasks']):
        raise ValueError('Unchanged example did not resume every task')
    return {'version': 1, 'success': True, 'project': str(project), 'tasksCompleted': len(built['completed']),
            'tasksResumed': len(resumed['resumed']), 'media': outputs,
            'providerInference': 'not called; plans and mocked transport tests only',
            'optionalModels': 'RVM tested separately with VIDEO_TOOLKIT_RVM_MODEL; WhisperX/Demucs not required'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, help='New directory for persistent artifacts; otherwise uses a temporary directory')
    parser.add_argument('--report', type=Path, help='Write a fresh verification JSON report')
    args = parser.parse_args(argv)
    try:
        if args.project:
            result = verify(args.project.resolve())
        else:
            with tempfile.TemporaryDirectory(prefix='video-capabilities-') as folder:
                result = verify(Path(folder) / 'demo')
            result['artifactsRetained'] = False
        if args.report:
            write_json(args.report, result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
