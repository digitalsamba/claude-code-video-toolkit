#!/usr/bin/env python3
"""Verify production features offline, optionally including real browser/render checks."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT, timeout: int = 180, env=None, combined: bool = False) -> str:
    print('Checking: ' + ' '.join(command[:4]), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, text=True, encoding='utf-8', errors='replace',
                            capture_output=True, timeout=timeout, shell=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if not combined and result.stderr and 'warning' not in result.stderr.lower():
        print(result.stderr[-1000:])
    return result.stdout + result.stderr if combined else result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', action='store_true', help='Compile helpers, typecheck and render actual template frames/video')
    parser.add_argument('--browser', action='store_true', help='Exercise flow recorder and review UI in a real browser')
    parser.add_argument('--capabilities', action='store_true', help='Run the eleven-task local production graph; requires the vision extra')
    args = parser.parse_args(argv)
    evidence = {'python': False, 'media': False, 'browser': False, 'renders': []}
    try:
        output = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-q'], combined=True)
        print(output[-1000:])
        evidence['python'] = True
        match = re.search(r'Ran (\d+) tests? in', output)
        if match:
            evidence['pythonTests'] = int(match[1])
        if args.capabilities:
            (ROOT / 'scratch').mkdir(exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix='capability-verification-', dir=ROOT / 'scratch'))
            evidence['capabilities'] = json.loads(run([sys.executable, 'scripts/verify_capabilities.py',
                '--project', str(folder / 'demo'), '--report', str(folder / 'report.json')], timeout=600))
        browser = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
        if not browser and (args.browser or args.media):
            browser = run(['node', '-e', 'process.stdout.write(require("playwright").chromium.executablePath())'], cwd=ROOT / 'playwright').strip()
            if not Path(browser).is_file():
                raise RuntimeError('Install the browser with cd playwright && npx playwright install chromium')
        environment = {**os.environ, 'VIDEO_TOOLKIT_PYTHON': sys.executable}
        if browser:
            environment['PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH'] = browser
        # executablePath() returns full Chromium, not chrome-headless-shell.
        remotion_browser_args = ['--chrome-mode=chrome-for-testing', '--browser-executable=' + browser] if browser else []
        if args.media:
            for helper in ('cues', 'formats'):
                run(['node', 'templates/storyboard/node_modules/typescript/bin/tsc', f'lib/{helper}.ts',
                    '--target', 'ES2020', '--module', 'commonjs', '--strict', '--outDir', f'scratch/{helper}-ts'])
                run(['node', '--test', f'tests/{helper}.test.cjs'])
            run(['node', 'node_modules/typescript/bin/tsc', '--noEmit'], cwd=ROOT / 'templates/storyboard')
            run(['node', 'node_modules/typescript/bin/tsc', '--noEmit'], cwd=ROOT / 'examples/narration-cues')
            for composition, dimensions in [('Storyboard', (480, 270)), ('StoryboardPortrait', (270, 480)), ('StoryboardSquare', (270, 270))]:
                folder = ROOT / 'templates/storyboard'
                destination = f'out/verified-{composition}.mp4'
                run(['node', 'node_modules/@remotion/cli/remotion-cli.js', 'render', composition, destination,
                    '--frames=15-29', '--scale=0.25', '--concurrency=1', *remotion_browser_args, '--log=error'], cwd=folder)
                probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', destination], cwd=folder))
                stream = next(item for item in probe['streams'] if item['codec_type'] == 'video')
                if (stream['width'], stream['height']) != dimensions:
                    raise RuntimeError(f'{composition} rendered incorrect dimensions')
                evidence['renders'].append({'composition': composition, 'width': stream['width'], 'height': stream['height']})
            run(['node', 'node_modules/@remotion/cli/remotion-cli.js', 'still', 'NarrationCues', 'out/verified.png',
                '--frame=75', *remotion_browser_args, '--log=error'], cwd=ROOT / 'examples/narration-cues')
            sys.path.insert(0, str(ROOT / 'tools'))
            from build import builtin_plan
            from variants import derive_variants, materialize
            folder = ROOT / 'templates/storyboard'
            plan = builtin_plan(folder)
            render_task = next(task for task in plan['tasks'] if task['id'] == 'render')
            render_task['command'] += ['--frames=15-29', '--scale=0.25', '--concurrency=1', *remotion_browser_args]
            plan_file = folder / '.video-toolkit/verification-plan.json'
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(json.dumps(plan), encoding='utf-8')
            result = json.loads(run([sys.executable, 'tools/build.py', '--project', str(folder), '--plan', str(plan_file), '--run', '--budget-usd', '0', '--json']))
            if not result['success']:
                raise RuntimeError(f'Storyboard build failed: {result}')
            evidence['storyboardBuild'] = result
            base = json.loads((folder / 'project.json').read_text(encoding='utf-8'))
            specification = {'version': 1, 'variants': [{'id': 'es-check', 'locale': 'es', 'scenes': {
                scene['id']: {'title': 'Una historia para todos', 'onScreenText': 'El próximo vídeo empieza aquí.', 'durationSeconds': .7}
                for scene in base['scenes']}}]}
            (ROOT / 'scratch').mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=ROOT / 'scratch', prefix='verify-variants-') as directory:
                variant = materialize(Path(directory), derive_variants(base, specification), write=True)
                props_file = variant['variants'][0]['props']
                run(['node', 'node_modules/@remotion/cli/remotion-cli.js', 'render', 'Storyboard', 'out/verified-Spanish.mp4',
                     '--props', props_file, '--frames=15-29', '--scale=0.25', '--concurrency=1', *remotion_browser_args, '--log=error'], cwd=folder)
            evidence['localizedRender'] = 'templates/storyboard/out/verified-Spanish.mp4'
            evidence['media'] = True
        if args.browser:
            run(['node', 'node_modules/typescript/bin/tsc', '--noEmit', '--target', 'ES2022', '--module', 'NodeNext',
                '--moduleResolution', 'NodeNext', '--strict', '--esModuleInterop', '--skipLibCheck',
                'scripts/flow.ts', 'tests/flow.test.ts'], cwd=ROOT / 'playwright')
            run(['node', '--import', 'tsx', '--test', 'tests/flow.test.ts'], cwd=ROOT / 'playwright', env=environment)
            run(['node', '--test', 'tests/review.test.mjs'], env=environment)
            evidence['browser'] = True
        destination = ROOT / 'scratch/feature-verification.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(evidence, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
