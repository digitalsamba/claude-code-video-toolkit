#!/usr/bin/env python3
"""A stable JSON/argv boundary for locally installed generation adapters."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from asset_history import record_asset
from production import BuildError, canonical, project_path, read_json
from win_encoding import utf8_env


def validate_adapter(adapter: object, directory: Path) -> dict:
    if not isinstance(adapter, dict) or type(adapter.get('version')) is not int or adapter['version'] != 1:
        raise BuildError('Adapter version must be 1')
    if not isinstance(adapter.get('name'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', adapter['name']):
        raise BuildError('Adapter name must be lowercase letters, numbers and hyphens')
    capabilities = adapter.get('capabilities')
    if not isinstance(capabilities, list) or not capabilities or any(item not in ('image', 'audio', 'video') for item in capabilities):
        raise BuildError('Adapter capabilities must list image, audio or video')
    command = adapter.get('command')
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or '\0' in arg for arg in command):
        raise BuildError('Adapter command must be an argv string array')
    if not all(any(key in arg for arg in command) for key in ('{input}', '{output}')):
        raise BuildError('Adapter command must consume {input} and {output}')
    required = adapter.get('requiredEnv', [])
    if not isinstance(required, list) or any(not isinstance(name, str) or not re.fullmatch(r'[A-Z_][A-Z0-9_]*', name) for name in required):
        raise BuildError('requiredEnv must list environment variable names')
    timeout = adapter.get('timeoutSeconds', 1800)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise BuildError('timeoutSeconds must be an integer between 1 and 86400')
    if not directory.is_dir():
        raise BuildError('Adapter directory does not exist')
    return {**adapter, 'requiredEnv': required, 'timeoutSeconds': timeout}


def prepare(adapter_file: Path, request: object, project: Path, output: str) -> tuple[dict, Path]:
    adapter = validate_adapter(read_json(adapter_file), adapter_file.parent)
    if not isinstance(request, dict) or request.get('capability') not in adapter['capabilities']:
        raise BuildError('Request capability must be supported by the adapter')
    if not isinstance(request.get('prompt'), str) or not request['prompt'].strip():
        raise BuildError('Request needs a nonempty prompt')
    if not isinstance(request.get('parameters', {}), dict):
        raise BuildError('Request parameters must be an object')
    try:
        json.dumps(request, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise BuildError(f'Request must contain finite JSON values: {error}') from error
    if not project.is_dir():
        raise BuildError('Project directory must exist')
    target = project_path(project, output)
    if target.exists():
        raise BuildError('Output already exists. Use a new filename to preserve the current asset.')
    if '.video-toolkit' in target.relative_to(canonical(project)).parts:
        raise BuildError('Adapter output cannot target toolkit state')
    return adapter, target


def run_adapter(adapter_file: Path, request: dict, project: Path, output: str, *, run: bool = False) -> dict:
    adapter_file = adapter_file.resolve()
    adapter, target = prepare(adapter_file, request, project, output)
    missing = [name for name in adapter['requiredEnv'] if not os.environ.get(name)]
    result = {'version': 1, 'provider': adapter['name'], 'capability': request['capability'],
        'output': str(target), 'ready': not missing, 'missingEnv': missing, 'executed': False}
    if not run:
        return result
    if missing:
        raise BuildError('Missing adapter environment: ' + ', '.join(missing))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='video-provider-') as directory:
        input_file = Path(directory) / 'input.json'
        input_file.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        replacements = {'{python}': sys.executable, '{adapter}': str(adapter_file.parent),
            '{input}': str(input_file), '{output}': str(target)}
        command = []
        for arg in adapter['command']:
            for key, value in replacements.items():
                arg = arg.replace(key, value)
            command.append(arg)
        completed = subprocess.run(command, cwd=adapter_file.parent, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=adapter['timeoutSeconds'], shell=False, env=utf8_env())
        if completed.returncode:
            raise BuildError(f"Adapter {adapter['name']} exited {completed.returncode}: {completed.stderr[-2000:]}")
        try:
            response = json.loads(completed.stdout)
        except ValueError as error:
            raise BuildError('Adapter stdout must contain one JSON result object') from error
        if (not isinstance(response, dict) or response.get('version') != 1 or not isinstance(response.get('output'), str)
                or canonical(Path(response['output'])) != target):
            raise BuildError('Adapter result must confirm version 1 and the requested output path')
        if not target.is_file() or target.stat().st_size == 0:
            raise BuildError('Adapter did not produce a nonempty output')
    record = record_asset(project, target, kind=request['capability'], metadata={
        'provider': adapter['name'], 'adapterVersion': adapter['version'], 'prompt': request['prompt'],
        'adapterSha256': hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        'parameters': request.get('parameters', {})})
    return {**result, 'executed': True, 'sha256': record['sha256'], 'assetVersion': record['id']}


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', required=True, help='Project-relative output path')
    parser.add_argument('--run', action='store_true', help='Execute adapter; default only checks readiness')
    args = parser.parse_args(argv)
    try:
        result = run_adapter(args.adapter, read_json(args.input), args.project, args.output, run=args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BuildError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'version': 1, 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
