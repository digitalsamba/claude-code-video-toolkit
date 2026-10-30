#!/usr/bin/env python3
"""Discover and validate local template/provider contracts, or scaffold a template."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
from production import BuildError, atomic_write, project_path, read_json
from provider import validate_adapter

TOOLKIT = Path(__file__).resolve().parents[1]


def validate_template(directory: Path) -> dict:
    data = read_json(directory / 'template-manifest.json')
    if type(data.get('version')) is not int or data['version'] != 1 or not isinstance(data.get('name'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', data['name']):
        raise BuildError('Template requires version 1 and a portable name')
    if data.get('engine') not in ('remotion', 'python'):
        raise BuildError('Template engine must be remotion or python')
    if not isinstance(data.get('entry'), str) or not project_path(directory, data['entry']).is_file():
        raise BuildError('Template entry must name an existing file within the template')
    formats = data.get('formats')
    if not isinstance(formats, list) or not formats or any(item not in ('landscape', 'portrait', 'square') for item in formats):
        raise BuildError('Template formats must list landscape, portrait or square')
    if data['engine'] == 'remotion':
        if not isinstance(data.get('composition'), str) or not data['composition']:
            raise BuildError('Remotion templates require a composition ID')
        if not (directory / 'package.json').is_file():
            raise BuildError('Remotion templates require package.json')
    return data


def scaffold(directory: Path, name: str) -> dict:
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name):
        raise BuildError('Use a lowercase template name with letters, digits and hyphens')
    if directory.exists():
        raise BuildError('Scaffold destination already exists')
    data = {'version': 1, 'name': name, 'engine': 'python', 'entry': 'build.py',
            'formats': ['landscape'], 'description': 'A minimal local FFmpeg video template'}
    atomic_write(directory / 'template-manifest.json', json.dumps(data, indent=2) + '\n')
    atomic_write(directory / 'build.py', '''import argparse
from pathlib import Path
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('out/video.mp4'))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
    'color=c=0x13251f:s=1920x1080:r=30:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(args.output)], check=True)
''')
    atomic_write(directory / 'README.md', '# ' + name + '\n\nRun `python build.py --output out/video.mp4`. Requires FFmpeg.\n')
    return validate_template(directory)


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    catalog = command.add_parser('list')
    catalog.add_argument('--root', type=Path, default=TOOLKIT)
    validate = command.add_parser('validate')
    validate.add_argument('path', type=Path)
    create = command.add_parser('scaffold')
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--name', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            results = []
            for base in (args.root / 'templates', args.root / 'examples/providers'):
                if not base.is_dir():
                    continue
                for item in sorted(base.iterdir()):
                    if (item / 'template-manifest.json').is_file():
                        results.append({'path': str(item), 'kind': 'template', **validate_template(item)})
                    elif (item / 'adapter.json').is_file():
                        results.append({'path': str(item), 'kind': 'provider', **validate_adapter(read_json(item / 'adapter.json'), item)})
            result = results
        elif args.command == 'validate':
            result = (validate_template(args.path) if args.path.is_dir()
                      else validate_adapter(read_json(args.path), args.path.parent))
        else:
            result = scaffold(args.output, args.name)
        print(json.dumps(result, indent=2))
        return 0
    except (BuildError, OSError, ValueError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
