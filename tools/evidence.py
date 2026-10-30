#!/usr/bin/env python3
"""Link script claims to local quotes or dataset cells and flag missing/stale evidence."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from media_ops import load_json, output_path, sha256, write_json
from project_manifest import load_project


def normalize(value):
    return ' '.join(value.split())


def path_at(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Source and script paths must be nonempty strings')
    return (root / value).resolve()


def spec_data(path):
    value = load_json(path)
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1:
        raise ValueError('Evidence spec version must be 1')
    if not isinstance(value.get('sources'), dict) or not value['sources']:
        raise ValueError('Evidence spec needs sources keyed by ID')
    for name, source in value['sources'].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(source, dict):
            raise ValueError('Sources must be named objects')
        if not isinstance(source.get('path'), str):
            raise ValueError(f'Source {name} needs path')
    if not isinstance(value.get('claims'), list) or not value['claims']:
        raise ValueError('Evidence spec needs a nonempty claims array')
    if ('script' in value) == ('project' in value):
        raise ValueError('Evidence spec must select either script or project')
    return value


def content_blocks(spec, root):
    if 'script' in spec:
        path = path_at(root, spec['script'])
        return path, [{'id': None, 'text': normalize(path.read_text(encoding='utf-8-sig')), 'startSeconds': 0}]
    path = path_at(root, spec['project'])
    if path.is_dir():
        path /= 'project.json'
    project = load_project(path)
    fields = spec.get('textFields', ['title', 'narration', 'onScreenText'])
    if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or not f for f in fields):
        raise ValueError('textFields must list scene text field names')
    blocks, cursor = [], 0
    for scene in project['scenes']:
        values = [scene.get(field) for field in fields if scene.get(field) is not None]
        if any(not isinstance(value, str) for value in values):
            raise ValueError('Selected project text fields must contain strings')
        start = scene.get('startSeconds', cursor)
        blocks.append({'id': scene['id'], 'text': normalize(' '.join(values)), 'startSeconds': start})
        cursor = start + scene.get('durationSeconds', 0)
    return path, blocks


def lock_spec(spec_file, output):
    value = spec_data(spec_file)
    root, target_root = Path(spec_file).resolve().parent, Path(output).resolve().parent
    content_file, _ = content_blocks(value, root)
    key = 'script' if 'script' in value else 'project'
    value[key] = Path(os.path.relpath(content_file, target_root)).as_posix()
    value[key + 'Sha256'] = sha256(content_file)
    for source in value['sources'].values():
        path = path_at(root, source['path'])
        if path.suffix.lower() not in ('.txt', '.md', '.csv', '.json'):
            raise ValueError('Sources must be local UTF-8 .txt, .md, .csv or .json files')
        path.read_text(encoding='utf-8-sig')
        source['sha256'] = sha256(path)
        source['path'] = Path(os.path.relpath(path, target_root)).as_posix()
    return value


def selected_value(path, selector):
    if not isinstance(selector, dict):
        raise ValueError('selector must be an object')
    if path.suffix.lower() == '.csv':
        row, column = selector.get('row'), selector.get('column')
        if type(row) is not int or row < 0 or not isinstance(column, str):
            raise ValueError('CSV selector needs zero-based row and column name')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV source needs unique headers')
            rows = list(reader)
        if row >= len(rows) or column not in rows[row]:
            raise ValueError('CSV cell does not exist')
        return rows[row][column]
    if path.suffix.lower() != '.json':
        raise ValueError('Cell selectors require .csv or .json sources')
    pointer = selector.get('pointer')
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise ValueError('JSON selector needs an RFC 6901 pointer')
    value = load_json(path)
    for part in pointer.split('/')[1:] if pointer else []:
        if re.search(r'~(?![01])', part):
            raise ValueError('Invalid JSON pointer escape')
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list) and re.fullmatch(r'0|[1-9][0-9]*', part):
            index = int(part)
            if index >= len(value):
                raise ValueError('JSON pointer index does not exist')
            value = value[index]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError('JSON pointer does not exist')
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError('Evidence cell must contain text or a finite number')
    return value


def equal_value(actual, expected):
    if isinstance(expected, bool) or not isinstance(expected, (str, int, float)):
        raise ValueError('expected must be text or a finite number')
    if isinstance(expected, str):
        return isinstance(actual, str) and normalize(actual) == normalize(expected)
    try:
        left, right = Decimal(str(actual)), Decimal(str(expected))
        if not left.is_finite() or not right.is_finite():
            raise ValueError('Evidence values must be finite')
        return left == right
    except InvalidOperation:
        return False


def check(spec_file, *, strict=False):
    spec = spec_data(spec_file)
    root = Path(spec_file).resolve().parent
    content_file, blocks = content_blocks(spec, root)
    findings, claims, sources = [], [], {}
    def finding(code, message, claim=None, level='error'):
        item = {'code': code, 'level': level, 'message': message}
        if claim:
            item['claimId'] = claim['id']
            if isinstance(claim.get('sceneId'), str):
                item['sceneId'] = claim['sceneId']
                block = next((b for b in blocks if b['id'] == claim['sceneId']), None)
                if block:
                    item['timeSeconds'] = block['startSeconds']
        findings.append(item)
    key = 'script' if 'script' in spec else 'project'
    if spec.get(key + 'Sha256') != sha256(content_file):
        finding('stale-script', 'Script differs from its evidence lock. Review changes and create a new lock.')
    for name, source in spec['sources'].items():
        path = path_at(root, source['path'])
        try:
            digest = sha256(path)
            sources[name] = {**source, 'resolvedPath': str(path), 'currentSha256': digest}
            if source.get('sha256') != digest:
                finding('stale-source', f'Source {name} changed or has no locked hash')
        except OSError:
            finding('missing-source', f'Source {name} does not exist: {path}')
    seen, covered = set(), {block['id']: [] for block in blocks}
    for claim in spec['claims']:
        if not isinstance(claim, dict) or not isinstance(claim.get('id'), str) or not claim['id'].strip():
            raise ValueError('Each claim needs a nonempty ID')
        if claim['id'] in seen:
            raise ValueError('Duplicate claim ID')
        seen.add(claim['id'])
        content = claim.get('text')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Each claim needs its exact script text')
        content = normalize(content)
        status = 'missing-evidence'
        target_block = next((b for b in blocks if b['id'] == claim.get('sceneId')), None) if 'project' in spec else blocks[0]
        if target_block is None:
            raise ValueError('Project claims must name an existing sceneId')
        if content not in target_block['text']:
            finding('claim-not-in-script', 'Claim text does not appear in the script', claim)
        else:
            # Cover every literal occurrence, leaving other script content visible.
            covered[target_block['id']].append(content)
        source_id = claim.get('source')
        if not isinstance(source_id, str) or source_id not in sources:
            finding('missing-evidence', 'Claim references an unknown or missing source', claim)
        else:
            path = Path(sources[source_id]['resolvedPath'])
            try:
                if ('quote' in claim) == ('selector' in claim):
                    raise ValueError('Use exactly one quote or cell selector per claim')
                if 'quote' in claim:
                    quote = claim['quote']
                    if not isinstance(quote, str) or not quote.strip():
                        raise ValueError('quote must be nonempty text')
                    quote = normalize(quote)
                    if quote not in normalize(path.read_text(encoding='utf-8-sig')):
                        finding('quote-not-found', 'Quoted evidence does not occur in the source', claim)
                    elif quote == content:
                        status = 'quote-found'
                    else:
                        status = 'needs-human-review'
                        finding('semantic-review', 'Quote exists, but the claim is a paraphrase. Review its meaning and context.', claim, 'warning')
                else:
                    actual = selected_value(path, claim['selector'])
                    if not equal_value(actual, claim.get('expected')):
                        finding('value-mismatch', f'Dataset cell is {actual!r}, expected {claim.get("expected")!r}', claim)
                    else:
                        display = claim.get('display', str(claim['expected']))
                        if not isinstance(display, str) or not display or not re.search(r'(?<!\w)' + re.escape(display) + r'(?!\w)', content):
                            raise ValueError('display must appear as a complete value in the claim text')
                        if isinstance(claim['expected'], (int, float)) and not equal_value(display, claim['expected']):
                            raise ValueError('display must represent the same numeric value as expected')
                        status = 'value-matches'
                        finding('semantic-review', 'Dataset value matches. Review the claim label, unit and interpretation.', claim, 'warning')
            except (OSError, ValueError) as error:
                finding('invalid-evidence', str(error), claim)
        claims.append({'id': claim['id'], 'text': content, 'source': source_id, 'status': status})
    for block in blocks:
        remaining = block['text']
        for content in sorted(covered[block['id']], key=len, reverse=True):
            remaining = remaining.replace(content, ' ')
        remaining = re.sub(r'\[pause\s+[\d.]+s\]', ' ', remaining)
        if re.search(r'\w', remaining, re.UNICODE):
            finding('uncited-text', 'Unlinked script content: ' + normalize(remaining)[:500],
                    {'id': 'uncited-' + block['id'], 'sceneId': block['id']} if block['id'] else None)
    success = not any(f['level'] == 'error' or strict for f in findings)
    return {'version': 1, 'success': success, 'strict': strict, 'claims': claims, 'findings': findings,
            'sources': sources, 'scope': 'Evidence location, values, hashes and coverage; not semantic truth verification'}


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    lock = commands.add_parser('lock')
    lock.add_argument('--spec', type=Path, required=True)
    lock.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('check')
    verify.add_argument('--spec', type=Path, required=True)
    verify.add_argument('--output', type=Path)
    verify.add_argument('--strict', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'lock':
            output_path(args.output, args.spec)
            result = lock_spec(args.spec, args.output)
            write_json(args.output, result)
        else:
            result = check(args.spec, strict=args.strict)
            if args.output:
                output_path(args.output, args.spec)
                write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('success', True) else 1
    except (OSError, ValueError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
