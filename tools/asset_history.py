#!/usr/bin/env python3
"""Searchable, content-addressed versions of a project's generated assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AssetHistoryError(ValueError):
    """An invalid asset, record, or restore request."""


def resolved_path(value: Path) -> Path:
    resolved = value.resolve()
    # Windows can retain the extended prefix when another worker creates a
    # parent during realpath(). Compare the same path spelling on both sides.
    if os.name == 'nt':
        text = str(resolved)
        if text.startswith('\\\\?\\UNC\\'):
            return Path('\\\\' + text[8:])
        if text.startswith('\\\\?\\'):
            return Path(text[4:])
    return resolved


def project_path(project_root: Path, value: Path) -> Path:
    root = resolved_path(project_root)
    path = resolved_path(value if value.is_absolute() else root / value)
    if not path.is_relative_to(root):
        raise AssetHistoryError(f"Asset path must stay inside the project: {value} (resolved {path}, root {root})")
    return path


def history_dir(project_root: Path) -> Path:
    return project_path(project_root, Path('.video-toolkit/assets'))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_asset(project_root: Path, asset: Path, *, metadata: dict | None = None,
                 kind: str = 'asset', scene_id: str | None = None) -> dict:
    """Snapshot bytes and supplied provenance without changing the source asset."""
    root = resolved_path(project_root)
    source = project_path(root, asset)
    if not source.is_file():
        raise AssetHistoryError(f"Asset does not exist: {source}")
    store = history_dir(root)
    if source.is_relative_to(store):
        raise AssetHistoryError('Cannot record the history store itself')
    if not isinstance(kind, str) or not kind.strip():
        raise AssetHistoryError('kind must be a non-empty string')
    if scene_id is not None and (not isinstance(scene_id, str) or not scene_id.strip()):
        raise AssetHistoryError('scene_id must be a non-empty string')
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise AssetHistoryError('metadata must be a JSON object')
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise AssetHistoryError(f'metadata must contain JSON values: {error}') from error
    objects = project_path(root, store / 'objects')
    objects.mkdir(parents=True, exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open('rb') as incoming, tempfile.NamedTemporaryFile(
                dir=objects, suffix='.tmp', delete=False) as outgoing:
            temporary = Path(outgoing.name)
            for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                outgoing.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        checksum = digest.hexdigest()
        blob = project_path(root, objects / checksum)
        try:
            os.link(temporary, blob)
        except FileExistsError:
            if sha256_file(blob) != checksum:
                raise AssetHistoryError(f'Corrupt history object: {checksum}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    record = {
        'id': uuid.uuid4().hex,
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'path': source.relative_to(root).as_posix(),
        'sha256': checksum,
        'bytes': size,
        'kind': kind,
        'sceneId': scene_id,
        'metadata': metadata,
    }
    atomic_json(project_path(root, store / 'records' / f"{record['id']}.json"), record)
    return record


def read_record(project_root: Path, record_id: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{32}', record_id):
        raise AssetHistoryError('Asset version ID must be 32 lowercase hexadecimal characters')
    path = project_path(project_root, history_dir(project_root) / 'records' / f'{record_id}.json')
    if not path.is_file():
        raise AssetHistoryError(f'Unknown asset version: {record_id}')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        raise AssetHistoryError(f'Cannot read asset version {record_id}: {error}') from error
    if (not isinstance(record, dict) or record.get('id') != record_id
            or not isinstance(record.get('sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])
            or not isinstance(record.get('path'), str)
            or not isinstance(record.get('createdAt'), str)):
        raise AssetHistoryError(f'Malformed asset version: {record_id}')
    return record


def list_assets(project_root: Path, *, query: str = '', kind: str | None = None,
                scene_id: str | None = None) -> list[dict]:
    """Search version metadata; listing never creates a history directory."""
    records = history_dir(project_root) / 'records'
    if not records.is_dir():
        return []
    result = []
    for path in records.glob('*.json'):
        record = read_record(project_root, path.stem)
        if kind is not None and record.get('kind') != kind:
            continue
        if scene_id is not None and record.get('sceneId') != scene_id:
            continue
        if query.casefold() not in json.dumps(record, ensure_ascii=False).casefold():
            continue
        result.append(record)
    return sorted(result, key=lambda record: (record['createdAt'], record['id']), reverse=True)


def restore_asset(project_root: Path, record_id: str, *, output: Path | None = None,
                  force: bool = False) -> Path:
    record = read_record(project_root, record_id)
    blob = project_path(project_root, history_dir(project_root) / 'objects' / record['sha256'])
    if not blob.is_file() or sha256_file(blob) != record['sha256']:
        raise AssetHistoryError(f"Missing or corrupt history object: {record['sha256']}")
    destination = project_path(project_root, output if output is not None else Path(record['path']))
    if destination.is_relative_to(history_dir(project_root)):
        raise AssetHistoryError('Cannot restore into the history store')
    if destination.exists() and not force:
        raise AssetHistoryError(f'Output already exists: {destination}. Use --force to replace it.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            with blob.open('rb') as incoming:
                shutil.copyfileobj(incoming, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, destination)
        else:
            # Link the complete file atomically, failing if a concurrent writer won.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Project directory')
    parser.add_argument('--json', action='store_true', help='Machine-readable output')
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record', help='Save an immutable version and provenance')
    record.add_argument('asset', type=Path, help='Path relative to the project, or absolute')
    record.add_argument('--kind', default='asset')
    record.add_argument('--scene')
    record.add_argument('--metadata', type=Path, help='JSON object containing prompt/model/seed/source')
    search = commands.add_parser('list', help='Search recorded versions')
    search.add_argument('--query', default='')
    search.add_argument('--kind')
    search.add_argument('--scene')
    inspect = commands.add_parser('inspect', help='Show a version and its provenance')
    inspect.add_argument('id')
    restore = commands.add_parser('restore', help='Restore an earlier version')
    restore.add_argument('id')
    restore.add_argument('--output', type=Path, help='Defaults to the recorded project path')
    restore.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not args.project.is_dir():
            raise AssetHistoryError(f'Project directory does not exist: {args.project}')
        if args.command == 'record':
            metadata = json.loads(args.metadata.read_text(encoding='utf-8')) if args.metadata else None
            result = record_asset(args.project, args.asset, metadata=metadata,
                                  kind=args.kind, scene_id=args.scene)
        elif args.command == 'list':
            result = list_assets(args.project, query=args.query, kind=args.kind, scene_id=args.scene)
        elif args.command == 'inspect':
            result = read_record(args.project, args.id)
        else:
            path = restore_asset(args.project, args.id, output=args.output, force=args.force)
            result = {'id': args.id, 'restoredTo': str(path)}
    except (AssetHistoryError, OSError, ValueError) as error:
        if args.json:
            print(json.dumps({'error': str(error)}, ensure_ascii=False))
        else:
            print(f'Error: {error}', file=sys.stderr)
        return 1
    if args.json or not isinstance(result, list):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for item in result:
            print(f"{item['id']}  {item['kind']}  {item['path']}  {item['createdAt']}")
        print(f'{len(result)} version(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
