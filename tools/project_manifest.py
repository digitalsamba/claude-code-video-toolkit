"""Read, validate and atomically write versioned video project manifests.

The canonical schema is lib/project/schema.json. This module deliberately uses
only the standard library so project inspection does not install media/AI tools.
It implements the schema vocabulary used by that file, not arbitrary schemas.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
from datetime import date, datetime


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "lib" / "project" / "schema.json"
SCHEMA_VERSION = 1


def _canonical(path: Path) -> Path:
    value = str(path.resolve())
    if os.name == 'nt' and value.startswith('\\\\?\\UNC\\'):
        value = '\\\\' + value[8:]
    elif os.name == 'nt' and value.startswith('\\\\?\\'):
        value = value[4:]
    return Path(value)


class ProjectValidationError(ValueError):
    """A manifest failed validation; errors have JSON-pointer paths and messages."""

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__("\n".join(f"{e['path']}: {e['message']}" for e in errors))


def manifest_path(path: str | Path) -> Path:
    """Accept an existing project directory or the manifest filename."""
    result = Path(path)
    return result / "project.json" if result.is_dir() else result


def _path_error(
    value: str, *, allow_root: bool = False, allow_glob: bool = True, file_path: bool = False
) -> str | None:
    if not value or not value.strip() or any(ord(c) < 32 for c in value):
        return "must be a nonempty project-relative path without control characters"
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or PureWindowsPath(value).drive or ":" in value:
        return "must be project-relative, not an absolute path, drive path or URL"
    parts = normalized.split("/")
    if ".." in parts:
        return "must stay inside the project; '..' path components are not allowed"
    if not allow_root and all(p in ("", ".") for p in parts):
        return "must name a file or subdirectory, not the project root"
    if not allow_glob and any(c in value for c in "*?[]"):
        return "must name an output file, not a glob pattern"
    if file_path and normalized.endswith("/"):
        return "must name an output file, not a directory"
    return None


def resolve_project_path(
    project_dir: str | Path, value: str, *, allow_root: bool = False
) -> Path:
    """Resolve a local reference, rejecting traversal and escaping symlinks.

    Public-relative visual/audio references need project_dir / 'public' here;
    render, script and production paths use the project directory itself.
    File existence is intentionally not required during project planning.
    """
    problem = _path_error(value, allow_root=allow_root)
    if problem:
        raise ProjectValidationError([{"path": "/path", "message": problem}])
    root = _canonical(Path(project_dir))
    candidate = _canonical(root / value.replace("\\", "/"))
    if not candidate.is_relative_to(root):
        raise ProjectValidationError([{
            "path": "/path", "message": "resolves outside the project through a symlink",
        }])
    return candidate


def _pointer(parent: str, key: object) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _matches_type(value: object, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return ((isinstance(value, int) and not isinstance(value, bool)) or
                (isinstance(value, float) and math.isfinite(value) and value.is_integer()))
    if kind == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and math.isfinite(value))
    raise ValueError(f"Unsupported schema type: {kind}")


def _format_error(value: str, kind: str) -> str | None:
    if kind.startswith("project-"):
        return _path_error(
            value, allow_root=kind in ("project-directory", "project-input"),
            allow_glob=kind in ("project-path", "project-input"),
            file_path=kind == "project-output",
        )
    try:
        if kind == "date":
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError()
            date.fromisoformat(value)
        elif kind == "date-time":
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if "T" not in value or parsed.tzinfo is None:
                raise ValueError()
        else:
            raise RuntimeError(f"Unsupported schema format: {kind}")
    except ValueError:
        return f"must be an ISO {kind}" + (" with a timezone" if kind == "date-time" else "")
    return None


def _validate(value: object, schema: dict, root: dict, path: str, errors: list[dict[str, str]]) -> None:
    def fail(message: str) -> None:
        errors.append({"path": path or "/", "message": message})

    if "$ref" in schema:
        target = root
        for key in schema["$ref"].removeprefix("#/").split("/"):
            target = target[key]
        _validate(value, target, root, path, errors)
        return
    if "anyOf" in schema:
        failures = []
        for choice in schema["anyOf"]:
            branch: list[dict[str, str]] = []
            _validate(value, choice, root, path, branch)
            if not branch:
                return
            failures.append(branch)
        # Prefer detailed field errors from a branch of the right type.
        closest = min(failures, key=lambda items: (
            sum(e["path"] == (path or "/") for e in items), len(items),
        ))
        errors.extend(closest)
        return
    kinds = schema.get("type")
    if kinds:
        accepted = [kinds] if isinstance(kinds, str) else kinds
        if not any(_matches_type(value, kind) for kind in accepted):
            fail(f"must be of type {' or '.join(accepted)}")
            return
    if "enum" in schema:
        matches = any(
            value == option and (type(value) is type(option) or
                                 (_matches_type(value, "number") and _matches_type(option, "number")))
            for option in schema["enum"]
        )
        if not matches:
            fail(f"must be one of {schema['enum']!r}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append({"path": _pointer(path, key), "message": "required field is missing"})
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], subschema, root, _pointer(path, key), errors)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            fail(f"must contain at least {schema['minItems']} item(s)")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(item, schema["items"], root, _pointer(path, index), errors)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            fail(f"must contain at least {schema['minLength']} character(s)")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            fail(f"must contain at most {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            fail(f"must match {schema['pattern']!r}")
        if "format" in schema:
            problem = _format_error(value, schema["format"])
            if problem:
                fail(problem)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            fail("must be a finite number")
        elif "minimum" in schema and value < schema["minimum"]:
            fail(f"must be at least {schema['minimum']}")
        elif "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            fail(f"must be greater than {schema['exclusiveMinimum']}")


def _check_unique(items: list[dict], path: str, errors: list[dict[str, str]]) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        if item["id"] in seen:
            errors.append({"path": f"{path}/{index}/id", "message": f"duplicate id {item['id']!r}"})
        seen.add(item["id"])


def _check_dependencies(tasks: list[dict], errors: list[dict[str, str]]) -> None:
    by_id = {task["id"]: task for task in tasks}
    for index, task in enumerate(tasks):
        deps = task.get("dependsOn", [])
        for dep_index, dependency in enumerate(deps):
            if dependency not in by_id:
                errors.append({"path": f"/production/tasks/{index}/dependsOn/{dep_index}",
                               "message": f"unknown task {dependency!r}"})
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            errors.append({"path": "/production/tasks", "message": f"dependency cycle includes {task_id!r}"})
            return
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].get("dependsOn", []):
            if dependency in by_id:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)


def validate_project(data: object) -> dict:
    """Return a validated copy with schemaVersion=1; never discard extensions.

    Legacy files can omit schemaVersion and optional workflow metadata. Scene
    and visual types stay open for custom templates. Structural validation does
    not require assets to exist or inferred estimates to equal rendered timing.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors: list[dict[str, str]] = []
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProjectValidationError([{"path": "/", "message": f"must contain only finite JSON values: {error}"}]) from error
    _validate(data, schema, schema, "", errors)
    if errors:
        raise ProjectValidationError(errors)
    _check_unique(data["scenes"], "/scenes", errors)
    if "production" in data:
        tasks = data["production"]["tasks"]
        _check_unique(tasks, "/production/tasks", errors)
        _check_dependencies(tasks, errors)
    if errors:
        raise ProjectValidationError(errors)
    result = copy.deepcopy(data)
    result.setdefault("schemaVersion", SCHEMA_VERSION)
    return result


def load_project(path: str | Path) -> dict:
    """Load a UTF-8 JSON manifest; malformed JSON has the same error interface."""
    target = manifest_path(path)
    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProjectValidationError([{"path": "/", "message": f"duplicate JSON property {key!r}"}])
            result[key] = value
        return result
    try:
        data = json.loads(target.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ProjectValidationError([{"path": "/", "message":
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"}]) from error
    return validate_project(data)


def save_project(path: str | Path, data: object) -> Path:
    """Validate first, then replace one manifest atomically on the same volume."""
    target = manifest_path(path)
    normalized = validate_project(data)
    payload = json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
