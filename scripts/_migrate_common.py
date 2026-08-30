from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandSpec:
    name: str
    path: Path
    description: str


@dataclass
class SkillSpec:
    name: str
    path: Path
    description: str


def parse_skill_frontmatter(skill_md: Path) -> tuple[str, str]:
    content = skill_md.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        return skill_md.parent.name, ""

    name = skill_md.parent.name
    description = ""
    for line in lines[1:]:
        if line == "---":
            break
        if line.startswith("name:"):
            name = line[5:].strip().strip("\"'")
        elif line.startswith("description:"):
            description = line[12:].strip().strip("\"'")
    return name, description


def ensure_clean_dir(path: Path, force: bool, dry_run: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"{path} already exists. Use --force to overwrite.")
        if not dry_run:
            shutil.rmtree(path)
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dest: Path, force: bool, dry_run: bool) -> None:
    ensure_clean_dir(dest, force=force, dry_run=dry_run)
    if not dry_run:
        shutil.copytree(src, dest, dirs_exist_ok=True)


def remove_dir(path: Path, dry_run: bool) -> bool:
    if path.exists() and path.is_dir():
        if not dry_run:
            shutil.rmtree(path)
        return True
    return False


def yaml_quote(value: str) -> str:
    if not value:
        return '""'
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def contains_generated_block(text: str, begin_marker: str, end_marker: str) -> bool:
    return begin_marker in text and end_marker in text


def replace_generated_block(text: str, new_block: str, begin_marker: str, end_marker: str) -> str:
    start = text.index(begin_marker)
    end = text.index(end_marker) + len(end_marker)
    before = text[:start].rstrip()
    after = text[end:].lstrip()

    parts: list[str] = []
    if before:
        parts.append(before)
    parts.append(new_block)
    if after:
        parts.append(after)
    return "\n\n".join(parts) + "\n"


def remove_generated_block(text: str, begin_marker: str, end_marker: str) -> str:
    start = text.index(begin_marker)
    end = text.index(end_marker) + len(end_marker)
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and after:
        return f"{before}\n\n{after}\n"
    if before:
        return f"{before}\n"
    if after:
        return f"{after}\n"
    return ""

def find_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()

    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "_internal" / "toolkit-registry.json").exists() and (
            candidate / ".claude"
        ).exists():
            return candidate

    raise SystemExit("Error: Could not find toolkit root (must contain .claude/ and _internal/toolkit-registry.json)")

def load_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
