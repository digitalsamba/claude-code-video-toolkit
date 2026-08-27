from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    path: Path

@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    path: Path

def find_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()

    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "_internal" / "toolkit-registry.json").exists() and (
            candidate / ".claude"
        ).exists():
            return candidate

    raise SystemExit("Could not auto-detect repository root.")

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_mapping(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        raise SystemExit(f"Mapping file not found: {path}")
    data = load_json(path)
    return {
        "skip_commands": set(data.get("skip_commands", [])),
        "skip_skills": set(data.get("skip_skills", [])),
        "command_name_overrides": data.get("command_name_overrides", {}),
        "skill_name_overrides": data.get("skill_name_overrides", {}),
    }

def load_registry(repo_root: Path) -> dict[str, Any]:
    return load_json(repo_root / "_internal" / "toolkit-registry.json")

def load_command_specs(
    repo_root: Path, registry: dict[str, Any], mapping: dict[str, Any]
) -> list[CommandSpec]:
    commands: list[CommandSpec] = []
    entries = registry.get("commands", {})

    for original_name, entry in sorted(entries.items()):
        if original_name in mapping["skip_commands"]:
            continue

        command_name = mapping["command_name_overrides"].get(original_name, original_name)
        relative_path = entry.get("path")
        if not relative_path:
            continue

        command_path = repo_root / relative_path
        commands.append(
            CommandSpec(
                name=command_name,
                description=entry.get("description", f"wrapper for /{original_name}"),
                path=command_path,
            )
        )

    return commands

def parse_skill_frontmatter(skill_md: Path) -> tuple[str, str]:
    text = skill_md.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise SystemExit(f"Skill frontmatter missing in {skill_md}")

    name = ""
    description = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip()
        if stripped.startswith("description:"):
            description = stripped.split(":", 1)[1].strip()

    if not name or not description:
        raise SystemExit(f"Skill name/description missing in {skill_md}")
    return name, description

def load_skill_specs(
    repo_root: Path, mapping: dict[str, Any]
) -> list[SkillSpec]:
    results: list[SkillSpec] = []
    for skill_md in sorted((repo_root / ".claude" / "skills").glob("*/SKILL.md")):
        source_name, description = parse_skill_frontmatter(skill_md)
        if source_name in mapping["skip_skills"]:
            continue

        skill_name = mapping["skill_name_overrides"].get(source_name, source_name)
        results.append(
            SkillSpec(
                name=skill_name,
                description=description,
                path=skill_md.parent,
            )
        )
    return results

def ensure_clean_dir(path: Path, force: bool, dry_run: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(
                f"Destination already exists: {path}. Re-run with --force to overwrite."
            )
        if dry_run:
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def copy_tree(src: Path, dest: Path, force: bool, dry_run: bool) -> None:
    ensure_clean_dir(dest, force=force, dry_run=dry_run)
    if dry_run:
        return
    shutil.copytree(src, dest)

def remove_dir(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True
