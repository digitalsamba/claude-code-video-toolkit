#!/usr/bin/env python3
"""Install Kiro CLI-compatible skills and steering from claude-code-video-toolkit.

Kiro CLI sibling of scripts/migrate_to_codex.py:

1. Copies each toolkit skill from `.claude/skills/` into `~/.kiro/skills/`
   (Kiro uses the same SKILL.md frontmatter format, so skills copy verbatim).
2. Generates a thin wrapper skill per Claude Code slash command in
   `.claude/commands/` — Kiro invokes skills as `/name` slash commands with
   the same `$ARGUMENTS` placeholder, and each wrapper points back at the
   original command file (by absolute path) as the source of truth.
3. Generates `.kiro/steering/video-toolkit.md` from `CLAUDE.md` inside a
   managed marker block (Kiro steering files are always-loaded context,
   the equivalent of CLAUDE.md in Claude Code).

Skills install to `~/.kiro/skills` by default — the direct analog of the
Codex script's `~/.codex/skills`. Wrappers reference the toolkit by
absolute path, which gives parity with Claude Code, where the toolkit's
commands work from any subdirectory of the repo (Kiro resolves workspace
`.kiro/` from the current directory only and does not walk up the tree the
way Claude Code does). Use `--workspace-skills` to install into the repo's
`.kiro/skills/` instead; that variant only loads when kiro-cli starts at
the toolkit root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _migrate_common import (
    CommandSpec,
    SkillSpec,
    find_repo_root,
    load_mapping,
    parse_skill_frontmatter,
    ensure_clean_dir,
    copy_tree,
    remove_dir,
    yaml_quote,
    contains_generated_block,
    replace_generated_block,
    remove_generated_block,
)

GENERATED_STEERING_BEGIN = "<!-- BEGIN GENERATED: claude-to-kiro -->"
GENERATED_STEERING_END = "<!-- END GENERATED: claude-to-kiro -->"
STEERING_FILE_NAME = "video-toolkit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Kiro CLI-compatible skills from claude-code-video-toolkit."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Toolkit repository root. Auto-detected by default.",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=None,
        help="Optional migration map override file.",
    )
    parser.add_argument(
        "--workspace-skills",
        action="store_true",
        help="Install skills to the workspace .kiro/skills instead of ~/.kiro/skills. "
        "Workspace skills only load when kiro-cli starts at the toolkit root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated/copied skills if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove previously installed toolkit skills and the generated steering file.",
    )
    return parser.parse_args()


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
    data: dict[str, Any] = {}
    if path is not None:
        if not path.exists():
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
                description=entry.get("description", f"Kiro wrapper for /{original_name}"),
                path=command_path,
            )
        )
    return commands


def load_skill_specs(repo_root: Path, mapping: dict[str, Any]) -> list[SkillSpec]:
    # Only `.claude/skills/` is scanned. Platform-specific skills under top-level
    # `skills/` (e.g. OpenClaw) are intentionally not migrated to Kiro.
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


    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)


    if dry_run:
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def build_steering_block(repo_root: Path) -> str:
    claude_md = (repo_root / "CLAUDE.md").read_text(encoding="utf-8").strip()
    return (
        f"{GENERATED_STEERING_BEGIN}\n"
        "## Kiro Migration Note\n\n"
        "This file is generated by `scripts/migrate_to_kiro.py` from `CLAUDE.md`.\n"
        "Re-run `uv run scripts/migrate_to_kiro.py --force` after updating `CLAUDE.md`.\n\n"
        f"{claude_md}\n"
        f"{GENERATED_STEERING_END}"
    )


def sync_steering_file(repo_root: Path, force: bool, dry_run: bool) -> None:
    steering_path = repo_root / ".kiro" / "steering" / STEERING_FILE_NAME
    new_block = build_steering_block(repo_root)
    existing = steering_path.read_text(encoding="utf-8") if steering_path.exists() else ""

    if steering_path.exists() and contains_generated_block(existing, GENERATED_STEERING_BEGIN, GENERATED_STEERING_END):
        new_content = replace_generated_block(existing, new_block, GENERATED_STEERING_BEGIN, GENERATED_STEERING_END)
        if existing == new_content:
            print("steering_status=up_to_date")
            return
    elif steering_path.exists():
        if not force:
            raise SystemExit(
                f"{steering_path} exists and has no generated Kiro block. "
                "Re-run with --force to append a CLAUDE.md-derived block."
            )
        base = existing.rstrip()
        new_content = f"{base}\n\n{new_block}\n" if base else f"{new_block}\n"
    else:
        new_content = f"{new_block}\n"

    print(f"steering_sync={steering_path}")
    if dry_run:
        return
    steering_path.parent.mkdir(parents=True, exist_ok=True)
    steering_path.write_text(new_content, encoding="utf-8")


def command_wrapper_content(
    repo_root: Path,
    command: CommandSpec,
    installed_skill_names: list[str],
    toolkit_name: str,
) -> str:
    related = ", ".join(f"`{name}`" for name in installed_skill_names) or "(none)"
    source_abs = command.path.resolve().as_posix()
    root_abs = repo_root.resolve().as_posix()

    return f"""---
name: {command.name}
description: Kiro wrapper for the Claude Code `/{command.name}` command in `{toolkit_name}` — use when the user wants {command.description}
---

# /{command.name} for Kiro CLI

This skill is the Kiro CLI entrypoint equivalent of the Claude Code slash command `/{command.name}`.

## User Request

$ARGUMENTS

## Toolkit Location

The toolkit lives at `{root_abs}`. Run all toolkit commands and scripts
with that directory as the working directory, regardless of where this
session was started.

## Source of Truth

Before acting, read the original workflow document:

`{source_abs}`

Also use these files when relevant:

1. `{root_abs}/CLAUDE.md`
2. `{root_abs}/_internal/toolkit-registry.json`
3. `{root_abs}/docs/getting-started.md`

## Operating Rules

1. Treat `{source_abs}` as the authoritative workflow.
2. Do not rewrite or replace the original Claude resources just to satisfy Kiro.
3. Reuse installed toolkit skills when they help the task.
4. Where the original workflow says to restart Claude Code, restart Kiro CLI instead.

## Related Toolkit Skills

{related}
"""


def install_copied_skills(
    repo_root: Path,
    dest_root: Path,
    skills: list[SkillSpec],
    force: bool,
    dry_run: bool,
) -> list[Path]:
    installed: list[Path] = []
    for skill in skills:
        target = dest_root / skill.name
        copy_tree(skill.path, target, force=force, dry_run=dry_run)
        installed.append(target)
    return installed


def install_command_wrappers(
    repo_root: Path,
    dest_root: Path,
    commands: list[CommandSpec],
    installed_skill_names: list[str],
    toolkit_name: str,
    force: bool,
    dry_run: bool,
) -> list[Path]:
    installed: list[Path] = []
    for command in commands:
        target = dest_root / command.name
        ensure_clean_dir(target, force=force, dry_run=dry_run)
        write_text(
            target / "SKILL.md",
            command_wrapper_content(repo_root, command, installed_skill_names, toolkit_name),
            dry_run=dry_run,
        )
        installed.append(target)
    return installed


def print_plan(
    repo_root: Path,
    dest_root: Path,
    skills: list[SkillSpec],
    commands: list[CommandSpec],
) -> None:
    print(f"repo_root={repo_root}")
    print(f"dest={dest_root}")
    print(
        f"steering={repo_root / '.kiro' / 'steering' / STEERING_FILE_NAME}"
        " <- generated from CLAUDE.md"
    )
    print(f"copy_skills={len(skills)}")
    for skill in skills:
        print(f"  skill:{skill.name} <- {skill.path.relative_to(repo_root)}")
    print(f"generate_command_wrappers={len(commands)}")
    for command in commands:
        print(f"  command:{command.name} <- {command.path.relative_to(repo_root)}")


def reset_installed(
    repo_root: Path,
    dest_root: Path,
    skills: list[SkillSpec],
    commands: list[CommandSpec],
    dry_run: bool,
) -> int:
    removable_names = {skill.name for skill in skills}
    removable_names.update(command.name for command in commands)

    print(f"reset_dest={dest_root}")
    removed_any = False
    for name in sorted(removable_names):
        target = dest_root / name
        removed = remove_dir(target, dry_run=dry_run)
        status = "would_remove" if dry_run else "removed"
        if removed:
            removed_any = True
            print(f"  {status}:{name}")
        else:
            print(f"  missing:{name}")

    if not removed_any:
        print("nothing_to_remove")

    steering_path = repo_root / ".kiro" / "steering" / STEERING_FILE_NAME
    if steering_path.exists():
        existing = steering_path.read_text(encoding="utf-8")
        if contains_generated_block(existing, GENERATED_STEERING_BEGIN, GENERATED_STEERING_END):
            new_content = remove_generated_block(existing, GENERATED_STEERING_BEGIN, GENERATED_STEERING_END)
            print(f"steering_reset={steering_path}")
            if not dry_run:
                if new_content:
                    steering_path.write_text(new_content, encoding="utf-8")
                else:
                    steering_path.unlink()
        else:
            print("steering_status=no_generated_block")
    else:
        print("steering_status=missing")
    return 0


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root)
    default_map = repo_root / "kiro" / "migration_map.json"
    map_file = args.map_file or (default_map if default_map.exists() else None)
    mapping = load_mapping(map_file)
    registry = load_registry(repo_root)
    toolkit_name = registry.get("name") or repo_root.name
    commands = load_command_specs(repo_root, registry, mapping)
    skills = load_skill_specs(repo_root, mapping)
    if args.workspace_skills:
        dest_root = repo_root / ".kiro" / "skills"
    else:
        dest_root = (Path.home() / ".kiro" / "skills").expanduser().resolve()

    if args.force and args.reset:
        raise SystemExit("--force and --reset cannot be used together.")

    if args.reset:
        return reset_installed(
            repo_root=repo_root,
            dest_root=dest_root,
            skills=skills,
            commands=commands,
            dry_run=args.dry_run,
        )

    print_plan(repo_root, dest_root, skills, commands)
    if args.dry_run:
        return 0

    sync_steering_file(repo_root=repo_root, force=args.force, dry_run=False)
    dest_root.mkdir(parents=True, exist_ok=True)
    install_copied_skills(
        repo_root=repo_root,
        dest_root=dest_root,
        skills=skills,
        force=args.force,
        dry_run=False,
    )
    install_command_wrappers(
        repo_root=repo_root,
        dest_root=dest_root,
        commands=commands,
        installed_skill_names=[skill.name for skill in skills],
        toolkit_name=toolkit_name,
        force=args.force,
        dry_run=False,
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
