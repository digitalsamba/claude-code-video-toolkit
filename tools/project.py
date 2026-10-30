#!/usr/bin/env python3
"""Validate or migrate project.json without rendering or calling cloud services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from project_manifest import ProjectValidationError, load_project, manifest_path, save_project
from win_encoding import ensure_utf8_output


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "migrate"):
        sub = commands.add_parser(command, help=f"{command.capitalize()} a project manifest")
        sub.add_argument("project", type=Path, help="Project directory or project.json file")
        sub.add_argument("--json", action="store_true", help="Machine-readable result on stdout")
        if command == "migrate":
            sub.add_argument("--write", action="store_true", help="Atomically save schemaVersion; default is read-only")
    args = parser.parse_args(argv)
    target = manifest_path(args.project)
    try:
        project = load_project(target)
        written = args.command == "migrate" and args.write
        if written:
            save_project(target, project)
        result = {"valid": True, "path": str(target), "schemaVersion": project["schemaVersion"],
                  "scenes": len(project["scenes"]), "written": written}
    except (OSError, UnicodeError, ProjectValidationError) as error:
        errors = error.errors if isinstance(error, ProjectValidationError) else [{"path": "/", "message": str(error)}]
        result = {"valid": False, "path": str(target), "errors": errors}
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            for issue in errors:
                print(f"{target}{issue['path']}: {issue['message']}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Valid project: {target} ({result['scenes']} scenes, schema v{result['schemaVersion']})")
        if args.command == "migrate":
            print("Saved atomically." if written else "Dry run. Add --write to save the version field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
