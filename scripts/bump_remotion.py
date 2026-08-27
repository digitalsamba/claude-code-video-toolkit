#!/usr/bin/env python3
"""
Bump the toolkit's pinned Remotion version — the mechanical half of
`/versions --bump-toolkit`.

For every package.json under templates/, examples/, showcase/, tests/ that
pins `remotion` / `@remotion/*`:
  1. rewrite the pins to <version> (exact, never a caret)
  2. refresh the lockfile (`npm install --package-lock-only`)
  3. optionally smoke-render frame 0 of the project's first composition
     (`--smoke`; requires a full `npm install`, so it is slower and only runs
     for Remotion projects with an entry point)
Then, unless --no-baseline, run the render-baseline A/B (before = current
pin, after = <version>) via scripts/render-baseline.mjs and include its
report in the summary.

Writes a markdown summary to stdout (or --summary <file>) that the skill
pastes into the bump PR body.

Usage:
    uv run scripts/bump_remotion.py 4.0.518                # pins + lockfiles + baseline A/B
    uv run scripts/bump_remotion.py 4.0.518 --smoke        # also smoke-render each project
    uv run scripts/bump_remotion.py 4.0.518 --only templates/sprint-review-v2
    uv run scripts/bump_remotion.py 4.0.518 --dry-run      # report what would change
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCAN_DIRS = ("templates", "examples", "showcase", "tests")
REMOTION_RE = re.compile(r"^(remotion|@remotion/.+)$")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "_internal" / "toolkit-registry.json").exists():
            return p
    return Path.cwd()


def find_projects(root: Path, only: list[str]) -> list[Path]:
    out = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for pj in base.rglob("package.json"):
            if "node_modules" in pj.parts:
                continue
            if only and not any(str(pj.parent.relative_to(root)).startswith(o.rstrip("/")) for o in only):
                continue
            out.append(pj)
    return sorted(out)


def current_pins(pj: Path) -> dict[str, str]:
    data = json.loads(pj.read_text())
    pins = {}
    for section in ("dependencies", "devDependencies"):
        for name, ver in (data.get(section) or {}).items():
            if REMOTION_RE.match(name):
                pins[name] = ver
    return pins


def rewrite_pins(pj: Path, version: str) -> int:
    """Rewrite in place with a regex so formatting/key order is preserved."""
    text = pj.read_text()
    new, n = re.subn(
        r'("(?:remotion|@remotion/[^"]+)"\s*:\s*")[^"]+(")',
        lambda m: f"{m.group(1)}{version}{m.group(2)}",
        text,
    )
    if n:
        pj.write_text(new)
    return n


def npm(args: list[str], cwd: Path, timeout: int = 600) -> tuple[bool, str]:
    cmd = ["npm", *args, "--no-audit", "--no-fund", "--loglevel=error"]
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           shell=(sys.platform == "win32"))
        return r.returncode == 0, (r.stderr or r.stdout).strip()[-800:]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)


def first_composition(project: Path) -> tuple[str, str] | None:
    entry = project / "src" / "index.ts"
    root_tsx = project / "src" / "Root.tsx"
    if not entry.exists() or not root_tsx.exists():
        return None
    m = re.search(r'id\s*=\s*["\']([^"\']+)["\']', root_tsx.read_text())
    return ("src/index.ts", m.group(1)) if m else None


def smoke_render(project: Path) -> tuple[str, str]:
    """Render frame 0 of the first composition at tiny scale. Returns (status, detail)."""
    comp = first_composition(project)
    if not comp:
        return "skipped", "no Remotion entry point"
    ok, msg = npm(["install"], project)
    if not ok:
        return "FAILED", f"npm install: {msg}"
    out = project / "out" / "smoke-frame0.png"
    out.parent.mkdir(exist_ok=True)
    cmd = ["npx", "remotion", "still", comp[0], comp[1], str(out),
           "--frame=0", "--scale=0.1", "--image-format=png", "--gl=swangle", "--log=error"]
    try:
        r = subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=900,
                           shell=(sys.platform == "win32"))
    except subprocess.TimeoutExpired:
        return "FAILED", "render timed out"
    if r.returncode != 0:
        return "FAILED", (r.stderr or r.stdout).strip()[-800:]
    out.unlink(missing_ok=True)
    return "ok", f"{comp[1]} frame 0 rendered"


def run_baseline(root: Path, version: str) -> tuple[bool | None, str]:
    script = root / "scripts" / "render-baseline.mjs"
    project = root / "tests" / "render-baseline"
    if not script.exists() or not project.exists():
        return None, "render-baseline harness not present"
    if not (project / "node_modules").exists():
        ok, msg = npm(["install"], project)
        if not ok:
            return False, f"npm install for harness failed: {msg}"
    r = subprocess.run(["node", str(script), "ab", version], cwd=root, capture_output=True, text=True,
                       shell=(sys.platform == "win32"))
    report = project / "out" / "diff-before-vs-after" / "report.md"
    text = report.read_text() if report.exists() else (r.stderr or r.stdout)[-1500:]
    return r.returncode == 0, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="Target Remotion version, e.g. 4.0.518")
    ap.add_argument("--only", action="append", default=[], help="Limit to project dirs starting with this path (repeatable)")
    ap.add_argument("--smoke", action="store_true", help="Full npm install + render frame 0 per Remotion project")
    ap.add_argument("--no-baseline", action="store_true", help="Skip the render-baseline A/B")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without touching files")
    ap.add_argument("--summary", help="Write the markdown summary to this file as well as stdout")
    args = ap.parse_args()

    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        print(f"error: version must be exact (got {args.version!r}); the toolkit never pins ranges", file=sys.stderr)
        return 2
    if not shutil.which("npm"):
        print("error: npm not found", file=sys.stderr)
        return 2

    root = repo_root()
    projects = find_projects(root, args.only)
    rows = []
    before_versions = set()
    failed = False

    # The baseline harness manages its own pin during the A/B; exclude it from the walk
    # unless the user targeted it explicitly, and bump it last (below).
    harness = root / "tests" / "render-baseline" / "package.json"

    for pj in projects:
        pins = current_pins(pj)
        if not pins:
            continue
        rel = str(pj.parent.relative_to(root))
        old = sorted(set(pins.values()))
        before_versions.update(old)
        row = {"project": rel, "from": ", ".join(old), "to": args.version, "lock": "—", "smoke": "—"}
        if args.dry_run:
            row["lock"] = "would refresh"
            rows.append(row)
            continue
        if pj == harness and not args.no_baseline:
            row["lock"] = "after A/B"
            rows.append(row)
            continue
        rewrite_pins(pj, args.version)
        ok, msg = npm(["install", "--package-lock-only", "--ignore-scripts"], pj.parent)
        row["lock"] = "refreshed" if ok else f"FAILED: {msg}"
        failed |= not ok
        if args.smoke and ok:
            status, detail = smoke_render(pj.parent)
            row["smoke"] = f"{status} — {detail}"
            failed |= status == "FAILED"
        rows.append(row)

    baseline_ok, baseline_text = (None, "skipped")
    if not args.dry_run and not args.no_baseline:
        baseline_ok, baseline_text = run_baseline(root, args.version)
        # `ab` restores the harness pin; now move it forward like everything else.
        if harness.exists():
            rewrite_pins(harness, args.version)
            ok, msg = npm(["install", "--package-lock-only", "--ignore-scripts"], harness.parent)
            for r in rows:
                if r["project"] == "tests/render-baseline":
                    r["lock"] = "refreshed" if ok else f"FAILED: {msg}"
            failed |= not ok
        failed |= baseline_ok is False

    # ─── summary ────────────────────────────────────────────
    frm = ", ".join(sorted(before_versions)) or "?"
    L = [f"## Remotion bump {frm} → {args.version}" + (" (dry run)" if args.dry_run else ""), ""]
    L += ["| Project | From | Lockfile | Smoke render |", "|---|---|---|---|"]
    L += [f"| `{r['project']}` | {r['from']} | {r['lock']} | {r['smoke']} |" for r in rows]
    L += [""]
    if baseline_ok is None:
        L += [f"**Render baseline:** {baseline_text}"]
    else:
        L += ["**Render baseline (tests/render-baseline, A/B on this machine):**", "", baseline_text.strip()]
    L += ["", "Next: review, `git diff --stat`, then open the PR — CI re-runs the A/B on Linux and comments the table."]
    summary = "\n".join(L) + "\n"
    print(summary)
    if args.summary:
        Path(args.summary).write_text(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
