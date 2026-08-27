#!/usr/bin/env python3
"""
Toolkit-wide version staleness check — the data source for the /versions skill.

Reports three things:
  1. Remotion pins in every package.json under templates/, examples/, showcase/,
     tests/ — versus the latest published `remotion`. Flags caret ranges
     (the toolkit pins exact versions for reproducibility) and any directory
     whose remotion/@remotion/* packages disagree with each other.
  2. Python: whether uv.lock matches pyproject.toml, and which core deps have a
     newer release (via `uv pip list --outdated`, skipped if uv is missing).
  3. Toolkit: _internal/toolkit-registry.json version vs the latest GitHub release.

Usage:
    uv run scripts/check_versions.py            # human-readable table
    uv run scripts/check_versions.py --json     # machine-readable
    uv run scripts/check_versions.py --offline  # skip npm / GitHub / uv network calls

Exit code is 0 unless --strict is given, in which case caret ranges or
internal mismatches exit 1 (staleness alone never fails — bumps are deliberate).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "digitalsamba/claude-code-video-toolkit"
SCAN_DIRS = ("templates", "examples", "showcase", "tests")
REMOTION_RE = re.compile(r"^(remotion|@remotion/.+)$")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "_internal" / "toolkit-registry.json").exists():
            return p
    return Path.cwd()


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ─── Remotion ────────────────────────────────────────────────

def find_package_jsons(root: Path) -> list[Path]:
    out = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for pj in base.rglob("package.json"):
            if "node_modules" in pj.parts:
                continue
            out.append(pj)
    return sorted(out)


def remotion_pins(pj: Path) -> dict[str, str]:
    try:
        data = json.loads(pj.read_text())
    except (OSError, ValueError):
        return {}
    pins = {}
    for section in ("dependencies", "devDependencies"):
        for name, ver in (data.get(section) or {}).items():
            if REMOTION_RE.match(name):
                pins[name] = ver
    return pins


def npm_remotion_info(offline: bool) -> dict:
    """Latest remotion version plus publish dates, from the npm registry."""
    if offline:
        return {}
    raw = run(["npm", "view", "remotion", "version", "time", "--json"], timeout=90)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return {"latest": data.get("version"), "time": data.get("time", {})}


def patches_between(a: str, b: str, time: dict) -> int | None:
    """Count published 4.0.x releases strictly after a up to b (non-prerelease)."""
    def key(v: str):
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", v)
        return tuple(int(x) for x in m.groups()) if m else None
    ka, kb = key(a), key(b)
    if not ka or not kb:
        return None
    return sum(1 for v in time if (k := key(v)) and ka < k <= kb)


def check_remotion(root: Path, offline: bool) -> dict:
    info = npm_remotion_info(offline)
    latest = info.get("latest")
    time = info.get("time", {})
    now = datetime.now(timezone.utc)

    projects = []
    for pj in find_package_jsons(root):
        pins = remotion_pins(pj)
        if not pins:
            continue
        versions = set(pins.values())
        exact = [v for v in versions if re.match(r"^\d+\.\d+\.\d+$", v)]
        caret = [v for v in versions if not re.match(r"^\d+\.\d+\.\d+$", v)]
        pinned = exact[0] if len(exact) == 1 and not caret else None
        entry = {
            "path": str(pj.parent.relative_to(root)),
            "packages": pins,
            "pinned": pinned,
            "caret": bool(caret),
            "mismatch": len(versions) > 1,
        }
        if pinned and latest:
            entry["behind"] = patches_between(pinned, latest, time)
            pub = time.get(pinned)
            if pub:
                try:
                    entry["pinned_age_days"] = (now - datetime.fromisoformat(pub.replace("Z", "+00:00"))).days
                except ValueError:
                    pass
        projects.append(entry)

    pinned_versions = sorted({p["pinned"] for p in projects if p["pinned"]})
    return {
        "latest": latest,
        "projects": projects,
        "pinned_versions": pinned_versions,
        "consistent": len(pinned_versions) <= 1 and not any(p["caret"] or p["mismatch"] for p in projects),
    }


# ─── Python ──────────────────────────────────────────────────

def _direct_python_deps(root: Path) -> set[str]:
    """Package names declared in pyproject.toml (dependencies + all extras)."""
    try:
        text = (root / "pyproject.toml").read_text()
    except OSError:
        return set()
    names = set()
    for m in re.finditer(r'^\s*"([A-Za-z0-9_.-]+)\s*[><=!~\[]', text, flags=re.M):
        names.add(m.group(1).lower().replace("_", "-"))
    return names


def check_python(root: Path, offline: bool) -> dict:
    result: dict = {"uv": bool(run(["uv", "--version"]))}
    if not result["uv"]:
        return result
    lock_ok = subprocess.run(["uv", "lock", "--check"], cwd=root, capture_output=True, text=True)
    result["lock_in_sync"] = lock_ok.returncode == 0
    if offline:
        return result
    raw = run(["uv", "pip", "list", "--outdated", "--format", "json"], cwd=root, timeout=120)
    direct = _direct_python_deps(root)
    outdated = []
    if raw:
        try:
            outdated = [
                {"name": o["name"], "current": o["version"], "latest": o["latest_version"]}
                for o in json.loads(raw)
                if o["name"].lower().replace("_", "-") in direct
            ]
        except (ValueError, KeyError):
            pass
    result["outdated"] = outdated  # direct deps only; transitive drift is uv's job
    return result


# ─── Toolkit ─────────────────────────────────────────────────

def check_toolkit(root: Path, offline: bool) -> dict:
    try:
        current = json.loads((root / "_internal" / "toolkit-registry.json").read_text()).get("version")
    except (OSError, ValueError):
        current = None
    result = {"current": current}
    if offline:
        return result
    raw = run(["gh", "api", f"repos/{REPO}/releases/latest", "--jq", ".tag_name"])
    if raw is None:
        try:
            import urllib.request
            with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/releases/latest", timeout=10) as r:
                raw = json.load(r).get("tag_name", "")
        except Exception:
            raw = None
    if raw:
        result["latest"] = raw.strip().lstrip("v")
        result["up_to_date"] = result["latest"] == current
    return result


# ─── Output ──────────────────────────────────────────────────

def format_human(report: dict) -> str:
    L = []
    r = report["remotion"]
    L.append("Remotion")
    L.append(f"  Latest on npm: {r['latest'] or 'unknown (offline?)'}")
    for p in r["projects"]:
        flag = ""
        if p["caret"]:
            flag = "  ⚠ caret range — pin exactly"
        elif p["mismatch"]:
            flag = "  ⚠ packages disagree — run npx remotion upgrade"
        elif p.get("behind"):
            age = f", pinned {p['pinned_age_days']}d ago" if p.get("pinned_age_days") is not None else ""
            flag = f"  {p['behind']} patches behind{age}"
        ver = p["pinned"] or ", ".join(sorted(set(p["packages"].values())))
        L.append(f"  {p['path']:<42} {ver:<10}{flag}")
    if r["pinned_versions"] and len(r["pinned_versions"]) > 1:
        L.append(f"  ⚠ Toolkit pins disagree across projects: {', '.join(r['pinned_versions'])}")
    L.append("")

    py = report["python"]
    L.append("Python")
    if not py.get("uv"):
        L.append("  uv not installed — skipped")
    else:
        L.append(f"  uv.lock in sync with pyproject.toml: {'yes' if py.get('lock_in_sync') else 'NO — run uv lock'}")
        od = py.get("outdated")
        if od is None:
            L.append("  outdated check skipped (offline)")
        elif not od:
            L.append("  all installed packages current")
        else:
            L.append(f"  {len(od)} outdated: " + ", ".join(f"{o['name']} {o['current']}→{o['latest']}" for o in od[:8]) + (" …" if len(od) > 8 else ""))
    L.append("")

    t = report["toolkit"]
    L.append("Toolkit")
    L.append(f"  Current: v{t.get('current')}   Latest release: " + (f"v{t['latest']}" if t.get("latest") else "unknown"))
    if t.get("latest") and not t.get("up_to_date"):
        L.append("  → git pull origin main")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--offline", action="store_true", help="Skip npm / GitHub / uv network calls")
    ap.add_argument("--strict", action="store_true", help="Exit 1 on caret ranges or internal mismatches")
    args = ap.parse_args()

    root = repo_root()
    report = {
        "remotion": check_remotion(root, args.offline),
        "python": check_python(root, args.offline),
        "toolkit": check_toolkit(root, args.offline),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_human(report))

    if args.strict and not report["remotion"]["consistent"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
