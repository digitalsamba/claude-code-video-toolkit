#!/usr/bin/env python3
"""Deploy the toolkit's Modal apps and write their endpoint URLs into .env.

One command replaces the manual loop of `modal deploy`, copying a URL out of a
wrapped deploy log, and pasting it into .env. The URL comes from Modal's API
(`Cls.from_name(...).<method>.get_web_url()`), not from the deploy output, so
line-wrapping and truncated labels can't garble it.

Usage:
    uv run tools/modal_setup.py --list                 # what's deployed vs. what .env says
    uv run tools/modal_setup.py qwen3_tts flux2        # deploy these, save their URLs
    uv run tools/modal_setup.py --all                  # every app except ltx2
    uv run tools/modal_setup.py --sync-env             # no deploy: fix .env from what's live
    uv run tools/modal_setup.py ltx2 --hf-secret huggingface-secret
    FLUX2_MODEL=klein-9b uv run tools/modal_setup.py flux2 --recreate

Requires the Modal SDK in the toolkit env and an authenticated CLI:
    uv sync --extra modal && uv run modal setup

Deploys run one at a time (Modal rate-limits AppCreate). soulx's weights volume
is populated before its deploy; the step is idempotent, so a re-run is quick.

Apps that pull gated Hugging Face weights (ltx2 always, flux2 with
FLUX2_MODEL=klein-9b) read the Modal secret named by HF_SECRET_NAME, default
`huggingface-token`. Modal's Hugging Face template names it `huggingface-secret`;
when only that one exists, it is picked up automatically.

A new Modal workspace (seen 2026-11-10) builds images with builder version 2026.10,
which the Modal client in uv.lock refuses before building anything. When a
deploy fails that way and MODAL_IMAGE_BUILDER_VERSION isn't set, it is retried with
2025.06, and the rest of the run uses it too. A workspace whose version the client
accepts keeps it, so its cached images aren't rebuilt.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cloud_gpu import _MODAL_ENV_VARS, _log
from config import find_workspace_root
from win_encoding import ensure_utf8_output, utf8_env

ROOT = find_workspace_root()
DEFAULT_HF_SECRET = "huggingface-token"
# What Modal's "Hugging Face" secret template fills in as the name.
TEMPLATE_HF_SECRET = "huggingface-secret"

# Image builder version for a workspace whose default the Modal client refuses.
# qwen3_tts, flux2 and music_gen were deployed on 2025.06 from a new workspace
# (2026-11-10). A client that accepts 2026.10 (modal 1.6 does) never falls back, and
# 2026.10 drops Python 3.10, which sadtalker and dewatermark build on: bump modal in
# uv.lock only together with a pin here or a move of those two apps to 3.11.
BUILDER_VAR = "MODAL_IMAGE_BUILDER_VERSION"
FALLBACK_BUILDER_VERSION = "2025.06"
# The start of the client's error when it refuses the workspace's builder version.
_BUILDER_REFUSED = "supports the following image builder versions"


@dataclass(frozen=True)
class ModalApp:
    tool: str
    app_dir: str
    app_name: str
    cls: str
    method: str
    label: str
    # Deploy-time variable selecting a model variant, if the app has one.
    variant_var: str | None = None
    # Function run with `modal run` before the deploy (soulx's weights volume).
    prepare: str | None = None
    # Left out of --all; must be named explicitly.
    explicit_only: bool = False

    @property
    def app_file(self) -> Path:
        return ROOT / "docker" / self.app_dir / "app.py"

    @property
    def env_var(self) -> str:
        return _MODAL_ENV_VARS[self.tool]

    def needs_hf_secret(self, env: dict) -> bool:
        if self.tool == "ltx2":
            return True
        if self.tool == "flux2":
            return env.get("FLUX2_MODEL", "klein-4b") == "klein-9b"
        return False


APPS = {a.tool: a for a in [
    ModalApp("qwen3_tts", "modal-qwen3-tts", "video-toolkit-qwen3-tts", "Qwen3TTS", "generate",
             "Speech (Qwen3-TTS)"),
    ModalApp("flux2", "modal-flux2", "video-toolkit-flux2", "Flux2", "generate",
             "Images (FLUX.2)", variant_var="FLUX2_MODEL"),
    ModalApp("image_edit", "modal-image-edit", "video-toolkit-image-edit", "ImageEditor", "edit",
             "Image editing (Qwen-Image)", variant_var="IMAGE_EDIT_MODEL"),
    ModalApp("upscale", "modal-upscale", "video-toolkit-upscale", "Upscaler", "upscale",
             "Upscaling (RealESRGAN)"),
    ModalApp("music_gen", "modal-music-gen", "video-toolkit-music-gen", "MusicGen", "generate",
             "Music (ACE-Step)"),
    ModalApp("sadtalker", "modal-sadtalker", "video-toolkit-sadtalker", "SadTalkerGen", "generate",
             "Talking heads (SadTalker)"),
    ModalApp("soulx", "modal-soulx", "video-toolkit-soulx", "SoulXFlashHead", "generate_web",
             "Talking heads (SoulX-FlashHead)", prepare="populate_weights"),
    ModalApp("dewatermark", "modal-propainter", "video-toolkit-dewatermark", "Dewatermark",
             "dewatermark", "Watermark removal (ProPainter)"),
    # ~80GB image, A100-80GB, and gated Gemma 3 access approved by hand -- not
    # something to start as a side effect of --all.
    ModalApp("ltx2", "modal-ltx2", "video-toolkit-ltx2", "LTX2", "generate",
             "Video (LTX-2)", variant_var="LTX2_VERSION", explicit_only=True),
]}

# Substrings of a failed deploy's output (any one, case-insensitive) -> what to do.
_HF_ACCESS = ("Hugging Face refused the token. Accept the gated repo's terms with the account "
              "behind the Modal secret: ltx2 needs "
              "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized (approved by "
              "hand) or Lightricks/LTX-2.5 with LTX2_VERSION=2.5; flux2 klein-9b needs "
              "black-forest-labs/FLUX.2-klein-9B.")
DEPLOY_HINTS = [
    ((_BUILDER_REFUSED,),
     f"The Modal client refused the image builder version. Set {BUILDER_VAR} to one it "
     f"lists ({FALLBACK_BUILDER_VERSION} is tested), in the shell or in .env, or remove "
     "image_builder_version from ~/.modal.toml, then re-run."),
    (("payment method",),
     "Add a payment method at https://modal.com/settings/billing. The Starter plan's "
     "$30/month of included compute still covers typical use, and the image is already "
     "cached, so the re-deploy is fast."),
    (("rate limit",),
     "Modal rate-limited app creation. Wait a minute and re-run."),
    (("gatedrepoerror", "401 client error", "403 client error", "cannot access gated repo"),
     _HF_ACCESS),
    (("secret '",),
     "Check the Hugging Face secret: --hf-secret NAME, or create one with "
     f"`uv run modal secret create {DEFAULT_HF_SECRET} HF_TOKEN=hf_...`"),
    (("token missing", "not authenticated", "unauthenticated"),
     "The Modal CLI isn't authenticated. Run: uv run modal setup"),
]


# ---------------------------------------------------------------------------
# Modal access
# ---------------------------------------------------------------------------

def _modal_cmd(*args: str) -> list[str]:
    # The SDK's own CLI, so the deploy and the URL lookup use one Modal version.
    return [sys.executable, "-m", "modal", *args]


def _require_sdk():
    try:
        import modal  # noqa: F401
    except ImportError:
        _log("Modal SDK not installed in the toolkit env. Run:", "error")
        _log("  uv sync --extra modal && uv run modal setup", "error")
        sys.exit(1)


def lookup_url(app: ModalApp) -> tuple[str | None, str | None]:
    """(url, None) for a deployed app, (None, reason) otherwise."""
    import modal

    try:
        obj = modal.Cls.from_name(app.app_name, app.cls)()
        return getattr(obj, app.method).get_web_url(), None
    except modal.exception.NotFoundError:
        return None, "not deployed"
    except Exception as e:  # auth, network
        return None, f"{type(e).__name__}: {e}"


def list_secrets() -> list[str] | None:
    try:
        r = subprocess.run(_modal_cmd("secret", "list", "--json"), capture_output=True,
                           encoding="utf-8", errors="replace", env=utf8_env(), timeout=60)
        return [s["name"] for s in json.loads(r.stdout)] if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def resolve_hf_secret(requested: str | None) -> tuple[str | None, str | None]:
    """(secret name, None), or (None, error) when no usable secret exists."""
    names = list_secrets()
    if names is None:
        # Can't tell; let the deploy itself report a missing secret.
        return requested or DEFAULT_HF_SECRET, None
    if requested:
        if requested in names:
            return requested, None
        return None, f"Modal secret '{requested}' not found (have: {', '.join(names) or 'none'})"
    if DEFAULT_HF_SECRET in names:
        return DEFAULT_HF_SECRET, None
    if TEMPLATE_HF_SECRET in names:
        _log(f"Using Modal secret '{TEMPLATE_HF_SECRET}' (Modal's template name) "
             f"in place of '{DEFAULT_HF_SECRET}'", "warn")
        return TEMPLATE_HF_SECRET, None
    return None, (
        f"No Hugging Face secret on Modal. Create one with key HF_TOKEN:\n"
        f"  uv run modal secret create {DEFAULT_HF_SECRET} HF_TOKEN=hf_...\n"
        f"or name an existing one with --hf-secret NAME"
    )


def _run_streamed(cmd: list[str], env: dict) -> tuple[int, str]:
    """Run cmd, echoing its output to stderr; return (code, tail of output)."""
    tail: collections.deque[str] = collections.deque(maxlen=200)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", env=utf8_env(env), cwd=ROOT)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line)
    return proc.wait(), "".join(tail)


def _flat(output: str) -> str:
    """Output as one lowercase line of words.

    The Modal CLI prints an error inside a rich panel wrapped at the console width
    (80 columns when piped), so a phrase can be split across lines with a border
    character on each side. Box-drawing characters and `|` are dropped first.
    """
    return " ".join(re.sub(r"[─-╿|]", " ", output).split()).lower()


def _hint(output: str) -> str | None:
    low = _flat(output)
    for needles, hint in DEPLOY_HINTS:
        if any(n in low for n in needles):
            return hint
    return None


def _builder_fallback(output: str, env: dict) -> bool:
    """True when the client refused the workspace's default builder version.

    Not when the version came from MODAL_IMAGE_BUILDER_VERSION or ~/.modal.toml (the
    error then says "based on your ..."): that choice is the user's, so the hint
    reports it instead.
    """
    flat = _flat(output)
    return _BUILDER_REFUSED in flat and BUILDER_VAR not in env and "based on your" not in flat


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

def env_path() -> Path:
    return ROOT / ".env"


def read_env_value(key: str) -> str | None:
    path = env_path()
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def write_env_value(key: str, value: str) -> None:
    """Replace key's line in .env, or append it. Everything else is left alone."""
    path = env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def deploy(app: ModalApp, hf_secret: str | None = None, recreate: bool = False,
           dry_run: bool = False) -> dict:
    """Deploy one app and save its URL to .env. Returns a result dict."""
    result = {"tool": app.tool, "env_var": app.env_var, "success": False,
              "endpoint_url": None, "error": None}
    env = dict(os.environ)
    if app.variant_var and env.get(app.variant_var):
        result["variant"] = f"{app.variant_var}={env[app.variant_var]}"
    if env.get(BUILDER_VAR):
        result["image_builder_version"] = env[BUILDER_VAR]

    if app.needs_hf_secret(env):
        secret, err = resolve_hf_secret(hf_secret or env.get("HF_SECRET_NAME"))
        if err:
            result["error"] = err
            return result
        if secret and secret != DEFAULT_HF_SECRET:
            env["HF_SECRET_NAME"] = secret
            result["hf_secret"] = secret

    steps = []
    if app.prepare:
        steps.append(_modal_cmd("run", f"{app.app_file.relative_to(ROOT)}::{app.prepare}"))
    strategy = ["--strategy", "recreate"] if recreate else []
    steps.append(_modal_cmd("deploy", *strategy, str(app.app_file.relative_to(ROOT))))

    if dry_run:
        prefix = "".join(f"{k}={env[k]} " for k in ("HF_SECRET_NAME", BUILDER_VAR) if k in env)
        result["commands"] = [prefix + " ".join(["modal", *c[3:]]) for c in steps]
        result["success"] = True
        return result

    for cmd in steps:
        _log(f"[{app.tool}] modal {' '.join(cmd[3:])}")
        code, output = _run_streamed(cmd, env)
        if code != 0 and _builder_fallback(output, env):
            _log(f"[{app.tool}] The Modal client refused this workspace's image builder "
                 f"version, which is newer than it knows. Retrying with "
                 f"{BUILDER_VAR}={FALLBACK_BUILDER_VERSION}; set it in your shell to "
                 f"run `modal deploy` by hand (docs/modal-setup.md).", "warn")
            # Later apps in this run start with it, so only the first one fails.
            os.environ[BUILDER_VAR] = env[BUILDER_VAR] = FALLBACK_BUILDER_VERSION
            result["image_builder_version"] = FALLBACK_BUILDER_VERSION
            code, output = _run_streamed(cmd, env)
        if code != 0:
            result["error"] = f"`modal {cmd[3]}` exited {code}"
            hint = _hint(output)
            if hint:
                result["hint"] = hint
            return result

    url, reason = lookup_url(app)
    if not url:
        result["error"] = f"deployed, but the endpoint URL lookup failed: {reason}"
        return result

    write_env_value(app.env_var, url)
    result.update(success=True, endpoint_url=url)
    return result


def status(app: ModalApp) -> dict:
    url, reason = lookup_url(app)
    saved = read_env_value(app.env_var) or os.getenv(app.env_var)
    if not url:
        state = "not deployed" if reason == "not deployed" else f"lookup failed ({reason})"
    elif not saved:
        state = "deployed, missing from .env"
    elif saved != url:
        state = "deployed, .env is stale"
    else:
        state = "ok"
    return {"tool": app.tool, "label": app.label, "env_var": app.env_var,
            "deployed_url": url, "env_url": saved, "state": state}


def sync_env(app: ModalApp) -> dict:
    s = status(app)
    s["updated"] = False
    if s["deployed_url"] and s["deployed_url"] != s["env_url"]:
        write_env_value(app.env_var, s["deployed_url"])
        s["updated"] = True
        s["state"] = "ok"
    return s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _select(names: list[str], all_: bool, default_all: bool) -> list[ModalApp]:
    unknown = [n for n in names if n not in APPS]
    if unknown:
        _log(f"Unknown app(s): {', '.join(unknown)}. Choose from: {', '.join(APPS)}", "error")
        sys.exit(2)
    if names:
        return [APPS[n] for n in names]
    if all_:
        return [a for a in APPS.values() if not a.explicit_only]
    return list(APPS.values()) if default_all else []


def main() -> int:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Deploy the toolkit's Modal apps and write their endpoint URLs into .env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Apps: " + ", ".join(
            f"{a.tool}{' (explicit only)' if a.explicit_only else ''}" for a in APPS.values()),
    )
    parser.add_argument("apps", nargs="*", metavar="APP", help="apps to deploy (see list below)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="deploy every app except ltx2")
    mode.add_argument("--list", action="store_true",
                      help="show deploy and .env status (no changes)")
    mode.add_argument("--sync-env", action="store_true",
                      help="no deploy: write the URLs of already-deployed apps into .env")
    parser.add_argument("--hf-secret", metavar="NAME",
                        help=f"Modal secret holding HF_TOKEN for gated weights "
                             f"(default: {DEFAULT_HF_SECRET}, or {TEMPLATE_HF_SECRET} if "
                             f"that's the only one)")
    parser.add_argument("--recreate", action="store_true",
                        help="deploy with --strategy recreate (use when switching a "
                             "FLUX2_MODEL / IMAGE_EDIT_MODEL / LTX2_VERSION variant)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--json", action="store_true", help="machine-readable output on stdout")
    args = parser.parse_args()

    _require_sdk()

    if args.list or args.sync_env:
        apps = _select(args.apps, args.all, default_all=True)
        rows = [sync_env(a) if args.sync_env else status(a) for a in apps]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                mark = "OK" if r["state"] == "ok" else "--"
                extra = "  (written to .env)" if r.get("updated") else ""
                print(f"  {mark}  {r['tool']:<12} {r['state']}{extra}")
                if r["deployed_url"] and r["state"] != "ok":
                    print(f"        {r['env_var']}={r['deployed_url']}")
        return 0

    apps = _select(args.apps, args.all, default_all=False)
    if not apps:
        parser.error("name one or more apps, or pass --all / --list / --sync-env")

    results = []
    for app in apps:
        r = deploy(app, hf_secret=args.hf_secret, recreate=args.recreate, dry_run=args.dry_run)
        results.append(r)
        if args.dry_run:
            for c in r.get("commands", [r.get("error")]):
                print(c)
        elif r["success"]:
            _log(f"{app.tool}: {app.env_var}={r['endpoint_url']}", "success")
        else:
            _log(f"{app.tool}: {r['error']}", "error")
            if r.get("hint"):
                _log(f"  {r['hint']}", "warn")

    if args.json:
        print(json.dumps(results, indent=2))
    failed = [r["tool"] for r in results if not r["success"]]
    if failed and len(results) > 1:
        _log(f"Failed: {', '.join(failed)}. Re-run with just those names once fixed.", "error")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
