#!/usr/bin/env python3
"""Verify toolkit setup — checks prerequisites, cloud GPU, R2, and voice configuration.

Run after /setup to confirm everything is working, or anytime to diagnose issues.

Usage:
    uv run tools/verify_setup.py           # Full check (no cloud calls)
    uv run tools/verify_setup.py --test    # Full check + smoke tests (makes cloud GPU calls)
    uv run tools/verify_setup.py --json    # Machine-readable output
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv

load_dotenv()


def check_command(cmd: list[str]) -> tuple[bool, str]:
    """Check if a command exists and return its version."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            return True, version
        return False, result.stderr.strip()[:100]
    except FileNotFoundError:
        return False, "not installed"
    except Exception as e:
        return False, str(e)[:100]


def check_prerequisites() -> list[dict]:
    """Check required and optional prerequisites."""
    results = []

    # Node.js (required)
    ok, version = check_command(["node", "--version"])
    results.append({
        "name": "Node.js",
        "required": True,
        "ok": ok,
        "detail": version if ok else "Install from https://nodejs.org/",
    })

    # Python (recommended)
    ok, version = check_command(["python3", "--version"])
    results.append({
        "name": "Python",
        "required": False,
        "ok": ok,
        "detail": version if ok else "Install from https://python.org/",
    })

    # FFmpeg (optional)
    ok, version = check_command(["ffmpeg", "-version"])
    if ok:
        # Parse just the version line
        version = version.split(" Copyright")[0] if " Copyright" in version else version
    results.append({
        "name": "FFmpeg",
        "required": False,
        "ok": ok,
        "detail": version if ok else "brew install ffmpeg (macOS)",
    })

    # pip packages
    try:
        import requests  # noqa: F401
        results.append({"name": "pip packages", "required": False, "ok": True, "detail": "OK"})
    except ImportError:
        results.append({
            "name": "pip packages",
            "required": False,
            "ok": False,
            "detail": "uv sync",
        })

    # Modal CLI
    ok, version = check_command(["modal", "--version"])
    results.append({
        "name": "Modal CLI",
        "required": False,
        "ok": ok,
        "detail": version if ok else "uv sync --extra modal",
    })

    return results


def check_r2() -> dict:
    """Check Cloudflare R2 configuration."""
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET_NAME", "video-toolkit")

    if not account_id or account_id == "your_account_id_here":
        return {"ok": False, "detail": "R2_ACCOUNT_ID not set", "bucket": None}
    if not access_key or access_key == "your_access_key_id_here":
        return {"ok": False, "detail": "R2_ACCESS_KEY_ID not set", "bucket": None}
    if not secret_key:
        return {"ok": False, "detail": "R2_SECRET_ACCESS_KEY not set", "bucket": None}

    return {"ok": True, "detail": f"bucket: {bucket}", "bucket": bucket}


def test_r2_connectivity() -> dict:
    """Actually test R2 upload/download (only with --test flag)."""
    try:
        from file_transfer import upload_to_r2, delete_from_r2

        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(b"verify_setup connectivity test")
        tmp.close()

        url, key = upload_to_r2(tmp.name, "setup-verify")
        os.unlink(tmp.name)

        if url and key:
            delete_from_r2(key)
            return {"ok": True, "detail": "upload/download verified"}
        return {"ok": False, "detail": "upload returned no URL"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


def check_modal_apps() -> dict:
    """Check which Modal apps are deployed."""
    if not shutil.which("modal"):
        return {"ok": False, "apps": [], "detail": "Modal CLI not installed"}

    try:
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {"ok": False, "apps": [], "detail": "modal app list failed — run `uv run modal setup`?"}

        apps = json.loads(result.stdout)
        # `modal app list --json` key casing depends on CLI version: <=1.3.x emits
        # the capitalised table headers ("Description", "State"), newer CLIs emit
        # lowercase ("description", "state"). Accept both.
        def field(app: dict, key: str) -> str:
            return app.get(key) or app.get(key.capitalize()) or ""

        toolkit_apps = [
            a for a in apps
            if field(a, "description").startswith("video-toolkit-")
            and field(a, "state") == "deployed"
        ]

        app_names = [field(a, "description") for a in toolkit_apps]
        return {
            "ok": len(toolkit_apps) > 0,
            "apps": app_names,
            "detail": f"{len(toolkit_apps)} deployed" if toolkit_apps else "no toolkit apps deployed",
        }
    except Exception as e:
        return {"ok": False, "apps": [], "detail": str(e)[:200]}


def check_modal_env_vars() -> list[dict]:
    """Check which Modal endpoint URLs are configured."""
    tools = {
        "MODAL_QWEN3_TTS_ENDPOINT_URL": "Speech (Qwen3-TTS)",
        "MODAL_FLUX2_ENDPOINT_URL": "Images (FLUX.2)",
        "MODAL_IMAGE_EDIT_ENDPOINT_URL": "Image Editing (Qwen-Edit)",
        "MODAL_UPSCALE_ENDPOINT_URL": "Upscaling (RealESRGAN)",
        "MODAL_MUSIC_GEN_ENDPOINT_URL": "Music (ACE-Step)",
        "MODAL_SADTALKER_ENDPOINT_URL": "Talking Heads (SadTalker)",
        "MODAL_DEWATERMARK_ENDPOINT_URL": "Watermark Removal (ProPainter)",
        "MODAL_LTX2_ENDPOINT_URL": "Video (LTX-2)",
    }

    results = []
    for env_var, name in tools.items():
        url = os.getenv(env_var)
        results.append({
            "name": name,
            "env_var": env_var,
            "ok": bool(url and url.startswith("https://")),
            "detail": "configured" if url and url.startswith("https://") else "not set",
        })
    return results


def check_runpod_env_vars() -> list[dict]:
    """Check which RunPod endpoint IDs are configured."""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        return [{"name": "RunPod API Key", "ok": False, "detail": "not set"}]

    tools = {
        "RUNPOD_QWEN3_TTS_ENDPOINT_ID": "Speech (Qwen3-TTS)",
        "RUNPOD_FLUX2_ENDPOINT_ID": "Images (FLUX.2)",
        "RUNPOD_QWEN_EDIT_ENDPOINT_ID": "Image Editing (Qwen-Edit)",
        "RUNPOD_UPSCALE_ENDPOINT_ID": "Upscaling (RealESRGAN)",
        "RUNPOD_ACESTEP_ENDPOINT_ID": "Music (ACE-Step)",
        "RUNPOD_SADTALKER_ENDPOINT_ID": "Talking Heads (SadTalker)",
        "RUNPOD_ENDPOINT_ID": "Watermark Removal (ProPainter)",
    }

    results = [{"name": "RunPod API Key", "ok": True, "detail": "configured"}]
    for env_var, name in tools.items():
        val = os.getenv(env_var)
        results.append({
            "name": name,
            "env_var": env_var,
            "ok": bool(val),
            "detail": "configured" if val else "not set",
        })
    return results


def check_voice() -> dict:
    """Check voice/TTS configuration."""
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice = os.getenv("ELEVENLABS_VOICE_ID")

    # Qwen3-TTS is "configured" if its endpoint exists (either Modal or RunPod)
    qwen3_modal = os.getenv("MODAL_QWEN3_TTS_ENDPOINT_URL")
    qwen3_runpod = os.getenv("RUNPOD_QWEN3_TTS_ENDPOINT_ID")
    qwen3_ok = bool(qwen3_modal) or bool(qwen3_runpod)

    return {
        "qwen3_tts": {"ok": qwen3_ok, "detail": "Modal" if qwen3_modal else ("RunPod" if qwen3_runpod else "not configured")},
        "elevenlabs": {"ok": bool(elevenlabs_key), "detail": "configured" if elevenlabs_key else "not configured (optional)"},
    }


def test_cloud_endpoint(tool_name: str, provider: str = "modal") -> dict:
    """Smoke test a cloud GPU endpoint (only with --test flag)."""
    try:
        from cloud_gpu import call_cloud_endpoint

        if tool_name == "qwen3_tts":
            payload = {
                "text": "Test.",
                "speaker": "Ryan",
                "tone": "neutral",
            }
            result, elapsed = call_cloud_endpoint(
                provider=provider,
                payload={"input": payload} if provider == "runpod" else payload,
                tool_name="qwen3_tts",
                timeout=120,
                progress_label="Testing TTS",
                verbose=False,
            )
            ok = isinstance(result, dict) and not result.get("error")
            return {"ok": ok, "detail": f"{elapsed:.1f}s" if ok else result.get("error", "failed")[:100]}

        elif tool_name == "flux2":
            payload = {
                "prompt": "solid blue square",
                "steps": 4,
                "guidance_scale": 1.0,
            }
            result, elapsed = call_cloud_endpoint(
                provider=provider,
                payload={"input": payload} if provider == "runpod" else payload,
                tool_name="flux2",
                timeout=120,
                progress_label="Testing image gen",
                verbose=False,
            )
            ok = isinstance(result, dict) and not result.get("error")
            return {"ok": ok, "detail": f"{elapsed:.1f}s" if ok else result.get("error", "failed")[:100]}

        return {"ok": False, "detail": f"no smoke test for {tool_name}"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify toolkit setup")
    parser.add_argument("--test", action="store_true", help="Run smoke tests (makes cloud GPU calls, costs ~$0.01)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    verbose = not args.json
    results = {}

    # 1. Prerequisites
    prereqs = check_prerequisites()
    results["prerequisites"] = prereqs
    if verbose:
        print("Prerequisites:")
        for p in prereqs:
            status = "OK" if p["ok"] else ("MISSING" if p["required"] else "not found")
            req = " (required)" if p["required"] else ""
            print(f"  {'[x]' if p['ok'] else '[ ]'} {p['name']}{req}: {p['detail']}")
        print()

    # 2. Cloudflare R2
    r2 = check_r2()
    results["r2"] = r2
    if verbose:
        print(f"Cloudflare R2: {'[x]' if r2['ok'] else '[ ]'} {r2['detail']}")

    if args.test and r2["ok"]:
        r2_test = test_r2_connectivity()
        results["r2_test"] = r2_test
        if verbose:
            print(f"  Connectivity: {'[x]' if r2_test['ok'] else '[ ]'} {r2_test['detail']}")
    print()

    # 3. Modal
    modal_env = check_modal_env_vars()
    modal_configured = sum(1 for t in modal_env if t["ok"])
    results["modal_tools"] = modal_env
    if verbose:
        print(f"Modal ({modal_configured}/{len(modal_env)} tools):")
        for t in modal_env:
            print(f"  {'[x]' if t['ok'] else '[ ]'} {t['name']}: {t['detail']}")

    if shutil.which("modal"):
        modal_apps = check_modal_apps()
        results["modal_apps"] = modal_apps
        if verbose:
            print(f"  Deployed apps: {modal_apps['detail']}")
    print()

    # 4. RunPod
    runpod_env = check_runpod_env_vars()
    results["runpod_tools"] = runpod_env
    if verbose:
        runpod_configured = sum(1 for t in runpod_env if t["ok"])
        print(f"RunPod ({runpod_configured}/{len(runpod_env)} configured):")
        for t in runpod_env:
            print(f"  {'[x]' if t['ok'] else '[ ]'} {t['name']}: {t['detail']}")
        print()

    # 5. Voice
    voice = check_voice()
    results["voice"] = voice
    if verbose:
        print("Voice:")
        print(f"  {'[x]' if voice['qwen3_tts']['ok'] else '[ ]'} Qwen3-TTS: {voice['qwen3_tts']['detail']}")
        print(f"  {'[x]' if voice['elevenlabs']['ok'] else '[ ]'} ElevenLabs: {voice['elevenlabs']['detail']}")
        print()

    # 6. Smoke tests (optional)
    if args.test:
        if verbose:
            print("Smoke tests:")

        # Pick best available provider
        provider = "modal" if modal_configured > 0 else "runpod"

        # Test TTS if available
        qwen3_ok = voice["qwen3_tts"]["ok"]
        if qwen3_ok:
            tts_test = test_cloud_endpoint("qwen3_tts", provider)
            results["test_qwen3_tts"] = tts_test
            if verbose:
                print(f"  {'[x]' if tts_test['ok'] else '[ ]'} Qwen3-TTS ({provider}): {tts_test['detail']}")

        # Test FLUX.2 if available
        flux2_ok = any(t["ok"] for t in modal_env if "FLUX2" in t.get("env_var", ""))
        if not flux2_ok:
            flux2_ok = any(t["ok"] for t in runpod_env if "FLUX2" in t.get("env_var", ""))
        if flux2_ok:
            flux2_test = test_cloud_endpoint("flux2", provider)
            results["test_flux2"] = flux2_test
            if verbose:
                print(f"  {'[x]' if flux2_test['ok'] else '[ ]'} FLUX.2 ({provider}): {flux2_test['detail']}")

        if verbose:
            print()

    # Summary
    all_ok = all(p["ok"] for p in prereqs if p["required"])
    cloud_ok = modal_configured > 0 or sum(1 for t in runpod_env if t["ok"]) > 1
    r2_ok = r2["ok"]
    voice_ok = voice["qwen3_tts"]["ok"] or voice["elevenlabs"]["ok"]

    results["summary"] = {
        "prerequisites": all_ok,
        "cloud_gpu": cloud_ok,
        "file_transfer": r2_ok,
        "voice": voice_ok,
        "ready": all_ok,  # Only prereqs are truly required
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 40)
        print(f"  Prerequisites:  {'ready' if all_ok else 'ISSUES'}")
        print(f"  Cloud GPU:      {'ready' if cloud_ok else 'not configured'}")
        print(f"  File transfer:  {'ready' if r2_ok else 'not configured (using fallback)'}")
        print(f"  Voice:          {'ready' if voice_ok else 'not configured'}")
        print()
        if all_ok and cloud_ok and r2_ok and voice_ok:
            print("  All systems go! Run /video to create a video.")
        elif all_ok:
            print("  Basics ready. Run /setup to configure cloud features.")
        else:
            print("  Some prerequisites missing. Check above for details.")
        print("=" * 40)


if __name__ == "__main__":
    main()
