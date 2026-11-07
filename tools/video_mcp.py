#!/usr/bin/env python3
"""MCP server for agents without a shell: brief or spec in, MP4 out.

A thin stdio wrapper over tools/video_job.py. Every tool here calls into that module and
holds no pipeline logic of its own, so the CLI and the MCP server can never drift.

    uv sync --extra mcp
    uv run tools/video_mcp.py            # stdio transport, for an MCP client to spawn

Claude Desktop / Claude Code (one line in the client's MCP config):

    {"mcpServers": {"video-toolkit": {"command": "uv",
      "args": ["run", "--directory", "/path/to/claude-code-video-toolkit", "--extra", "mcp",
               "tools/video_mcp.py"]}}}

    claude mcp add video-toolkit -- uv run --directory /path/to/claude-code-video-toolkit \\
        --extra mcp tools/video_mcp.py

Any other MCP client: spawn the same command and speak MCP over its stdin/stdout.

Tools: video_estimate, video_submit, video_status, video_result, video_list_presets, video_catalog
(plus video_list, video_cancel and video_resume, which cost nothing to expose). Every tool
returns a dict with `ok`; failures carry `error` and `error_code`, never an exception.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import video_job as vj  # noqa: E402

try:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp 2.x (FastMCP was renamed)
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
    except ImportError:
        sys.exit("The MCP server needs the 'mcp' extra:\n    uv sync --extra mcp\n"
                 "then run it with: uv run --extra mcp tools/video_mcp.py")

mcp = FastMCP(
    "video-toolkit",
    instructions=(
        "Make short explainer videos. Flow: video_list_presets (once, for prices and the spec "
        "shape) -> write a spec -> video_estimate -> video_submit -> poll video_status every "
        "20-30 s until done -> video_result for the MP4 path and URL. You are the LLM: write the "
        "spec yourself (scenes with narration text and a visual prompt each) rather than passing "
        "a brief, unless the server has a VIDEO_JOB_EXPANDER configured. video_status has an "
        "`events` list (what started and finished) and `warnings` to pass on to the user; a failed "
        "job with `resumable: true` can be carried on with video_resume, paying only for what is missing. "
        "Asked what you can make, or for a style this flow doesn't do (a product walkthrough rebuilt from an "
        "app's source, a data chart, a sprint review), call video_catalog: templates with agent_support `job` "
        "go through submit; `session` ones need a working session with the skill each one names."
    ),
)


def _err(e: Exception, code: str) -> dict[str, Any]:
    return {"ok": False, "error": str(e), "error_code": code}


def _resolve(spec: dict | None, brief: str | None, fmt: str, duration_s: int, allow_skeleton: bool,
             kit: str | None = None):
    if spec:
        if kit and not spec.get("kit"):
            spec = {**spec, "kit": kit}
        if fmt and not spec.get("format"):
            spec = {**spec, "format": fmt}
        return vj.normalize_spec(spec), "spec"
    fmt = fmt or (vj.load_kit(kit).get("format") if kit else None) or "9:16"   # the kit's format unless asked
    return vj.resolve_spec(brief, None, fmt, duration_s, allow_skeleton=allow_skeleton, kit=kit)


@mcp.tool()
def video_estimate(spec: dict | None = None, brief: str | None = None, format: str | None = None,
                   duration_s: int = 45, kit: str | None = None) -> dict:
    """Cost estimate in USD for a video, with a per-step breakdown. Pass `spec` (preferred) or
    `brief`, and `kit` to make it for someone with a kit (video_catalog lists them). Needs no API keys. With only a brief and no expander configured, the estimate is
    for the skeleton a brief of that length usually takes. `host_missing` lists providers the
    spec uses that this host has no key or endpoint for (video_submit refuses those; entries marked
    `soft` only mean the video is made without that part, e.g. no music bed)."""
    try:
        s, source = _resolve(spec, brief, format, duration_s, allow_skeleton=True, kit=kit)
        return {"ok": True, **vj.estimate(s), "spec_source": source,
                "host_missing": vj.missing_for(vj.normalize_spec(s), include_soft=True)}
    except (vj.SpecError, ValueError, vj.NoExpander, RuntimeError) as e:
        return _err(e, "bad_request")
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


@mcp.tool()
def video_submit(spec: dict | None = None, brief: str | None = None, format: str | None = None,
                 duration_s: int = 45, max_usd: float = vj.DEFAULT_MAX_USD, kit: str | None = None) -> dict:
    """Start a video job and return at once with its id and estimate. Refused (ok=false,
    error_code budget_exceeded) when the estimate is over max_usd, or (unavailable, with `missing`) when this
    host has no key or endpoint for a provider the spec uses, both before any spend; busy
    (error_code lock_held) when another job is running. Poll video_status afterwards."""
    try:
        s, source = _resolve(spec, brief, format, duration_s, allow_skeleton=False, kit=kit)
        job = vj.submit(s, max_usd, spec_source=source, brief=brief)
        return {"ok": True, "id": job["id"], "phase": job["phase"], "estimate_usd": job["estimate_usd"],
                "max_usd": job["max_usd"], "title": job["title"], "format": job["format"], "scenes": job["scenes"]}
    except vj.BudgetExceeded as e:
        return _err(e, "budget_exceeded")
    except vj.Unavailable as e:
        return {**_err(e, "unavailable"), "missing": e.missing}
    except vj.LockHeld as e:
        return {**_err(e, "lock_held"), "holder": e.holder}
    except vj.NoExpander as e:
        return _err(e, "no_expander")
    except (vj.SpecError, ValueError, RuntimeError) as e:
        return _err(e, "bad_request")
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


def _job_call(fn, job_id: str) -> dict:
    try:
        return {"ok": True, **fn(job_id)}
    except vj.JobUnreadable as e:
        return _err(e, "job_unreadable")
    except (FileNotFoundError, vj.SpecError) as e:
        return _err(e, "not_found")
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


@mcp.tool()
def video_status(job_id: str) -> dict:
    """Phase (queued, voiceover, visuals, captions, music, build, qc, publish, done, failed,
    cancelled), progress %, eta_s, spend so far, any error, `events` (timestamped: scenes
    voiced, assets made, QC, publish), `warnings`, `resumable`, and the last log lines."""
    return _job_call(vj.status, job_id)


@mcp.tool()
def video_result(job_id: str) -> dict:
    """The finished MP4: local path, a download URL when R2 is configured, duration, size, QC
    verdict, `warnings` to pass on, and actual spend. `ready` is false until the job is done."""
    return _job_call(vj.result, job_id)


@mcp.tool()
def video_resume(job_id: str, max_usd: float | None = None) -> dict:
    """Carry on a failed or cancelled job (status says `resumable: true`) from its first
    missing output: voice, visuals, captions and music already made are reused, so only the
    rest is paid for. `max_usd` raises the job's cap (needed after a budget_exceeded stop)."""
    try:
        j = vj.resume(job_id, max_usd=max_usd)
        return {"ok": True, "id": j["id"], "phase": j["phase"], "resumes": j["resumes"],
                "actual_usd": j.get("actual_usd"), "max_usd": j.get("max_usd")}
    except vj.LockHeld as e:
        return {**_err(e, "lock_held"), "holder": e.holder}
    except vj.JobUnreadable as e:
        return _err(e, "job_unreadable")
    except FileNotFoundError as e:
        return _err(e, "not_found")
    except vj.SpecError as e:
        return _err(e, "not_resumable")
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


@mcp.tool()
def video_list_presets() -> dict:
    """Unit prices, output formats, visual kinds, voice defaults and an example spec."""
    return {"ok": True, **vj.presets()}


@mcp.tool()
def video_catalog() -> dict:
    """What the toolkit can make: templates (with demo videos and the skill that drives each), example
    builds, components, brands, design sources and voice presets. `agent_support` is `job` (video_submit makes it) or `session`."""
    try:
        return {"ok": True, **vj.catalog()}
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


@mcp.tool()
def video_list() -> list[dict]:
    """Recent jobs, newest first (each with `resumable`)."""
    return vj.list_jobs()


@mcp.tool()
def video_cancel(job_id: str) -> dict:
    """Stop a running job and release the lock."""
    try:
        j = vj.cancel(job_id)
        return {"ok": True, "id": j["id"], "phase": j["phase"]}
    except (FileNotFoundError, vj.SpecError) as e:
        return _err(e, "not_found")
    except Exception as e:  # noqa: BLE001
        return _err(e, "internal")


if __name__ == "__main__":
    mcp.run(transport="stdio")
