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

Tools: video_estimate, video_submit, video_status, video_result, video_list_presets
(plus video_list and video_cancel, which cost nothing to expose).
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
        "a brief, unless the server has a VIDEO_JOB_EXPANDER configured."
    ),
)


def _err(e: Exception, code: str) -> dict[str, Any]:
    return {"ok": False, "error": str(e), "error_code": code}


def _resolve(spec: dict | None, brief: str | None, fmt: str, duration_s: int, allow_skeleton: bool):
    if spec:
        return vj.normalize_spec(spec), "spec"
    return vj.resolve_spec(brief, None, fmt, duration_s, allow_skeleton=allow_skeleton)


@mcp.tool()
def video_estimate(spec: dict | None = None, brief: str | None = None, format: str = "9:16",
                   duration_s: int = 45) -> dict:
    """Cost estimate in USD for a video, with a per-step breakdown. Pass `spec` (preferred) or
    `brief`. Needs no API keys. With only a brief and no expander configured, the estimate is
    for the skeleton a brief of that length usually takes."""
    try:
        s, source = _resolve(spec, brief, format, duration_s, allow_skeleton=True)
        return {"ok": True, **vj.estimate(s), "spec_source": source}
    except (vj.SpecError, ValueError, vj.NoExpander, RuntimeError) as e:
        return _err(e, "bad_request")


@mcp.tool()
def video_submit(spec: dict | None = None, brief: str | None = None, format: str = "9:16",
                 duration_s: int = 45, max_usd: float = vj.DEFAULT_MAX_USD) -> dict:
    """Start a video job and return at once with its id and estimate. Refused (ok=false,
    error_code budget_exceeded) when the estimate is over max_usd, before any spend; busy
    (error_code lock_held) when another job is running. Poll video_status afterwards."""
    try:
        s, source = _resolve(spec, brief, format, duration_s, allow_skeleton=False)
        job = vj.submit(s, max_usd, spec_source=source, brief=brief)
        return {"ok": True, "id": job["id"], "phase": job["phase"], "estimate_usd": job["estimate_usd"],
                "max_usd": job["max_usd"], "title": job["title"], "format": job["format"], "scenes": job["scenes"]}
    except vj.BudgetExceeded as e:
        return _err(e, "budget_exceeded")
    except vj.LockHeld as e:
        return {**_err(e, "lock_held"), "holder": e.holder}
    except vj.NoExpander as e:
        return _err(e, "no_expander")
    except (vj.SpecError, ValueError, RuntimeError) as e:
        return _err(e, "bad_request")


@mcp.tool()
def video_status(job_id: str) -> dict:
    """Phase (queued, voiceover, visuals, captions, music, build, qc, publish, done, failed,
    cancelled), progress %, spend so far, any error, and the last log lines."""
    try:
        return {"ok": True, **vj.status(job_id)}
    except (FileNotFoundError, vj.SpecError) as e:
        return _err(e, "not_found")


@mcp.tool()
def video_result(job_id: str) -> dict:
    """The finished MP4: local path, a download URL when R2 is configured, duration, QC verdict
    and actual spend. `ready` is false until the job is done."""
    try:
        return {"ok": True, **vj.result(job_id)}
    except (FileNotFoundError, vj.SpecError) as e:
        return _err(e, "not_found")


@mcp.tool()
def video_list_presets() -> dict:
    """Unit prices, output formats, visual kinds, voice defaults and an example spec."""
    return {"ok": True, **vj.presets()}


@mcp.tool()
def video_list() -> list[dict]:
    """Recent jobs, newest first."""
    return vj.list_jobs()


@mcp.tool()
def video_cancel(job_id: str) -> dict:
    """Stop a running job and release the lock."""
    try:
        j = vj.cancel(job_id)
        return {"ok": True, "id": j["id"], "phase": j["phase"]}
    except (FileNotFoundError, vj.SpecError) as e:
        return _err(e, "not_found")


if __name__ == "__main__":
    mcp.run(transport="stdio")
