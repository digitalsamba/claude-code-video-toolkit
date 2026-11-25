"""Shared JSONL receipts for video jobs and direct media requests."""
from __future__ import annotations

import datetime
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGED_SPEND_ENV = "MEDIA_SPEND_MANAGED"
ELEVENLABS_TTS_USD_PER_1K_CHARS = 0.20
SCRIBE_USD_PER_SECOND = 0.40 / 3600


def spend_log_path() -> Path:
    """MEDIA_SPEND_LOG, then the agent workspace's logs, then the toolkit's logs."""
    if path := os.environ.get("MEDIA_SPEND_LOG"):
        return Path(path)
    workspace = os.environ.get("VIDEO_JOB_WORKSPACE") or os.environ.get("LONGARM_WORKSPACE")
    return (Path(workspace) if workspace else ROOT) / "logs" / "media-spend.log"


@contextmanager
def _append_lock(path: Path):
    if os.name != "nt":
        yield  # O_APPEND makes the single write atomic on POSIX.
        return
    import msvcrt

    # Windows' CRT seeks before appending. Lock a separate file so readers can still read the ledger.
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def append_receipt(path: Path, entry: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(entry, allow_nan=False) + "\n").encode("utf-8")
    with _append_lock(path), path.open("ab", buffering=0) as ledger:
        if ledger.write(payload) != len(payload):
            raise OSError(f"short write to {path}")


def record_spend(tool: str, provider: str, unit: str, units: int | float | None, *,
                 cost_usd: float | None = None, elapsed_s: float | None = None, model: str | None = None) -> None:
    """Record an accepted provider call. The worker owns receipts for its child processes."""
    if os.environ.get(MANAGED_SPEND_ENV) == "1":
        return
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool,
        "provider": provider,
        "unit": unit,
        "units": units,
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
        "cost_source": "estimate" if cost_usd is not None else "unpriced",
        "elapsed_s": elapsed_s,
        "exit": 0,
        "args": [],
    }
    if model:
        entry["model"] = model
    try:
        append_receipt(spend_log_path(), entry)
    except (OSError, ValueError) as error:
        print(f"Warning: {tool} could not log media spend: {error}", file=sys.stderr)


def record_tts(tool: str, characters: int, model: str) -> None:
    record_spend(tool, "elevenlabs", "characters", characters, model=model,
                 cost_usd=characters / 1000 * ELEVENLABS_TTS_USD_PER_1K_CHARS)


def record_scribe(tool: str, seconds: float | None, model: str) -> None:
    record_spend(tool, "elevenlabs", "audio_seconds", seconds, model=model,
                 cost_usd=seconds * SCRIBE_USD_PER_SECOND if seconds is not None else None)
