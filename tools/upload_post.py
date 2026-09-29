#!/usr/bin/env python3
"""
Publish a rendered video to TikTok, Instagram, YouTube, LinkedIn, Facebook, X,
Threads, Pinterest and Bluesky in one call via the Upload-Post API.

Unlike tools/youtube_upload.py there is no OAuth dance here: you connect your
social accounts once in the Upload-Post dashboard (grouped under a "profile"), and
this tool authenticates with a single API key. In short:
  1. Create an account at https://upload-post.com, create a profile and connect
     your social accounts to it.
  2. Create an API key in the dashboard and put it in .env as UPLOAD_POST_API_KEY.
  3. Put the profile name in .env as UPLOAD_POST_USER (or pass --user).

See docs/upload-post.md for the full walkthrough.

Examples:
  # Validate everything without posting (also checks the key and which of the
  # requested platforms the profile actually has connected)
  uv run tools/upload_post.py --video out/short.mp4 --title "My short" \\
      --platforms tiktok,instagram,youtube --dry-run --json-out

  # Post a vertical short to TikTok + Reels + Shorts now, and wait for the result
  uv run tools/upload_post.py --video out/short.mp4 --title "My short" \\
      --platforms tiktok,instagram,youtube --json-out

  # Schedule a LinkedIn + X post for tomorrow morning (your timezone)
  uv run tools/upload_post.py --video out/video.mp4 --title "Launch" \\
      --description-file DESCRIPTION.md --platforms linkedin,x \\
      --schedule 2026-10-01T09:00:00 --timezone Europe/London

  # Check on an earlier upload
  uv run tools/upload_post.py --status <request_id> --json-out

# ---------------------------------------------------------------------------
# HOW PUBLISHING WORKS
#   * The video is sent once; Upload-Post fans it out to every platform and
#     returns one result per platform (URL on success, error otherwise).
#   * Uploads are asynchronous: the API returns a request_id straight away and
#     this tool polls GET /api/uploadposts/status until every platform finishes
#     (or --wait-timeout expires; the upload keeps going server-side).
#   * Scheduled uploads return a job_id instead; poll it later with --status.
#   * The tool generates its own request_id and sends it as an Idempotency-Key,
#     so a network error mid-upload never double-posts: it resumes by polling the
#     same request_id instead of re-sending the file.
#   * Platforms the profile has no account for come back as "skipped" rather
#     than failing the whole upload.
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv sync  (or pip install requests python-dotenv)")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

API_BASE = "https://api.upload-post.com"

# Platforms that accept video through /api/upload.
VIDEO_PLATFORMS = (
    "tiktok",
    "instagram",
    "youtube",
    "linkedin",
    "facebook",
    "x",
    "threads",
    "pinterest",
    "bluesky",
)
PLATFORM_ALIASES = {"twitter": "x", "reels": "instagram", "shorts": "youtube"}

YOUTUBE_PRIVACY = ("private", "unlisted", "public")
TIKTOK_PRIVACY = (
    "PUBLIC_TO_EVERYONE",
    "MUTUAL_FOLLOW_FRIENDS",
    "FOLLOWER_OF_CREATOR",
    "SELF_ONLY",
)
FINAL_STATUSES = {"completed", "failed", "not_found"}

POLL_INTERVAL_SECS = 10
DEFAULT_WAIT_SECS = 600
UPLOAD_TIMEOUT = (30, 900)  # (connect, read) — the read covers sending the file
API_TIMEOUT = (15, 60)

TITLE_MAX = {"youtube": 100, "tiktok": 2200}


class ApiError(Exception):
    """An Upload-Post API call failed. errorType mirrors youtube_upload.py's."""

    def __init__(self, message: str, error_type: str = "http", status: Optional[int] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status = status


def log(msg: str, level: str = "info"):
    """Print a formatted log message to stderr (stdout is reserved for --json-out)."""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "error": "\033[91m",
        "warn": "\033[93m",
        "dim": "\033[90m",
    }
    reset = "\033[0m"
    prefix = {"info": "->", "success": "OK", "error": "!!", "warn": "??", "dim": "  "}
    color = colors.get(level, "")
    print(f"{color}{prefix.get(level, '->')} {msg}{reset}", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _headers(api_key: str) -> dict:
    # Upload-Post API keys go in an "Apikey" header, not "Bearer".
    return {"Authorization": f"Apikey {api_key}"}


def _error_message(resp: "requests.Response") -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300] or resp.reason
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)
    return str(body)


def classify_status(status: int) -> str:
    """Map an HTTP status to errorType (auth | validation | quota | forbidden | http)."""
    if status == 401:
        return "auth"
    if status in (400, 422):
        return "validation"
    if status == 429:
        return "quota"
    if status == 403:
        return "forbidden"
    return "http"


def _check(resp: "requests.Response") -> dict:
    if resp.ok:
        try:
            return resp.json()
        except ValueError:
            raise ApiError(f"Unexpected non-JSON response (HTTP {resp.status_code})", "http", resp.status_code)
    raise ApiError(
        f"HTTP {resp.status_code}: {_error_message(resp)}",
        classify_status(resp.status_code),
        resp.status_code,
    )


def api_get(api_key: str, path: str, params: Optional[dict] = None) -> dict:
    try:
        resp = requests.get(f"{API_BASE}{path}", headers=_headers(api_key), params=params, timeout=API_TIMEOUT)
    except requests.RequestException as e:
        raise ApiError(f"Network error: {e}", "http")
    return _check(resp)


def check_account(api_key: str, user: str) -> dict:
    """Confirm the key works and list the platforms connected to `user`.

    Returns {"plan": ..., "connected": [platform, ...]}. Raises ApiError.
    """
    me = api_get(api_key, "/api/uploadposts/me")
    profile = api_get(api_key, f"/api/uploadposts/users/{user}")
    accounts = (profile.get("profile") or profile).get("social_accounts") or {}
    connected = sorted(
        PLATFORM_ALIASES.get(p, p) for p, acc in accounts.items() if acc
    )
    return {"plan": me.get("plan"), "connected": connected}


def get_status(api_key: str, *, request_id: Optional[str] = None, job_id: Optional[str] = None) -> dict:
    params = {"request_id": request_id} if request_id else {"job_id": job_id}
    try:
        resp = requests.get(
            f"{API_BASE}/api/uploadposts/status",
            headers=_headers(api_key),
            params=params,
            timeout=API_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ApiError(f"Network error: {e}", "http")
    if resp.status_code == 404:
        return {"status": "not_found", **params}
    return _check(resp)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
def parse_platforms(raw: str) -> list[str]:
    platforms: list[str] = []
    for p in (raw or "").split(","):
        p = p.strip().lower()
        if not p:
            continue
        p = PLATFORM_ALIASES.get(p, p)
        if p not in platforms:
            platforms.append(p)
    return platforms


def parse_tags(raw: Optional[str]) -> list[str]:
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


def build_form(args, description: str, request_id: str) -> list[tuple[str, str]]:
    """Build the multipart form fields (everything except the video file itself).

    A list of tuples, because platform[] and tags[] repeat.
    """
    form: list[tuple[str, str]] = [
        ("user", args.user),
        ("title", args.title),
        ("request_id", request_id),
        ("async_upload", "true"),
    ]
    form += [("platform[]", p) for p in args.platforms]

    if description:
        form.append(("description", description))
    if args.schedule:
        form.append(("scheduled_date", args.schedule))
        if args.timezone:
            form.append(("timezone", args.timezone))
    if args.first_comment:
        form.append(("first_comment", args.first_comment))
    if args.ai_generated:
        # Cross-platform alias: TikTok AIGC label, Instagram "AI info",
        # YouTube containsSyntheticMedia, X made_with_ai.
        form.append(("is_ai_generated", "true"))

    if "youtube" in args.platforms:
        form.append(("privacyStatus", args.youtube_privacy))
        form.append(("categoryId", str(args.youtube_category)))
        form += [("tags[]", t) for t in parse_tags(args.tags)]
    if "tiktok" in args.platforms:
        if args.tiktok_privacy:
            form.append(("privacy_level", args.tiktok_privacy))
        if args.tiktok_draft:
            form.append(("post_mode", "MEDIA_UPLOAD"))
    if "instagram" in args.platforms and args.instagram_story:
        form.append(("media_type", "STORIES"))
    if "facebook" in args.platforms and args.facebook_page_id:
        form.append(("facebook_page_id", args.facebook_page_id))
    if "linkedin" in args.platforms and args.linkedin_page_id:
        form.append(("target_linkedin_page_id", args.linkedin_page_id))
    if "pinterest" in args.platforms:
        form.append(("pinterest_board_id", args.pinterest_board))
    return form


def submit_upload(api_key: str, video: str, form: list, thumbnail: Optional[str], request_id: str) -> dict:
    """POST the video. Returns the API response (request_id or job_id).

    On a transport error the upload may or may not have been received, so this
    never re-sends: the caller falls back to polling the same request_id.
    """
    headers = {**_headers(api_key), "Idempotency-Key": request_id}
    with ExitStack() as stack:
        files = [("video", (Path(video).name, stack.enter_context(open(video, "rb")), "video/mp4"))]
        if thumbnail:
            thumb_mime = mimetypes.guess_type(thumbnail)[0] or "image/png"
            files.append(("thumbnail", (Path(thumbnail).name, stack.enter_context(open(thumbnail, "rb")), thumb_mime)))
        resp = requests.post(
            f"{API_BASE}/api/upload",
            headers=headers,
            data=form,
            files=files,
            timeout=UPLOAD_TIMEOUT,
        )
    return _check(resp)


def wait_for_result(api_key: str, request_id: str, wait_secs: int) -> dict:
    """Poll the status endpoint until every platform finishes or wait_secs passes."""
    deadline = time.monotonic() + wait_secs
    last = None
    while True:
        status = get_status(api_key, request_id=request_id)
        state = status.get("status")
        progress = f"{status.get('completed', 0)}/{status.get('total', '?')}"
        if (state, progress) != last:
            log(f"Status: {state} ({progress} platforms done)", "dim")
            last = (state, progress)
        if state in FINAL_STATUSES:
            return status
        if time.monotonic() >= deadline:
            status["timedOut"] = True
            return status
        time.sleep(POLL_INTERVAL_SECS)


def summarize_results(status: dict) -> list[dict]:
    """Normalize per-platform results into [{platform, status, url, error}]."""
    raw = status.get("results") or []
    if isinstance(raw, dict):  # synchronous shape: {"tiktok": {...}}
        raw = [{"platform": p, **r} for p, r in raw.items()]
    out = []
    for r in raw:
        if r.get("skipped"):
            state = "skipped"
        elif r.get("status"):
            state = r["status"]
        else:
            state = "completed" if r.get("success") else "failed"
        platform = r.get("platform")
        post_id = r.get("platform_post_id") or r.get("video_id")
        raw_url = r.get("post_url") or r.get("url")
        url = raw_url if raw_url and str(raw_url).startswith("http") else None
        if not url and platform == "youtube" and post_id:
            url = f"https://www.youtube.com/watch?v={post_id}"  # also valid for private videos
        out.append({
            "platform": platform,
            "status": state,
            "url": url,
            "postId": post_id,
            # e.g. "Post uploaded as Private. No public URL available."
            "note": raw_url if raw_url and not url else None,
            "error": (r.get("error_message") or r.get("error")) if state != "completed" else None,
            "inbox": bool(r.get("fallback_to_inbox")),
        })
    return out


# ---------------------------------------------------------------------------
# Helpers / validation
# ---------------------------------------------------------------------------
def read_description(args) -> str:
    if args.description_file:
        if args.description_file == "-":
            return sys.stdin.read()
        return Path(args.description_file).read_text()
    return args.description or ""


def parse_schedule(value: str, tz: Optional[str]) -> str:
    """Validate an ISO8601 timestamp. With --timezone it stays local (the API
    interprets it in that zone); without, naive values are UTC."""
    raw = value.strip()
    parseable = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    dt = datetime.fromisoformat(parseable)  # raises ValueError on bad input
    if tz:
        return dt.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if dt <= datetime.now(timezone.utc):
        raise ValueError("in the past")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_args(args) -> Optional[str]:
    """Return an error string if invalid, else None. Normalizes args in place.
    Runs before any network call so a bad flag never costs an upload."""
    if not args.video:
        return "Missing --video (the file to upload)."
    vpath = Path(args.video)
    if not vpath.exists():
        return f"Video file not found: {args.video}"
    if vpath.stat().st_size == 0:
        return f"Video file is empty: {args.video}"
    if not args.title or not args.title.strip():
        return "Missing --title."
    if not args.user:
        return "Missing --user (the Upload-Post profile). Set UPLOAD_POST_USER in .env or pass --user."

    args.platforms = parse_platforms(args.platforms)
    if not args.platforms:
        return f"Missing --platforms. Choose from: {', '.join(VIDEO_PLATFORMS)}"
    unknown = [p for p in args.platforms if p not in VIDEO_PLATFORMS]
    if unknown:
        return f"Unsupported platform(s) for video: {', '.join(unknown)}. Choose from: {', '.join(VIDEO_PLATFORMS)}"
    if "pinterest" in args.platforms and not args.pinterest_board:
        return "Pinterest needs --pinterest-board BOARD_ID."

    for platform, limit in TITLE_MAX.items():
        if platform in args.platforms and len(args.title) > limit:
            return f"--title is {len(args.title)} chars; {platform} allows {limit}."

    if args.schedule:
        try:
            args.schedule = parse_schedule(args.schedule, args.timezone)
        except ValueError:
            return f"--schedule must be a future ISO8601 timestamp: {args.schedule}"
    if args.thumbnail and not Path(args.thumbnail).exists():
        return f"--thumbnail file not found: {args.thumbnail}"
    return None


def emit_json(payload: dict):
    print(json.dumps(payload))


def fail(msg: str, error_type: str, json_out: bool, **extra):
    log(msg, "error")
    if json_out:
        emit_json({"success": False, "error": msg, "errorType": error_type, **extra})
    sys.exit(1)


def report(status: dict, results: list[dict], json_out: bool, **extra) -> int:
    """Log per-platform outcomes, emit JSON, and return the exit code."""
    for r in results:
        if r["status"] == "completed":
            if r["inbox"]:
                where = "sent to TikTok inbox (open the app to publish)"
            else:
                where = r["url"] or r["note"] or f"published (post id {r['postId']})"
            log(f"{r['platform']}: {where}", "success")
        elif r["status"] == "skipped":
            log(f"{r['platform']}: skipped — no account connected to this profile", "warn")
        elif r["status"] in ("failed", "retryable"):
            log(f"{r['platform']}: {r['status']} — {r['error']}", "error")
        else:
            log(f"{r['platform']}: {r['status']}", "dim")

    state = status.get("status")
    published = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] in ("failed", "retryable")]
    ok = state != "not_found" and not failed and (bool(published) or state not in FINAL_STATUSES)

    if status.get("timedOut"):
        log(
            f"Still running after the wait window — the upload continues server-side. "
            f"Check later: uv run tools/upload_post.py --status {status.get('request_id')}",
            "warn",
        )
    if json_out:
        emit_json({
            "success": ok,
            "status": state,
            "requestId": status.get("request_id"),
            "jobId": status.get("job_id"),
            "results": results,
            "timedOut": bool(status.get("timedOut")),
            **extra,
        })
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a rendered video to TikTok, Instagram, YouTube, LinkedIn, Facebook, X, "
        "Threads, Pinterest and Bluesky via the Upload-Post API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video out/v.mp4 --title "T" --platforms tiktok,instagram,youtube --dry-run --json-out
  %(prog)s --video out/v.mp4 --title "T" --platforms tiktok,instagram,youtube --json-out
  %(prog)s --video out/v.mp4 --title "T" --platforms linkedin,x --schedule 2026-10-01T09:00:00 --timezone Europe/London
  %(prog)s --status REQUEST_OR_JOB_ID --json-out

Setup (API key + profile) is documented in docs/upload-post.md and .env.example.
        """,
    )

    parser.add_argument("--video", "--input", dest="video", help="Path to the video file to publish")
    parser.add_argument("--title", help="Caption/title used on every platform (YouTube: max 100 chars)")
    desc_group = parser.add_mutually_exclusive_group()
    desc_group.add_argument("--description", help="Longer text for YouTube, LinkedIn, Facebook and Pinterest")
    desc_group.add_argument("--description-file", help="Read the description from a file ('-' for stdin)")
    parser.add_argument(
        "--platforms",
        help=f"Comma-separated: {', '.join(VIDEO_PLATFORMS)} (aliases: twitter, reels, shorts)",
    )
    parser.add_argument(
        "--user", default=os.getenv("UPLOAD_POST_USER"),
        help="Upload-Post profile whose connected accounts to post from (default: $UPLOAD_POST_USER)",
    )

    parser.add_argument("--schedule", help="Publish later, ISO8601 e.g. 2026-10-01T09:00:00Z (max 365 days ahead)")
    parser.add_argument("--timezone", help="IANA zone for --schedule, e.g. Europe/London (default: UTC)")
    parser.add_argument("--first-comment", help="Post this as the first comment/reply after publishing")
    parser.add_argument(
        "--ai-generated", action="store_true",
        help="Disclose the video as AI-generated (TikTok, Instagram, YouTube and X labels)",
    )
    parser.add_argument("--thumbnail", help="Custom thumbnail image (YouTube and LinkedIn; YouTube max 2MB)")

    yt = parser.add_argument_group("YouTube")
    yt.add_argument("--youtube-privacy", choices=YOUTUBE_PRIVACY, default="private", help="Default: private")
    yt.add_argument("--youtube-category", default="22", help='Category ID (default "22"; "28" = Science & Tech)')
    yt.add_argument("--tags", help="Comma-separated YouTube tags")

    tt = parser.add_argument_group("TikTok")
    tt.add_argument("--tiktok-privacy", choices=TIKTOK_PRIVACY, help="Default: the account's own default")
    tt.add_argument("--tiktok-draft", action="store_true", help="Send to TikTok drafts instead of posting")

    other = parser.add_argument_group("Other platforms")
    other.add_argument("--instagram-story", action="store_true", help="Post to Instagram Stories instead of Reels")
    other.add_argument("--facebook-page-id", help="Facebook Page to post to (default: the profile's page)")
    other.add_argument("--linkedin-page-id", help="Post as a LinkedIn company page instead of the member")
    other.add_argument("--pinterest-board", help="Pinterest board ID (required for Pinterest)")

    parser.add_argument("--status", metavar="ID", help="Check an earlier upload by request_id or job_id, then exit")
    parser.add_argument("--no-wait", action="store_true", help="Return right after submitting instead of polling")
    parser.add_argument(
        "--wait-timeout", type=int, default=DEFAULT_WAIT_SECS,
        help=f"Seconds to poll for results (default {DEFAULT_WAIT_SECS})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate + check the account without posting")
    parser.add_argument("--json-out", action="store_true", help="Emit a single machine-readable JSON line to stdout")
    return parser


def main():
    args = build_parser().parse_args()

    from config import get_upload_post_api_key

    api_key = get_upload_post_api_key()
    if not api_key and not args.dry_run:
        fail(
            "UPLOAD_POST_API_KEY is not set. Add it to .env — see docs/upload-post.md.",
            "auth", args.json_out,
        )

    # --- Status lookup -----------------------------------------------------------
    if args.status:
        # Scheduled job_ids are 32 hex chars; request_ids (ours are UUIDs) usually
        # aren't. Ask the likely kind first so the fallback is the rare path.
        kinds = ["job_id", "request_id"] if re.fullmatch(r"[0-9a-f]{32}", args.status) else ["request_id", "job_id"]
        try:
            for kind in kinds:
                status = get_status(api_key, **{kind: args.status})
                if status.get("status") != "not_found":
                    break
        except ApiError as e:
            fail(str(e), e.error_type, args.json_out)
        if status.get("status") == "not_found":
            log(f"No upload found with id {args.status}.", "error")
        elif not status.get("job_id"):
            status.setdefault("request_id", args.status)
        sys.exit(report(status, summarize_results(status), args.json_out))

    # --- Validate ----------------------------------------------------------------
    err = validate_args(args)
    if err:
        fail(err, "validation", args.json_out)

    description = read_description(args)
    request_id = str(uuid.uuid4())
    form = build_form(args, description, request_id)

    # --- Dry run -----------------------------------------------------------------
    if args.dry_run:
        log("Dry run — form fields (no upload):", "info")
        print(json.dumps(form, indent=2), file=sys.stderr)
        auth_ok, auth_msg, connected, missing = False, None, [], []
        if not api_key:
            auth_msg = "UPLOAD_POST_API_KEY is not set"
        else:
            try:
                account = check_account(api_key, args.user)
                auth_ok, connected = True, account["connected"]
                missing = [p for p in args.platforms if p not in connected]
                log(f"Auth: key valid (plan: {account['plan']}); profile '{args.user}' found.", "success")
            except ApiError as e:
                auth_msg = str(e)
        if auth_msg:
            log(f"Auth: not ready — {auth_msg}", "warn")
        if missing:
            log(
                f"Not connected on profile '{args.user}': {', '.join(missing)} — these would be skipped. "
                "Connect them in the Upload-Post dashboard.",
                "warn",
            )
        if args.json_out:
            emit_json({
                "success": True, "dryRun": True, "authOk": auth_ok, "authError": auth_msg,
                "user": args.user, "platforms": args.platforms,
                "connectedPlatforms": connected, "missingPlatforms": missing,
                "form": {k: v for k, v in form if not k.endswith("[]")},
            })
        sys.exit(0)

    # --- Upload ------------------------------------------------------------------
    log(f"Uploading '{Path(args.video).name}' to {', '.join(args.platforms)} as '{args.user}'...", "info")
    try:
        submitted = submit_upload(api_key, args.video, form, args.thumbnail, request_id)
    except ApiError as e:
        fail(str(e), e.error_type, args.json_out, requestId=request_id)
    except requests.RequestException as e:
        # The server may have the file already. Don't resend — poll the same id.
        log(f"Network error while uploading ({e}); checking whether it arrived...", "warn")
        submitted = {"request_id": request_id}

    # Scheduled posts return a job_id and run later.
    if args.schedule:
        job_id = submitted.get("job_id")
        log(f"Scheduled for {args.schedule}{' ' + args.timezone if args.timezone else ''} (job {job_id}).", "success")
        log(f"Check it later: uv run tools/upload_post.py --status {request_id}", "dim")
        if args.json_out:
            emit_json({
                "success": True, "status": "scheduled", "jobId": job_id,
                "requestId": request_id, "scheduledDate": args.schedule,
                "timezone": args.timezone, "platforms": args.platforms,
            })
        sys.exit(0)

    if args.no_wait:
        log(f"Submitted. Check progress: uv run tools/upload_post.py --status {request_id}", "success")
        if args.json_out:
            emit_json({"success": True, "status": "submitted", "requestId": request_id, "platforms": args.platforms})
        sys.exit(0)

    try:
        status = wait_for_result(api_key, request_id, args.wait_timeout)
    except ApiError as e:
        fail(str(e), e.error_type, args.json_out, requestId=request_id)
    status.setdefault("request_id", request_id)
    sys.exit(report(status, summarize_results(status), args.json_out, platforms=args.platforms))


if __name__ == "__main__":
    main()
