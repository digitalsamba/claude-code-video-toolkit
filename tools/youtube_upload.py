#!/usr/bin/env python3
"""
Upload a rendered video to YouTube via the YouTube Data API v3.

YouTube uploads act on behalf of a channel, so they require OAuth 2.0 (NOT a plain
API key). The one-time setup is in Google Cloud Console; see docs/youtube-upload.md
and .env.example for the click-path. In short:
  1. Enable "YouTube Data API v3" on a Google Cloud project.
  2. Create an OAuth client ID of type "Desktop app", download client_secret_*.json.
  3. Point YOUTUBE_CLIENT_SECRETS_FILE at it (or pass --client-secrets).
  4. Log in once (opens a browser): uv run tools/youtube_upload.py --auth

After the first login the refresh token is cached under _internal/.youtube/ and
reused silently, so subsequent uploads need no browser.

Examples:
  # One-time login (interactive, opens a browser)
  uv run tools/youtube_upload.py --auth

  # Upload privately (the safe default)
  uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \\
      --description-file DESCRIPTION.md --tags "ai,agents,explainer"

  # Schedule a public go-live (only effective once the OAuth app is verified)
  uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \\
      --publish-at 2026-06-10T15:00:00Z --thumbnail out/thumb.png

  # Validate everything without uploading (also reports auth readiness)
  uv run tools/youtube_upload.py --video out/video.mp4 --title "Test" --dry-run --json-out

  # A second channel keeps its own cached token
  uv run tools/youtube_upload.py --auth --account work

# ---------------------------------------------------------------------------
# QUOTA & VERIFICATION REALITIES (YouTube Data API v3)
#   * Default quota: 10,000 units/day/project.
#   * videos.insert        ~1600 units  -> ~6 uploads/day on the default quota.
#   * thumbnails.set         ~50 units
#   * captions.insert       ~400 units
#   * playlistItems.insert   ~50 units
#   Hitting quota returns HTTP 403 'quotaExceeded' (NOT retriable). Request more
#   via the Google Cloud quota form if you need volume.
#
#   UNAUDITED-PROJECT PRIVATE LOCK: Google's docs state that videos uploaded via
#   videos.insert from unaudited API projects (created after 2020-07-28) are
#   restricted to private. In practice this targets THIRD-PARTY apps uploading to
#   OTHER users' channels — first-party uploads (your own Cloud project + your own
#   channel + your own account) generally are NOT affected and go public fine
#   (verified empirically). If it does bite, lift it via the YouTube API compliance
#   audit. 'Testing'-mode consent screens also expire refresh tokens after ~7 days.
#   This tool always reports the *actual* returned privacyStatus (the source of
#   truth), not just what was requested.
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import random
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import httplib2
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print(
        "Install with: pip install google-api-python-client google-auth-oauthlib "
        "google-auth-httplib2 python-dotenv"
    )
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

# OAuth scopes. youtube.upload covers videos.insert and thumbnails.set; youtube covers
# playlistItems.insert and videos.update. captions.insert accepts neither: it needs
# youtube.force-ssl (a 403 "insufficient scopes" otherwise). Request all three up front so
# a later step never forces a re-consent. A token from before force-ssl was added keeps
# working for uploads; only --captions asks for a new --auth.
CAPTION_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    CAPTION_SCOPE,
]

VALID_PRIVACY = ("private", "unlisted", "public")
RETRIABLE_STATUS = (500, 502, 503, 504)
MAX_RETRIES = 10
DEFAULT_CHUNK_MIB = 8

# Transport-layer errors worth retrying during a resumable upload.
RETRIABLE_EXCEPTIONS = (
    httplib2.HttpLib2Error,
    IOError,
    http.client.NotConnected,
    http.client.IncompleteRead,
    http.client.ImproperConnectionState,
    http.client.CannotSendRequest,
    http.client.CannotSendHeader,
    http.client.ResponseNotReady,
    http.client.BadStatusLine,
    socket.error,
    socket.timeout,
    ConnectionResetError,
    TimeoutError,
)

# Limits enforced by the API.
TITLE_MAX = 100
DESCRIPTION_MAX = 5000
TAGS_COMBINED_MAX = 500


class AuthError(Exception):
    """Authentication could not be established (no/invalid token, missing secrets)."""


class UploadError(Exception):
    """The resumable upload failed permanently."""


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
# Auth
# ---------------------------------------------------------------------------
def token_path_for(account: str) -> Path:
    """Per-account cached-token path. Account name is sanitized to avoid traversal."""
    from config import get_youtube_token_dir

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account) or "default"
    return get_youtube_token_dir() / f"token_{safe}.json"


def _save_token(token_file: Path, creds: "Credentials") -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json())
    try:
        os.chmod(token_file, 0o600)  # refresh token is a long-lived secret
    except OSError:
        pass


def get_credentials(
    account: str,
    client_secrets: Optional[str],
    *,
    allow_interactive: bool,
) -> "Credentials":
    """Load cached creds for `account`, refreshing silently. Run the browser flow only
    when allow_interactive=True (i.e. --auth, or an upload with --login-if-needed).

    Raises AuthError with an actionable message on any failure.
    """
    from config import get_youtube_client_secrets_file

    token_file = token_path_for(account)
    creds: Optional[Credentials] = None

    if token_file.exists():
        try:
            # no scopes argument: keep what was granted, so a refresh never asks for more
            creds = Credentials.from_authorized_user_file(str(token_file))
        except (ValueError, json.JSONDecodeError) as e:
            log(f"Cached token for '{account}' is unreadable ({e}); re-authing.", "warn")
            creds = None

    # Kept on a new consent, so logging in again here never drops a scope another tool asked for
    # (youtube_stats.py adds Analytics).
    granted = list(getattr(creds, "scopes", None) or [])

    # Logging in (--auth) with a token from before a scope was added: ask again for all of them.
    if creds and allow_interactive and missing_scopes(creds, SCOPES):
        log(f"Token for '{account}' lacks {', '.join(missing_scopes(creds, SCOPES))}; asking again.", "info")
        creds = None

    # Valid cached token -> zero interaction.
    if creds and creds.valid:
        return creds

    # Expired but refreshable -> refresh silently (the unattended path).
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(token_file, creds)
            return creds
        except RefreshError as e:
            if not allow_interactive:
                raise AuthError(
                    f"Refresh token for account '{account}' is invalid ({e}). "
                    f"Re-run interactively: uv run tools/youtube_upload.py --auth --account {account}"
                )
            creds = None  # fall through to a fresh consent flow

    # No usable token. Only the interactive path may open a browser.
    if not allow_interactive:
        raise AuthError(
            f"No cached credentials for account '{account}'. "
            f"Log in once interactively: uv run tools/youtube_upload.py --auth --account {account}"
        )

    secrets = client_secrets or get_youtube_client_secrets_file()
    if not secrets or not Path(secrets).exists():
        raise AuthError(
            "OAuth client secrets file not found. Set YOUTUBE_CLIENT_SECRETS_FILE in .env "
            "or pass --client-secrets PATH. See docs/youtube-upload.md for the Cloud Console steps."
        )

    flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES + [g for g in granted if g not in SCOPES])
    # port=0 -> OS-assigned free localhost port (matches the "Desktop app" client's
    # http://localhost redirect). access_type=offline + prompt=consent guarantee a
    # refresh_token is issued so future runs are silent.
    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message=(
            f"Authorizing YouTube account '{account}'. If a browser doesn't open automatically, "
            f"open this URL in any browser:\n\n{{url}}\n"
        ),
        success_message="Authorization complete. You can close this tab and return to the terminal.",
        open_browser=True,
    )
    _save_token(token_file, creds)
    return creds


def build_service(creds: "Credentials"):
    # cache_discovery=False silences the oauth2client cache warning and avoids disk writes.
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------
def parse_tags(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    # The API caps the combined length of all tags at 500 chars (plus quoting overhead
    # for multi-word tags). Drop overflow rather than letting the insert 400.
    kept: list[str] = []
    total = 0
    for t in tags:
        cost = len(t) + (2 if " " in t else 0)
        if total + cost > TAGS_COMBINED_MAX:
            log(f"Dropping tag '{t}' — exceeds the 500-char combined tag limit.", "warn")
            continue
        kept.append(t)
        total += cost
    return kept


def build_request_body(args, description: str, tags: list[str]) -> dict:
    title = args.title or ""
    if len(title) > TITLE_MAX:
        log(f"Title is {len(title)} chars; truncating to {TITLE_MAX}.", "warn")
        title = title[:TITLE_MAX]
    if len(description) > DESCRIPTION_MAX:
        log(f"Description is {len(description)} chars; truncating to {DESCRIPTION_MAX}.", "warn")
        description = description[:DESCRIPTION_MAX]

    snippet = {
        "title": title,
        "description": description,
        "categoryId": str(args.category),
    }
    if tags:
        snippet["tags"] = tags
    if args.default_language:
        snippet["defaultLanguage"] = args.default_language

    status = {
        "privacyStatus": args.privacy,
        "selfDeclaredMadeForKids": bool(args.made_for_kids),
        "license": args.license,
        "embeddable": True,
    }
    if args.synthetic_media:
        # YouTube's "altered or synthetic content" disclosure: required when realistic people, places or events
        # were generated or altered (an AI-animated cast, a cloned voice saying things the person didn't)
        status["containsSyntheticMedia"] = True
    if args.publish_at:
        # Scheduled publish REQUIRES privacyStatus == private at insert time.
        status["privacyStatus"] = "private"
        status["publishAt"] = args.publish_at

    return {"snippet": snippet, "status": status}


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# YouTube returns a machine-readable reason in the error body (e.g. "quotaExceeded",
# "accessNotConfigured", "forbidden"). A bare HTTP status is ambiguous — 403 covers
# a genuine quota wall, a disabled API, and a plain permission denial alike — so we
# read the reason to give the caller an accurate errorType instead of calling every
# 403 "quota".
_QUOTA_REASONS = {
    "quotaExceeded",
    "dailyLimitExceeded",
    "rateLimitExceeded",
    "userRateLimitExceeded",
}


def extract_api_reason(e: "HttpError") -> Optional[str]:
    """Pull the YouTube error 'reason' (e.g. 'accessNotConfigured') from an HttpError body."""
    content = getattr(e, "content", None)
    if content is None:
        return None
    try:
        if isinstance(content, (bytes, bytearray)):
            content = content.decode("utf-8", "replace")
        error = json.loads(content).get("error", {})
        errors = error.get("errors") or []
        if errors and errors[0].get("reason"):
            return errors[0]["reason"]
        return error.get("status")  # e.g. "PERMISSION_DENIED"
    except (ValueError, AttributeError, TypeError):
        return None


def classify_http_error(e: "HttpError") -> tuple[str, Optional[str]]:
    """Map an HttpError to (errorType, apiReason).

    errorType ∈ quota | config | forbidden | auth | http. The reason disambiguates
    same-status failures; we fall back to the status code when no reason is present.
    """
    status = getattr(e.resp, "status", None)
    reason = extract_api_reason(e)
    if reason in _QUOTA_REASONS:
        return "quota", reason
    if reason == "accessNotConfigured":
        return "config", reason
    if status == 401:
        return "auth", reason
    if status == 403:
        return "forbidden", reason
    return "http", reason


ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
APIS = {"data": "youtube.googleapis.com", "analytics": "youtubeanalytics.googleapis.com"}


def cloud_project_number(client_secrets: Optional[str]) -> Optional[str]:
    """The Cloud project number, from the OAuth client ID ('495036436430-abc.apps...' -> '495036436430')."""
    from config import get_youtube_client_secrets_file
    path = client_secrets or get_youtube_client_secrets_file()
    try:
        data = json.loads(Path(path).read_text())
        cid = (data.get("installed") or data.get("web") or {}).get("client_id", "")
    except (OSError, ValueError, TypeError):
        return None
    head = cid.split("-", 1)[0]
    return head if head.isdigit() else None


def enable_url(api: str, project: Optional[str]) -> str:
    """The Cloud Console page that switches `api` on (the same link Google puts in an accessNotConfigured error)."""
    url = f"https://console.developers.google.com/apis/api/{APIS[api]}/overview"
    return f"{url}?project={project}" if project else url


def check_setup(account: str, client_secrets: Optional[str]) -> dict:
    """Read-only: is the login there, which channel is it, which scopes does it have, and are the Data and
    Analytics APIs switched on? Costs about 1 quota unit (channels.list); nothing is written."""
    project = cloud_project_number(client_secrets)
    out: dict = {"account": account, "cloudProject": project, "ok": False, "fixes": []}
    try:
        creds = get_credentials(account, client_secrets, allow_interactive=False)
    except AuthError as e:
        out["auth"] = str(e)
        out["fixes"].append(f"log in: uv run --extra youtube tools/youtube_upload.py --auth --account {account}")
        return out
    granted = list(getattr(creds, "scopes", None) or [])
    out["scopes"] = granted
    out["missingScopes"] = missing_scopes(creds, SCOPES)
    if out["missingScopes"]:
        out["fixes"].append(f"log in again for {', '.join(s.rsplit('/', 1)[-1] for s in out['missingScopes'])}: "
                            f"uv run --extra youtube tools/youtube_upload.py --auth --account {account}")

    try:
        items = build_service(creds).channels().list(part="snippet", mine=True).execute().get("items", [])
        out["dataApi"] = "on"
        out["channel"] = ({"id": items[0]["id"], "title": items[0]["snippet"]["title"]} if items else None)
        if not items:
            out["fixes"].append("this Google account has no YouTube channel: create one, or log in with the right one")
    except HttpError as e:
        etype, reason = classify_http_error(e)
        out["dataApi"] = "off" if etype == "config" else f"error: {reason or getattr(e.resp, 'status', '')}"
        if etype == "config":
            out["fixes"].append(f"switch on the YouTube Data API v3: {enable_url('data', project)}")

    if ANALYTICS_SCOPE in granted:
        try:
            import datetime as dt
            day = dt.date.today()
            build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False).reports().query(
                ids="channel==MINE", startDate=str(day - dt.timedelta(days=7)), endDate=str(day), metrics="views",
            ).execute()
            out["analyticsApi"] = "on"
        except HttpError as e:
            etype, reason = classify_http_error(e)
            out["analyticsApi"] = "off" if etype == "config" else f"error: {reason or getattr(e.resp, 'status', '')}"
            if etype == "config":
                out["fixes"].append(f"switch on the YouTube Analytics API: {enable_url('analytics', project)}")
    else:
        out["analyticsApi"] = "no scope"   # optional: only youtube_stats.py needs it

    out["ok"] = out["dataApi"] == "on" and not out["missingScopes"] and bool(out.get("channel"))
    return out


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def resumable_upload(request) -> dict:
    """Drive request.next_chunk() to completion with capped exponential backoff.

    Returns the inserted video resource dict. Raises UploadError on permanent failure
    and re-raises non-retriable HttpError (4xx) immediately.
    """
    response = None
    error: Optional[str] = None
    retry = 0
    last_pct = -1

    while response is None:
        try:
            status, response = request.next_chunk()
            if status is not None:
                pct = int(status.progress() * 100)
                if pct != last_pct:
                    log(f"Uploading... {pct}%", "dim")
                    last_pct = pct
            if response is not None:
                if "id" in response:
                    return response
                raise UploadError(f"Upload finished with an unexpected response: {response}")
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS:
                error = f"Retriable HTTP {e.resp.status}: {e}"
            else:
                raise  # 4xx (bad body / quota / auth) is permanent — surface it now.
        except RETRIABLE_EXCEPTIONS as e:
            error = f"Retriable transport error: {e}"

        if error is not None:
            retry += 1
            if retry > MAX_RETRIES:
                raise UploadError(f"Giving up after {MAX_RETRIES} retries. Last error: {error}")
            sleep_secs = min(2 ** retry, 64) + random.random()
            log(f"{error} — retry {retry}/{MAX_RETRIES} in {sleep_secs:.1f}s", "warn")
            time.sleep(sleep_secs)
            error = None

    return response


def locked_private(args, status: dict) -> Optional[str]:
    """A warning when YouTube kept a video private that was meant to go public, else None.

    A scheduled upload is private until its publishAt by design, so for one the lock shows as YouTube dropping
    the publishAt, not as privacy=private."""
    lock = ("likely the unaudited-project lock. For your own channel this usually doesn't apply; "
            "otherwise lift it via the YouTube API compliance audit.")
    if args.publish_at:
        if status.get("privacyStatus") == "private" and not status.get("publishAt"):
            return f"YouTube dropped the scheduled publish time, so the video stays private: {lock}"
        return None
    if args.privacy == "public" and status.get("privacyStatus") == "private":
        return f"YouTube returned privacy=private for a public upload: {lock}"
    return None


def set_synthetic_media(youtube, video_id: str, status: dict) -> bool:
    """Set the "altered or synthetic content" disclosure after upload. videos.update replaces the whole status
    part, so the fields the insert set are sent back with it."""
    keep = {k: status[k] for k in ("privacyStatus", "publishAt", "license", "embeddable", "publicStatsViewable",
                                   "selfDeclaredMadeForKids") if k in status}
    try:
        r = youtube.videos().update(part="status", body={"id": video_id, "status": {
            **keep, "containsSyntheticMedia": True}}).execute()
        ok = bool(r.get("status", {}).get("containsSyntheticMedia"))
        log("AI-generated content disclosure set" if ok else "AI disclosure not confirmed: set it in YouTube Studio",
            "success" if ok else "warn")
        return ok
    except Exception as e:  # non-fatal: the video is up
        log(f"AI disclosure not set ({e}); set it in YouTube Studio (Details > Altered content)", "warn")
        return False


def set_thumbnail(youtube, video_id: str, thumb_path: str) -> bool:
    """Attach a custom thumbnail. Non-fatal: returns True/False, never raises out."""
    try:
        mime = mimetypes.guess_type(thumb_path)[0] or "image/png"
        media = MediaFileUpload(thumb_path, mimetype=mime, resumable=False)
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        log(f"Thumbnail set ({thumb_path})", "success")
        return True
    except HttpError as e:
        log(
            f"Thumbnail not set ({e}). Custom thumbnails require a verified-phone channel; "
            "the video itself uploaded fine.",
            "warn",
        )
        return False


def missing_scopes(creds, needed: list[str]) -> list[str]:
    """Scopes in `needed` that the cached token was not granted."""
    have = set(getattr(creds, "scopes", None) or [])
    return [s for s in needed if s not in have]


def needed_scopes(args) -> list[str]:
    return [CAPTION_SCOPE] if args.captions else []


def add_caption(youtube, video_id: str, caption_path: str, language: str) -> bool:
    """Insert a caption track from an .srt/.vtt file. Non-fatal."""
    try:
        media = MediaFileUpload(caption_path, mimetype="application/octet-stream", resumable=False)
        body = {
            "snippet": {
                "videoId": video_id,
                "language": language,
                "name": "",
                "isDraft": False,
            }
        }
        youtube.captions().insert(part="snippet", body=body, media_body=media).execute()
        log(f"Caption track added ({language})", "success")
        return True
    except HttpError as e:
        log(f"Caption not added ({e}); the video itself uploaded fine.", "warn")
        return False


def add_to_playlist(youtube, video_id: str, playlist_id: str) -> bool:
    """Add the uploaded video to a playlist. Non-fatal."""
    try:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()
        log(f"Added to playlist {playlist_id}", "success")
        return True
    except HttpError as e:
        log(f"Playlist add failed ({e}); the video itself uploaded fine.", "warn")
        return False


# ---------------------------------------------------------------------------
# Helpers / validation
# ---------------------------------------------------------------------------
def read_description(args) -> str:
    if args.description_file:
        if args.description_file == "-":
            return sys.stdin.read()
        return Path(args.description_file).read_text()
    return args.description or ""


def parse_publish_at(value: str) -> str:
    """Validate an ISO8601/RFC3339 timestamp and normalize to UTC 'Z' form.

    Accepts a trailing 'Z' or an explicit offset. Naive timestamps are assumed UTC.
    """
    raw = value.strip()
    parseable = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    dt = datetime.fromisoformat(parseable)  # raises ValueError on bad input
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if dt <= datetime.now(timezone.utc):
        log("--publish-at is in the past; YouTube may publish immediately.", "warn")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_args(args) -> Optional[str]:
    """Return an error string if invalid, else None. Mutates args.publish_at to the
    normalized form. Runs before any network call so we never burn quota on bad input."""
    if args.video_id:
        if args.video:
            return "--video-id works on an uploaded video; leave out --video."
        if not (args.thumbnail or args.captions or args.playlist):
            return "--video-id needs something to add: --captions, --thumbnail or --playlist."
        for label, p in (("--thumbnail", args.thumbnail), ("--captions", args.captions)):
            if p and not Path(p).exists():
                return f"{label} file not found: {p}"
        return None
    if not args.video:
        return "Missing --video (the file to upload)."
    vpath = Path(args.video)
    if not vpath.exists():
        return f"Video file not found: {args.video}"
    if vpath.stat().st_size == 0:
        return f"Video file is empty: {args.video}"
    if not args.title or not args.title.strip():
        return "Missing --title."
    if args.privacy not in VALID_PRIVACY:
        return f"--privacy must be one of {VALID_PRIVACY}"
    if args.publish_at:
        try:
            args.publish_at = parse_publish_at(args.publish_at)
        except ValueError:
            return f"--publish-at is not a valid ISO8601 timestamp: {args.publish_at}"
    for label, p in (("--thumbnail", args.thumbnail), ("--captions", args.captions)):
        if p and not Path(p).exists():
            return f"{label} file not found: {p}"
    return None


def add_extras_to_existing(youtube, args):
    """--video-id: thumbnail, captions and playlist for a video already on YouTube. Exits."""
    vid = args.video_id
    thumbnail_set = set_thumbnail(youtube, vid, args.thumbnail) if args.thumbnail else False
    captions_added = add_caption(youtube, vid, args.captions, args.captions_language) if args.captions else False
    playlist_added = add_to_playlist(youtube, vid, args.playlist) if args.playlist else False
    ok = all(done for asked, done in ((args.thumbnail, thumbnail_set), (args.captions, captions_added),
                                      (args.playlist, playlist_added)) if asked)
    if args.json_out:
        emit_json({
            "success": ok, "videoId": vid, "url": f"https://www.youtube.com/watch?v={vid}",
            "account": args.account, "thumbnailSet": thumbnail_set, "captionsAdded": captions_added,
            "playlistId": args.playlist if playlist_added else None, "uploadStatus": "existing",
        })
    sys.exit(0 if ok else 1)


def emit_json(payload: dict):
    print(json.dumps(payload))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a rendered video to YouTube (Data API v3, OAuth 2.0, resumable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --auth                                  # one-time browser login, caches a token
  %(prog)s --video out/v.mp4 --title "T" --tags "a,b"     # upload private (default)
  %(prog)s --video out/v.mp4 --title "T" --publish-at 2026-06-10T15:00:00Z --thumbnail t.png
  %(prog)s --video out/v.mp4 --title "T" --dry-run --json-out   # validate, no upload

Headless servers: run --auth once on a machine with a browser, then copy the cached
token (default _internal/.youtube/token_<account>.json) to the server.

First-time setup (Google Cloud Console) is documented in docs/youtube-upload.md and .env.example.
Note: until your OAuth app is Google-verified, uploads are force-locked to private.
        """,
    )

    parser.add_argument("--video", "--input", dest="video", help="Path to the video file to upload")
    parser.add_argument("--title", help="Video title (max 100 chars)")

    desc_group = parser.add_mutually_exclusive_group()
    desc_group.add_argument("--description", help="Video description text")
    desc_group.add_argument(
        "--description-file", help="Read description from a file ('-' for stdin)"
    )

    parser.add_argument("--tags", help="Comma-separated tags (combined <= 500 chars)")
    parser.add_argument(
        "--category", default="22",
        help='Numeric category ID as a string (default "22" = People & Blogs; "28" = Science & Tech)',
    )
    parser.add_argument(
        "--privacy", choices=VALID_PRIVACY, default="private",
        help="Visibility at insert (default: private). --publish-at forces private at insert.",
    )
    parser.add_argument(
        "--publish-at",
        help="Schedule public go-live, ISO8601/RFC3339 UTC e.g. 2026-06-10T15:00:00Z (forces private at insert)",
    )
    parser.add_argument("--thumbnail", help="Path to a custom thumbnail image (<=2MB, 1280x720)")
    parser.add_argument("--captions", help="Path to a caption file (.srt/.vtt)")
    parser.add_argument("--captions-language", default="en", help="Caption language code (default: en)")
    parser.add_argument("--playlist", help="Playlist ID to add the video to")
    parser.add_argument("--video-id", help="Add --captions, --thumbnail or --playlist to an already uploaded "
                        "video instead of uploading one")
    parser.add_argument("--made-for-kids", action="store_true", help="Set selfDeclaredMadeForKids")
    parser.add_argument("--synthetic-media", action="store_true",
                        help="Disclose realistic AI-generated or altered content (status.containsSyntheticMedia): "
                             "an animated AI cast, generated people, a cloned voice")
    parser.add_argument("--allow-copy", action="store_true",
                        help="Upload even if the title, description or tags fail tools/copy_lint.py "
                             "(dashes, stock AI or marketing phrases)")
    parser.add_argument("--default-language", help="snippet.defaultLanguage code, e.g. en")
    parser.add_argument(
        "--license", choices=("youtube", "creativeCommon"), default="youtube",
        help="Video license (default: youtube)",
    )

    parser.add_argument("--account", default="default", help="Channel/account name to namespace the cached token")
    parser.add_argument("--client-secrets", help="Path to the OAuth client_secret JSON (overrides env var)")
    parser.add_argument("--auth", action="store_true", help="Interactive login only: cache a token and exit")
    parser.add_argument("--check", action="store_true",
                        help="Check the setup and exit: login, channel, scopes, and whether the Data and Analytics "
                             "APIs are on (read-only, ~1 quota unit)")
    parser.add_argument(
        "--login-if-needed", action="store_true",
        help="Allow an upload run to open a browser for first-time consent (default: fail with instructions)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate + print the request body without uploading")
    parser.add_argument("--json-out", action="store_true", help="Emit a single machine-readable JSON line to stdout")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_MIB, help=f"Upload chunk size in MiB (default {DEFAULT_CHUNK_MIB})"
    )
    return parser


def main():
    args = build_parser().parse_args()

    # --- Interactive login only -------------------------------------------------
    if args.auth:
        try:
            get_credentials(args.account, args.client_secrets, allow_interactive=True)
        except AuthError as e:
            log(str(e), "error")
            if args.json_out:
                emit_json({"success": False, "error": str(e), "errorType": "auth"})
            sys.exit(1)
        log(f"Authorized account '{args.account}' — token cached at {token_path_for(args.account)}", "success")
        if args.json_out:
            emit_json({"success": True, "account": args.account, "action": "auth"})
        sys.exit(0)

    # --- Setup check ---------------------------------------------------------------
    if args.check:
        r = check_setup(args.account, args.client_secrets)
        ch = r.get("channel")
        log(f"Account '{args.account}': " + (f"channel '{ch['title']}' ({ch['id']})" if ch else
                                            r.get("auth") or "no channel"), "success" if ch else "error")
        if "scopes" in r:
            log(f"Scopes: {', '.join(x.rsplit('/', 1)[-1] for x in r['scopes'])}", "info")
            log(f"Data API: {r.get('dataApi')}; Analytics API: {r.get('analyticsApi')}", "info")
        for fix in r["fixes"]:
            log(f"Fix: {fix}", "warn")
        if r["ok"]:
            log("Ready to upload.", "success")
        if args.json_out:
            emit_json({"success": r["ok"], "action": "check", **r})
        sys.exit(0 if r["ok"] else 1)

    # --- Validate ----------------------------------------------------------------
    err = validate_args(args)
    if err:
        log(err, "error")
        if args.json_out:
            emit_json({"success": False, "error": err, "errorType": "validation"})
        sys.exit(1)

    body = None
    if not args.video_id:  # adding to an uploaded video sends no title or description
        description = read_description(args)
        tags = parse_tags(args.tags)

        # --- Copy lint: the title, description and tags are public ------------------
        if not args.allow_copy:
            from copy_lint import lint_text
            found = [f"{where}: '{m}' is {why}" for where, text in
                     (("title", args.title), ("description", description), ("tags", ", ".join(tags)))
                     for m, why in lint_text(text)]
            if found:
                msg = "Copy lint refused the upload (fix the wording, or --allow-copy if it's meant):\n  " + \
                      "\n  ".join(found)
                log(msg, "error")
                if args.json_out:
                    emit_json({"success": False, "error": msg, "errorType": "copy", "findings": found})
                sys.exit(1)
        body = build_request_body(args, description, tags)

    # --- Dry run -----------------------------------------------------------------
    if args.dry_run:
        if body is not None:
            log("Dry run — request body (no upload):", "info")
            print(json.dumps(body, indent=2), file=sys.stderr)
        auth_ok = False
        auth_msg = None
        missing = []
        try:
            creds = get_credentials(args.account, args.client_secrets, allow_interactive=False)
            missing = missing_scopes(creds, needed_scopes(args))
            if missing:
                auth_msg = f"token lacks {', '.join(missing)} (needed for --captions); re-run --auth"
                log(f"Auth: {auth_msg}", "warn")
            else:
                auth_ok = True
                log("Auth: cached credentials ready.", "success")
        except AuthError as e:
            auth_msg = str(e)
            log(f"Auth: not ready — {e}", "warn")
        if args.json_out:
            emit_json({
                "success": True, "dryRun": True, "authOk": auth_ok, "authError": auth_msg,
                "missingScopes": missing, "requestBody": body, "account": args.account,
            })
        sys.exit(0)

    # --- Note the possible unaudited-project private lock ------------------------
    if args.privacy == "public" or args.publish_at:
        log(
            "Requested public/scheduled visibility. Unaudited API projects MAY have public "
            "uploads force-locked to private (mainly affects uploads to OTHER users' channels; "
            "first-party uploads to your own channel usually go public). Check the actual "
            "privacy reported below.",
            "warn",
        )

    # --- Auth --------------------------------------------------------------------
    try:
        creds = get_credentials(args.account, args.client_secrets, allow_interactive=args.login_if_needed)
    except AuthError as e:
        log(str(e), "error")
        if args.json_out:
            emit_json({"success": False, "error": str(e), "errorType": "auth"})
        sys.exit(1)

    missing = missing_scopes(creds, needed_scopes(args))
    if missing:
        msg = (f"The cached token for '{args.account}' can't add captions (it lacks {', '.join(missing)}). "
               f"Log in again once, then re-run: uv run --extra youtube tools/youtube_upload.py --auth "
               f"--account {args.account}")
        log(msg, "error")
        if args.json_out:
            emit_json({"success": False, "error": msg, "errorType": "auth", "missingScopes": missing})
        sys.exit(1)

    youtube = build_service(creds)

    if args.video_id:
        add_extras_to_existing(youtube, args)

    # --- Upload ------------------------------------------------------------------
    media = MediaFileUpload(args.video, chunksize=args.chunk_size * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    log(f"Uploading '{Path(args.video).name}' to account '{args.account}'...", "info")
    try:
        response = resumable_upload(request)
    except HttpError as e:
        status = getattr(e.resp, "status", None)
        etype, reason = classify_http_error(e)
        msg = f"HTTP {status}: {e}"
        log(msg, "error")
        if etype == "config":
            log(
                "The YouTube Data API v3 isn't enabled on this Cloud project (or was just "
                "enabled and is still propagating). Enable it, wait 1-2 min, and retry — "
                "see docs/youtube-upload.md.",
                "warn",
            )
        elif etype == "quota":
            log("Daily API quota exhausted (~6 uploads/day on the default quota). Wait or request more.", "warn")
        if args.json_out:
            emit_json({
                "success": False, "error": msg,
                "errorType": etype, "errorReason": reason, "videoId": None,
            })
        sys.exit(1)
    except UploadError as e:
        log(str(e), "error")
        if args.json_out:
            emit_json({"success": False, "error": str(e), "errorType": "upload", "videoId": None})
        sys.exit(1)

    video_id = response["id"]
    actual_privacy = response.get("status", {}).get("privacyStatus", "unknown")
    url = f"https://www.youtube.com/watch?v={video_id}"
    log(f"Uploaded: {url}  (privacy: {actual_privacy})", "success")
    lock_warning = locked_private(args, response.get("status", {}))
    if lock_warning:
        log(lock_warning, "warn")
    elif args.publish_at:
        log(f"Scheduled: private until {args.publish_at}, then public.", "info")

    # --- Post-upload extras (each non-fatal) -------------------------------------
    synthetic = bool(response.get("status", {}).get("containsSyntheticMedia"))
    if args.synthetic_media and not synthetic:
        # videos.insert didn't keep the disclosure on the first uploads (2026-11-08); videos.update does
        synthetic = set_synthetic_media(youtube, video_id, response.get("status", {}))
    thumbnail_set = set_thumbnail(youtube, video_id, args.thumbnail) if args.thumbnail else False
    captions_added = (
        add_caption(youtube, video_id, args.captions, args.captions_language) if args.captions else False
    )
    if args.playlist:
        add_to_playlist(youtube, video_id, args.playlist)

    if args.json_out:
        emit_json({
            "success": True,
            "videoId": video_id,
            "url": url,
            "privacyStatus": actual_privacy,
            "requestedPrivacy": args.privacy,
            "publishAt": args.publish_at,
            "account": args.account,
            "thumbnailSet": thumbnail_set,
            "syntheticMedia": synthetic,
            "captionsAdded": captions_added,
            "playlistId": args.playlist,
            "uploadStatus": "uploaded",
        })

    sys.exit(0)


if __name__ == "__main__":
    main()
