#!/usr/bin/env python3
"""
How the channel's videos are doing: views, likes and comments per video (Data API v3), with watch time,
retention and traffic sources when the token also has the YouTube Analytics scope.

Every run saves a snapshot under _internal/.youtube/stats/<account>/ (gitignored), and the report shows the
change since the previous snapshot, so running it daily gives a trend without a database.

Uses the token youtube_upload.py cached for the account. Data API stats need nothing new. Analytics needs the
yt-analytics.readonly scope: enable the YouTube Analytics API in the Cloud project, then log in once more with
    uv run tools/youtube_stats.py --auth --account personal
which asks for the upload scopes and Analytics together, so the uploader keeps working with the new token.

Examples:
  uv run tools/youtube_stats.py --account personal                    # every upload, latest first
  uv run tools/youtube_stats.py --account personal --video 1hyDgC5Qg_o --since 7d --json
  uv run tools/youtube_stats.py --account personal --github           # plus the repo's traffic referrers

Needs: uv sync --extra youtube
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import youtube_upload as yu  # noqa: E402  (token paths, upload scopes, client secrets)

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
REPO = "digitalsamba/claude-code-video-toolkit"
CORE_METRICS = ["views", "estimatedMinutesWatched", "averageViewDuration", "averageViewPercentage", "likes",
                "comments", "shares", "subscribersGained"]
# Listed in the metrics reference, but older reports say the API refuses them; asked for separately.
REACH_METRICS = ["videoThumbnailImpressions", "videoThumbnailImpressionsClickRate"]


# ---------------------------------------------------------------------------
# auth

def load_credentials(account: str, *, interactive: bool, client_secrets: Optional[str] = None):
    """The cached token with whatever scopes it was granted. interactive=True runs a new consent for the upload
    scopes plus Analytics and saves it over the same file."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file = yu.token_path_for(account)
    if interactive:
        from config import get_youtube_client_secrets_file
        secrets = client_secrets or get_youtube_client_secrets_file()
        if not secrets or not Path(secrets).exists():
            raise yu.AuthError("OAuth client secrets not found: set YOUTUBE_CLIENT_SECRETS_FILE in .env")
        flow = InstalledAppFlow.from_client_secrets_file(secrets, yu.SCOPES + [ANALYTICS_SCOPE])
        creds = flow.run_local_server(port=0, access_type="offline", prompt="consent", open_browser=True,
                                      success_message="Authorized. You can close this tab.")
        yu._save_token(token_file, creds)
        return creds
    if not token_file.exists():
        raise yu.AuthError(f"No cached token for '{account}'. Log in: uv run tools/youtube_stats.py --auth "
                           f"--account {account}")
    data = json.loads(token_file.read_text())
    creds = Credentials.from_authorized_user_info(data)  # no scopes argument: keep what was granted
    if not creds.valid:
        creds.refresh(Request())
        yu._save_token(token_file, creds)
    return creds


def granted(creds) -> set[str]:
    return set(creds.scopes or [])


# ---------------------------------------------------------------------------
# Data API

def chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def channel(yt) -> dict:
    r = yt.channels().list(part="snippet,statistics,contentDetails", mine=True).execute()
    if not r.get("items"):
        raise SystemExit("Error: this token has no channel")
    c = r["items"][0]
    s = c["statistics"]
    return {"id": c["id"], "title": c["snippet"]["title"],
            "subscribers": int(s.get("subscriberCount", 0)), "views": int(s.get("viewCount", 0)),
            "videos": int(s.get("videoCount", 0)),
            "uploads": c["contentDetails"]["relatedPlaylists"]["uploads"]}


def upload_ids(yt, playlist: str, limit: int) -> list[str]:
    ids, token = [], None
    while len(ids) < limit:
        r = yt.playlistItems().list(part="contentDetails", playlistId=playlist, maxResults=50,
                                    pageToken=token).execute()
        ids += [i["contentDetails"]["videoId"] for i in r.get("items", [])]
        token = r.get("nextPageToken")
        if not token:
            break
    return ids[:limit]


def iso_seconds(d: str) -> int:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0
    days, h, mi, s = (int(x or 0) for x in m.groups())
    return days * 86400 + h * 3600 + mi * 60 + s


def videos(yt, ids: list[str]) -> list[dict]:
    out = []
    for batch in chunks(ids, 50):
        r = yt.videos().list(part="snippet,statistics,status,contentDetails", id=",".join(batch)).execute()
        for v in r.get("items", []):
            s, sn, st = v.get("statistics", {}), v["snippet"], v["status"]
            secs = iso_seconds(v["contentDetails"].get("duration", ""))
            out.append({
                "id": v["id"], "title": sn["title"], "published": sn["publishedAt"],
                "privacy": st.get("privacyStatus"), "syntheticMedia": st.get("containsSyntheticMedia"),
                "seconds": secs, "upToThreeMin": secs <= 180,  # a Short if vertical or square (the API has no aspect)
                "views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                "comments": int(s.get("commentCount", 0)),
            })
    out.sort(key=lambda v: v["published"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Analytics API

def parse_since(s: str) -> dt.date:
    m = re.fullmatch(r"(\d+)d", s)
    if m:
        return dt.date.today() - dt.timedelta(days=int(m.group(1)))
    return dt.date.fromisoformat(s)


def _rows(resp: dict) -> list[dict]:
    names = [h["name"] for h in resp.get("columnHeaders", [])]
    return [dict(zip(names, r)) for r in resp.get("rows", []) or []]


def analytics(creds, ids: list[str], start: dt.date, end: dt.date) -> dict:
    """Per-video analytics for [start, end]. Reach metrics and traffic sources are best effort: a refusal is
    recorded, not raised."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    ya = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
    q = dict(ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(), dimensions="video")
    result: dict = {"window": [start.isoformat(), end.isoformat()], "videos": {}, "notes": []}

    for batch in chunks(ids, 200):
        f = "video==" + ",".join(batch)
        for row in _rows(ya.reports().query(metrics=",".join(CORE_METRICS), filters=f, **q).execute()):
            result["videos"].setdefault(row.pop("video"), {}).update(row)
        try:
            for row in _rows(ya.reports().query(metrics=",".join(REACH_METRICS), filters=f, **q).execute()):
                result["videos"].setdefault(row.pop("video"), {}).update(row)
        except HttpError as e:
            note = f"impressions/CTR unavailable from the API: {yu.extract_api_reason(e) or e.resp.status}"
            if note not in result["notes"]:
                result["notes"].append(note)

    for vid in ids:
        try:
            r = ya.reports().query(ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(),
                                   dimensions="insightTrafficSourceType", metrics="views",
                                   filters=f"video=={vid}", sort="-views").execute()
            src = {row["insightTrafficSourceType"]: row["views"] for row in _rows(r)}
            if src:
                result["videos"].setdefault(vid, {})["traffic"] = src
        except HttpError as e:
            result["notes"].append(f"traffic sources for {vid}: {yu.extract_api_reason(e) or e.resp.status}")
    return result


# ---------------------------------------------------------------------------
# GitHub traffic (does YouTube send people to the repo?)

def github_traffic(repo: str = REPO) -> dict:
    def gh(path: str):
        r = subprocess.run(["gh", "api", f"repos/{repo}/traffic/{path}"], capture_output=True, text=True)
        return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()[:200]}
    views = gh("views")
    return {"referrers": gh("popular/referrers"),
            "views": {k: views.get(k) for k in ("count", "uniques")} if "error" not in views else views}


# ---------------------------------------------------------------------------
# snapshots

def stats_dir(account: str) -> Path:
    from config import get_youtube_token_dir
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account) or "default"
    return get_youtube_token_dir() / "stats" / safe


def previous_snapshot(d: Path) -> Optional[dict]:
    snaps = sorted(d.glob("*.json"))
    return json.loads(snaps[-1].read_text()) if snaps else None


def add_deltas(report: dict, prev: Optional[dict]) -> None:
    if not prev:
        return
    old = {v["id"]: v for v in prev.get("videos", [])}
    report["since"] = prev.get("taken")
    for v in report["videos"]:
        o = old.get(v["id"])
        if o:
            v["delta"] = {k: v[k] - o.get(k, 0) for k in ("views", "likes", "comments")}
    pc = prev.get("channel", {})
    report["channel"]["subscribersDelta"] = report["channel"]["subscribers"] - pc.get("subscribers", 0)


def save_snapshot(report: dict, d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / (report["taken"].replace(":", "").replace("-", "") + ".json")
    p.write_text(json.dumps(report, indent=2))
    return p


# ---------------------------------------------------------------------------
# report

def text_report(r: dict) -> str:
    c = r["channel"]
    d = c.get("subscribersDelta")
    lines = [f"{c['title']}: {c['subscribers']} subscribers" + (f" ({d:+d} since {r['since'][:16]})" if d is not None
             and r.get("since") else "") + f", {c['views']} views, {c['videos']} videos", ""]
    an = (r.get("analytics") or {}).get("videos", {})
    for v in r["videos"]:
        delta = v.get("delta")
        dtxt = f"  ({delta['views']:+d} views)" if delta else ""
        kind = "≤3 min" if v["upToThreeMin"] else f"{v['seconds'] // 60}:{v['seconds'] % 60:02d}"
        lines.append(f"{v['id']}  {v['title'][:60]}  [{kind}, {v['privacy']}]")
        lines.append(f"    {v['views']} views, {v['likes']} likes, {v['comments']} comments{dtxt}")
        a = an.get(v["id"])
        if a:
            bits = [f"avg {a.get('averageViewDuration', 0):.0f}s watched ({a.get('averageViewPercentage', 0):.0f}%)"]
            if "videoThumbnailImpressions" in a:
                bits.append(f"{a['videoThumbnailImpressions']} impressions, "
                            f"{a.get('videoThumbnailImpressionsClickRate', 0):.1f}% CTR")
            if a.get("traffic"):
                top = sorted(a["traffic"].items(), key=lambda kv: -kv[1])[:3]
                bits.append("from " + ", ".join(f"{k.lower()} {n}" for k, n in top))
            lines.append("    " + "; ".join(bits))
    if r.get("analytics"):
        w = r["analytics"]["window"]
        lines += ["", f"Analytics window {w[0]} to {w[1]} (YouTube lags 1-2 days)."]
        lines += [f"Note: {n}" for n in r["analytics"]["notes"]]
    elif r.get("analyticsNote"):
        lines += ["", r["analyticsNote"]]
    gh = r.get("github")
    if gh:
        refs = gh["referrers"]
        lines += ["", f"GitHub, last 14 days: {gh['views'].get('count')} views, {gh['views'].get('uniques')} unique"]
        if isinstance(refs, list):
            lines += [f"    {x['referrer']}: {x['count']} ({x['uniques']} unique)" for x in refs[:8]]
            yt = [x for x in refs if "youtube" in x["referrer"].lower()]
            lines.append("    YouTube sent " + (f"{sum(x['uniques'] for x in yt)} unique visitors" if yt else "nobody yet"))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account", default="default", help="the account name used with youtube_upload.py --auth")
    p.add_argument("--video", nargs="*", default=None, help="video IDs (default: the channel's uploads)")
    p.add_argument("--limit", type=int, default=50, help="most recent uploads to include (default 50)")
    p.add_argument("--since", default="28d", help="analytics window start: 7d, 28d or YYYY-MM-DD (default 28d)")
    p.add_argument("--github", action="store_true", help="add the repo's traffic views and referrers (needs gh)")
    p.add_argument("--no-save", action="store_true", help="don't write a snapshot")
    p.add_argument("--auth", action="store_true", help="log in again asking for upload + Analytics scopes")
    p.add_argument("--client-secrets", help="OAuth client secrets JSON (default: YOUTUBE_CLIENT_SECRETS_FILE)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    a = p.parse_args()

    try:
        creds = load_credentials(a.account, interactive=a.auth, client_secrets=a.client_secrets)
    except yu.AuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if a.auth:
        has = ANALYTICS_SCOPE in granted(creds)
        print(f"Authorized '{a.account}'" + ("" if has else " (Analytics was not granted)"), file=sys.stderr)
        return 0 if has else 1

    yt = yu.build_service(creds)
    ch = channel(yt)
    ids = a.video or upload_ids(yt, ch.pop("uploads"), a.limit)
    ch.pop("uploads", None)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    report: dict = {"taken": now, "account": a.account, "channel": ch, "videos": videos(yt, ids)}

    if ANALYTICS_SCOPE in granted(creds):
        from googleapiclient.errors import HttpError
        try:
            report["analytics"] = analytics(creds, [v["id"] for v in report["videos"]], parse_since(a.since),
                                            dt.date.today())
        except HttpError as e:
            reason = yu.extract_api_reason(e)
            report["analyticsNote"] = (
                f"Analytics is off in the Cloud project. Switch it on (then wait a minute or two): "
                f"{yu.enable_url('analytics', yu.cloud_project_number(a.client_secrets))}" if reason == "accessNotConfigured"
                else f"Analytics failed: {reason or e}")
    else:
        report["analyticsNote"] = ("No Analytics scope on this token, so no watch time, retention or traffic "
                                   f"sources. Enable the API, then: uv run tools/youtube_stats.py --auth "
                                   f"--account {a.account}")
    if a.github:
        report["github"] = github_traffic()

    d = stats_dir(a.account)
    add_deltas(report, previous_snapshot(d))
    if not a.no_save:
        report["snapshot"] = str(save_snapshot(report, d))
    print(json.dumps(report, indent=2) if a.json else text_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
