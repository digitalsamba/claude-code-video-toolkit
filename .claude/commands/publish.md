---
description: Publish a finished video to YouTube, or to TikTok/Instagram/LinkedIn/X and more via Upload-Post
---

# Publish a Finished Video

Upload a rendered project, auto-filling the metadata from `project.json`. Two destinations:

| Destination | Tool | Setup |
|-------------|------|-------|
| **YouTube only** | `tools/youtube_upload.py` (OAuth 2.0 + Data API v3, resumable upload) | Google Cloud OAuth client — `docs/youtube-upload.md` |
| **TikTok, Instagram, YouTube, LinkedIn, Facebook, X, Threads, Pinterest, Bluesky** | `tools/upload_post.py` (Upload-Post API) | One API key — `docs/upload-post.md` |

```
project.json + rendered MP4 → metadata draft → dry-run → upload → write back IDs/URLs
```

**Pick the destination first.** If the user named platforms other than YouTube (TikTok,
Reels, LinkedIn, X…), or the project is a 9:16 short headed for several platforms, use the
[multi-platform flow](#multi-platform-flow-upload-post). If they only want YouTube, follow
the steps below. If unclear, ask.

> **One-time setup required.** YouTube uploads need OAuth (not an API key). If the user
> hasn't set this up, point them at `docs/youtube-upload.md` and stop until
> `YOUTUBE_CLIENT_SECRETS_FILE` is in `.env` and `uv run tools/youtube_upload.py --auth`
> has been run once. Don't attempt an upload without a cached token.

## Entry Point

### Step 1: Locate the project and its rendered video

If the user named a project, use it. Otherwise scan for candidates:

```bash
cd /path/to/claude-code-video-toolkit && ls projects/*/project.json
```

Read the chosen `project.json` **defensively** — real files carry `render`, `format`,
and `publish` blocks beyond the `lib/project/types.ts` schema. Resolve the video file:
1. `render.file` if present (e.g. `out/ai-agent-short.mp4`), relative to the project dir.
2. Else scan the project's `out/*.mp4` and pick the most recent.
3. Confirm the resolved path exists and is non-empty before continuing.

If the project's `phase` isn't `complete`, warn the user and confirm they still want to publish.

### Step 2: Assemble metadata (into a `publish` block)

Build a `publish` object and write it back into `project.json` so it's reviewable,
editable, and re-runnable. If a `publish` block already exists, use it as defaults.

| Field | How to derive |
|-------|---------------|
| `title` | Existing `publish.title`, else the hook/title scene's `title`, else the project `name` (humanized). Keep ≤100 chars. |
| `description` | Existing `publish.description`, else auto-draft: a 1–2 line summary from the scene narration/titles + a channel footer (links, hashtags). Keep ≤5000 chars. |
| `tags` | Existing `publish.tags`, else derive 5–12 topical tags from scene titles + the brand. Comma-joined when passed to the tool. |
| `category` | Default `"22"` (People & Blogs). Override per channel/topic — e.g. `"28"` (Science & Tech), `"27"` (Education), `"24"` (Entertainment). |
| `thumbnail` | Look for `out/thumbnail.*` or `public/thumbnail.*`. If none and the user wants one, offer to generate via `tools/ideogram4.py` (see the `ideogram4` skill). |
| `privacy` | Default **`private`** (safe — the video uploads but stays hidden until you flip it). Offer `unlisted`, `public`, or a scheduled go-live (`--publish-at`, e.g. next morning ~09:00 local in UTC) if the user asks. |
| `playlist` | Optional; only if the user has one. |

**Show the assembled metadata to the user and let them edit before uploading.**

### Step 3: Dry-run first (no upload)

```bash
cd /path/to/claude-code-video-toolkit && uv run tools/youtube_upload.py \
  --video "projects/NAME/out/video.mp4" \
  --title "TITLE" \
  --description-file "projects/NAME/.publish-description.txt" \
  --tags "tag1,tag2,tag3" \
  --category "28" \
  --publish-at "2026-06-10T09:00:00Z" \
  --thumbnail "projects/NAME/out/thumbnail.png" \
  --dry-run --json-out
```

Write the description to a temp file (`--description-file`) rather than passing a long
`--description` on the command line. Parse the JSON: confirm `requestBody` looks right and
`authOk` is `true`. If `authOk` is `false`, surface the auth error and have the user run
`uv run tools/youtube_upload.py --auth` first — do not proceed.

### Step 4: Upload

Re-run the same command **without** `--dry-run`, keeping `--json-out`. Parse the result.

### Step 5: Write back and report

On `success`, merge into the project's `publish` block:
```json
"publish": {
  "platform": "youtube",
  "videoId": "<id>",
  "url": "https://www.youtube.com/watch?v=<id>",
  "privacyStatus": "<actual returned status>",
  "publishAt": "<scheduled time or null>",
  "uploadedAt": "<today ISO date>"
}
```
Append a `sessions[]` entry summarizing the upload, then report to the user:

```
Published to YouTube

Title:    <title>
URL:      https://www.youtube.com/watch?v=<id>
Privacy:  <actual>  (requested: <requested>)
Schedule: <publishAt or "—">
```

**If `privacyStatus` came back `private` but you requested public/scheduled**, tell the
user plainly: this is the unverified-app lock — the video uploaded but won't go public
until their Google Cloud OAuth app is verified. They can publish manually in YouTube Studio.

---

## Multi-platform Flow (Upload-Post)

Publishes the same render to any mix of TikTok, Instagram (Reels/Stories), YouTube, LinkedIn,
Facebook, X, Threads, Pinterest and Bluesky with `tools/upload_post.py`.

> **One-time setup required.** If `UPLOAD_POST_API_KEY` / `UPLOAD_POST_USER` aren't in `.env`,
> point the user at `docs/upload-post.md` (create an account, connect their accounts to a
> profile, create an API key) and stop until they're set.

### Step 1: Locate the project and its rendered video

Same as the YouTube flow above. Note the aspect ratio: TikTok, Reels and Shorts want 9:16. If
the render is landscape and the user picked those, say so before publishing.

### Step 2: Assemble metadata

| Field | How to derive |
|-------|---------------|
| `title` | Existing `publish.title`, else the hook/title scene. This is the caption on TikTok/Instagram/X/Threads — write it like a caption (hook + 2–4 hashtags), not like a YouTube title. ≤100 chars if YouTube is included. |
| `description` | Longer text, used on YouTube, LinkedIn, Facebook and Pinterest. Write to `projects/NAME/.publish-description.txt`. |
| `platforms` | What the user asked for. |
| `schedule` | Optional ISO time + IANA `timezone`. |
| `aiGenerated` | Ask whether to disclose AI-generated content (`--ai-generated`). Recommend yes when the visuals or voice are AI-generated — TikTok, Instagram and YouTube expect it for realistic synthetic media. |
| per-platform | `--youtube-privacy` (default `private`), `--tiktok-privacy`, `--tiktok-draft`, `--instagram-story`, `--pinterest-board` (required for Pinterest). |

**Show the assembled metadata and the platform list, and let the user edit before posting.**
Publishing to social accounts is public and hard to undo — never skip this confirmation.

### Step 3: Dry-run first (no upload)

```bash
cd /path/to/claude-code-video-toolkit && uv run tools/upload_post.py \
  --video "projects/NAME/out/video.mp4" \
  --title "CAPTION" \
  --description-file "projects/NAME/.publish-description.txt" \
  --platforms tiktok,instagram,youtube \
  --ai-generated \
  --dry-run --json-out
```

Confirm `authOk` is `true`. If `missingPlatforms` is non-empty, those accounts aren't connected
to the profile and would be skipped — tell the user and let them connect them or drop them.

### Step 4: Upload

Re-run **without** `--dry-run`, keeping `--json-out`. The tool waits for every platform (up to
10 min) and prints one result per platform. Never re-run the upload after a timeout, a network
error or a `status: "unknown"` result — a new run is a new post. Use
`uv run tools/upload_post.py --status <requestId> --json-out` instead.

### Step 5: Write back and report

Merge into the project's `publish` block (keep any existing YouTube fields):
```json
"publish": {
  "uploadPost": {
    "requestId": "<requestId or null>",
    "jobId": "<jobId for scheduled posts, else null>",
    "scheduledDate": "<or null>",
    "results": [{"platform": "tiktok", "status": "completed", "url": "..."}],
    "uploadedAt": "<today ISO date>"
  }
}
```
Append a `sessions[]` entry, then report one line per platform with its URL or error.
`skipped` = no account connected; `inbox: true` on TikTok = delivered to drafts, publish from the
TikTok app.

---

## Quick Mode

Direct invocation for experienced users:
```
/publish ai-agent-short
/publish ai-agent-short --privacy unlisted
/publish ai-agent-short tiktok,instagram,youtube
```
Parse the project name, any platform list and any privacy/schedule overrides, still show the
metadata and run a dry-run before the real upload. A platform list other than just YouTube
means the multi-platform flow.

For `tools/upload_post.py` options, see `docs/upload-post.md`.

---

## Tool Reference (`tools/youtube_upload.py`)

| Option | Description |
|--------|-------------|
| `--video, --input` | Path to the video file |
| `--title` | Title (≤100 chars) |
| `--description` / `--description-file` | Description text, or a file (`-` = stdin) |
| `--tags` | Comma-separated tags (combined ≤500 chars) |
| `--category` | Numeric category ID string (default `22`; `28` = Science & Tech) |
| `--privacy` | `private` (default) / `unlisted` / `public` |
| `--publish-at` | ISO8601 UTC schedule, e.g. `2026-06-10T09:00:00Z` (forces private at insert) |
| `--thumbnail` | Custom thumbnail (≤2MB, 1280×720) |
| `--captions` + `--captions-language` | Caption file + language code |
| `--playlist` | Playlist ID |
| `--account` | Channel name namespacing the cached token (default `default`) |
| `--auth` | Interactive login only — cache a token and exit |
| `--dry-run` | Validate + print the request body without uploading |
| `--json-out` | Single machine-readable JSON line on stdout |

---

## Quota & Limits (worth knowing)

- Default API quota is **10,000 units/day**; each upload costs **~1,600 units → ~6 uploads/day**.
- Hitting quota returns HTTP 403 `quotaExceeded` (the tool reports `errorType: "quota"`).
- Custom thumbnails require a channel with a verified phone number.

---

## Error Handling

| Symptom (`errorType`) | Solution |
|-----------------------|----------|
| `auth` | Run `uv run tools/youtube_upload.py --auth` once; if the refresh token expired (7-day Testing limit), re-run `--auth`. |
| `validation` | Fix the flagged field (missing video/title, bad `--publish-at`). |
| `config` | YouTube Data API v3 isn't enabled on the Cloud project (or just enabled, still propagating). Enable it, wait 1–2 min, retry. See `docs/youtube-upload.md`. |
| `quota` | Daily quota exhausted — wait, or request more in Google Cloud. |
| `forbidden` | Permission denied (e.g. the authorized account lacks rights to the target channel). Check the account / `--account`. |
| `upload` / `http` | Transient network/server issue; the tool already retried — try again later. |
| Video uploaded but stuck `private` | Unverified OAuth app lock — verify the app, or publish manually. |

---

## Evolution

This command evolves through use. If something's awkward or missing:
1. Say "improve this" → Claude captures it in `_internal/BACKLOG.md`
2. Edit `.claude/commands/publish.md` → update `_internal/CHANGELOG.md`
- Issues/PRs: `github.com/digitalsamba/claude-code-video-toolkit`
