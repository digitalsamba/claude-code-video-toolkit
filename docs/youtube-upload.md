# Uploading to YouTube

`tools/youtube_upload.py` uploads a rendered video to YouTube using the **YouTube Data API
v3**. The `/publish` command wraps it with metadata auto-filled from a project's `project.json`.

> **The one thing to know up front:** YouTube uploads act *on behalf of a channel*, so they
> require **OAuth 2.0**, not an API key. There is no key-only path. The setup below is a
> one-time ~10-minute click-through in Google Cloud Console.

---

## Fast path with Claude Code

Most of this setup is wiring Claude can do for you — the only parts that *require a human*
are the Google Cloud Console clicks and the one browser consent. Split the work like this:

**You do (in the browser, ~5 min) — three things, in this order:**

1. **Enable the API.** Pick/create a Cloud project, then **APIs & Services → Library →
   "YouTube Data API v3" → Enable.** ⚠️ *Don't skip this* — creating the OAuth client does
   **not** enable the API, and a missing enable surfaces only at upload time as
   `HTTP 403 accessNotConfigured`. After enabling, give it **1–2 minutes to propagate**.
2. **Add yourself as a Test user.** **APIs & Services → OAuth consent screen** (Google's newer
   UI calls this **"Google Auth Platform" → Audience** tab). Set **User type: External**, keep
   **Publishing status: Testing**, then under **Test users → Add users** add the *exact* Google
   account you'll log in with, and **Save**. ⚠️ If you skip this you get a hard
   *"Access blocked: … has not completed the Google verification process"* with no way through.
   (When you*are* a test user you instead get a softer "Google hasn't verified this app →
   Advanced → Continue" screen — that one is expected; click through it.)
3. **Create a Desktop OAuth client and download it.** **APIs & Services → Credentials → Create
   Credentials → OAuth client ID → Application type: Desktop app → Create → Download JSON.**

**Then hand the rest to Claude.** Tell it where the downloaded `client_secret_*.json` is, and
ask it to finish YouTube setup. Claude will: move the secret somewhere stable out of the repo,
add `YOUTUBE_CLIENT_SECRETS_FILE` to `.env`, install the `google-*` deps into `.venv`, launch
the `--auth` browser login (you click "Allow" once), confirm with a `--dry-run`, and run a
private test upload to verify the round-trip.

**Two account gotchas worth stating up front:**
- **Brand Accounts:** if your channel is a Brand Account, log in with the *personal* Google
  account that **manages** it (add *that* email as the Test user). The upload still lands on the
  brand channel.
- **Exact-match:** the email you add as a Test user must be byte-for-byte the account you pick
  in the consent screen.

The detailed manual version of all of this follows.

---

## One-time setup

### 1. Create a Google Cloud project + enable the API
1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and create (or pick) a project.
   Confirm the **project selector in the top bar** shows the project you mean — operating in the
   wrong project is the most common source of confusing failures here.
2. **APIs & Services → Library → "YouTube Data API v3" → Enable.**
   - **This is mandatory and easy to forget.** If it's not enabled, auth and `--dry-run` both
     succeed, but the real upload fails with `HTTP 403 accessNotConfigured` naming this exact
     project. After enabling, **wait 1–2 minutes** for it to propagate before retrying.

### 2. Configure the OAuth consent screen
1. **APIs & Services → OAuth consent screen** (newer UI: **"Google Auth Platform" → Audience**).
2. User type **External** → fill in app name + your email. Leave **Publishing status: Testing**.
3. On **Test users → Add users**, add the Google account that owns the YouTube channel (a Brand
   Account's *managing personal account* if applicable), and **Save**.
   - While the app is in "Testing", **only listed test users can authorize it** — a non-listed
     account gets a hard *"Access blocked … verification process"* with no override. Refresh
     tokens also expire after ~7 days in Testing (see [Gotchas](#gotchas)).

### 3. Create the OAuth client
1. **APIs & Services → Credentials → Create Credentials → OAuth client ID.**
2. Application type: **Desktop app**.
3. Download the `client_secret_*.json`.

### 4. Point the toolkit at it
```bash
echo 'YOUTUBE_CLIENT_SECRETS_FILE=/absolute/path/to/client_secret.json' >> .env
```

### 5. Install dependencies
The three `google-*` packages are **opt-in** — they're commented out in
`tools/requirements.txt` so base installs stay lean, so install them explicitly:
```bash
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
# (or uncomment the YouTube block in tools/requirements.txt and re-run the -r install)
```
The tool prints this exact command if you run it before the deps are present.

### 6. Log in once
```bash
uv run tools/youtube_upload.py --auth
```
A browser opens for consent. On success the refresh token is cached at
`_internal/.youtube/token_default.json` (chmod 600, gitignored). Every later upload reuses it
silently — no browser needed.

---

## Usage

```bash
# Upload privately (the safe default)
uv run tools/youtube_upload.py \
  --video out/video.mp4 \
  --title "My video title" \
  --description-file DESCRIPTION.md \
  --tags "ai,agents,explainer" \
  --category 28 \
  --json-out

# Schedule a public go-live (works for first-party uploads — see Gotchas re: the audit)
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \
  --publish-at 2026-06-10T09:00:00Z --thumbnail out/thumb.png

# Is everything set up? (channel, scopes, APIs; read-only)
uv run tools/youtube_upload.py --check --account personal

# Subtitles and chapters from the render (one transcription, ~$0.01 for 90 s)
uv run tools/publish_kit.py --project projects/my-video --video out/video.mp4
#   -> out/publish/video.srt for --captions, out/publish/chapters.txt for the description

# Captions for a video that's already up
uv run tools/youtube_upload.py --video-id VIDEO_ID --captions out/video.srt --json-out

# Validate everything without uploading (also reports whether auth is ready)
uv run tools/youtube_upload.py --video out/video.mp4 --title "Test" --dry-run --json-out
```

Prefer **`/publish`** for finished projects — it derives the title/description/tags from
`project.json`, runs a dry-run, then uploads and writes the resulting video URL back.

### Key flags

| Flag | Notes |
|------|-------|
| `--video`, `--input` | The file to upload (required) |
| `--title` | Required, ≤100 chars |
| `--description` / `--description-file` | Inline text, or a file (`-` = stdin); ≤5000 chars |
| `--tags` | Comma-separated; combined length ≤500 chars (overflow is dropped with a warning) |
| `--category` | Numeric ID string. `28` = Science & Tech, `27` = Education, `24` = Entertainment. Left out, it is `22` = People & Blogs, with a note |
| `--privacy` | `private` (default) / `unlisted` / `public` |
| `--publish-at` | ISO8601 UTC, e.g. `2026-06-10T09:00:00Z`. Forces `private` at insert, flips public at the time. |
| `--thumbnail` | Custom thumbnail (≤2MB, 1280×720). Requires a phone-verified channel. |
| `--captions` + `--captions-language` | Caption file (.srt/.vtt) + language code. Needs the `youtube.force-ssl` scope: a token from before it was added is refused before the upload starts, so re-run `--auth` once |
| `--playlist` | Playlist ID to add the video to |
| `--video-id` | Add `--captions`, `--thumbnail` or `--playlist` to a video already uploaded, with no new upload (no `--video` or `--title`) |
| `--synthetic-media` | YouTube's "altered or synthetic content" disclosure (`status.containsSyntheticMedia`). Required for realistic AI-generated or animated people, or a cloned voice; it adds a label, not a reach penalty |
| `--account` | Namespaces the cached token — use for multiple channels (e.g. `--account work`) |
| `--auth` | Interactive login only; caches a token and exits |
| `--check` | Read-only setup check: the channel the login belongs to, its scopes, and whether the Data and Analytics APIs are on. Prints the fix for each problem, including the Cloud Console page that switches an API on. ~1 quota unit |
| `--dry-run` | Builds + prints the request body and checks auth, but uploads nothing |
| `--json-out` | Emits one machine-readable JSON line on stdout |

### Notes on the metadata

Before it uploads (and in a dry run), the tool reads the title, description, tags and category and prints a note
for each thing a viewer or YouTube would miss. A note is advice: it never stops the upload. With `--json-out`
they are in `notes`.

| Note | Why |
|------|-----|
| No hashtags in the description | YouTube shows the first three above the title and links each to a topic page |
| No link in the description | Nothing sends the viewer on to the repository, the site or the full video |
| None of the tags appears in the title | The title counts for more than the tags in search |
| No `--category` | The upload falls back to People & Blogs |
| The file is vertical or square and 3 minutes or less | It becomes a Short. If it is a cut of a longer video, set that as its **Related video** by hand (Studio, Content, Shorts, Details): the API can't, and the field needs the channel's advanced features on |

The notes read the request, not the audience. Whether a video reached anyone is in `youtube_stats.py`
(see the `youtube-stats` skill): a Short is placed by the feed on how people watch it, not on its tags.

### JSON output (`--json-out`)

Success:
```json
{"success": true, "videoId": "abc123", "url": "https://www.youtube.com/watch?v=abc123",
 "privacyStatus": "private", "requestedPrivacy": "public", "publishAt": "2026-06-10T09:00:00Z",
 "account": "default", "thumbnailSet": true, "captionsAdded": false, "playlistId": null,
 "uploadStatus": "uploaded"}
```
Failure:
```json
{"success": false, "error": "...", "errorType": "config", "errorReason": "accessNotConfigured", "videoId": null}
```
`errorType` ∈ `auth | quota | config | forbidden | validation | upload | http | thumbnail | captions`.
For HTTP errors, `errorReason` carries YouTube's machine reason (e.g. `accessNotConfigured`,
`quotaExceeded`, `forbidden`) so a disabled API (`config`) is distinguishable from a real quota
wall (`quota`) or a permission denial (`forbidden`) — all three are HTTP 403.
Compare `requestedPrivacy` vs `privacyStatus` to detect the unverified-app force-to-private case.

---

## Quota

- Default quota: **10,000 units/day** per Google Cloud project.
- `videos.insert` ≈ **1,600 units** → about **6 uploads/day**.
- `thumbnails.set` ≈ 50, `captions.insert` ≈ 400, `playlistItems.insert` ≈ 50.
- Exceeding it returns HTTP 403 `quotaExceeded` (`errorType: "quota"`). Request more via the
  Google Cloud quota form if you need higher volume.

---

## Gotchas

- **Unaudited-project private lock (often a non-issue for first-party uploads).** Google's
  docs say videos uploaded via `videos.insert` from unaudited API projects (created after
  2020-07-28) are restricted to `private`. In practice this targets *third-party* apps
  uploading to *other people's* channels — when the Cloud project owner, the channel, and the
  authenticated account are all **you** (a first-party upload), it generally isn't enforced and
  public/scheduled uploads go live fine (verified empirically on this toolkit). If it does bite
  (e.g. uploading to a different channel via `--account`), lift it by completing the **YouTube
  API compliance audit** (publish the app to production, then submit the audit form). Either way
  the tool reports the *actual* returned privacy — treat that as the source of truth.
- **7-day refresh-token expiry in "Testing".** Consent screens left in Testing mode expire
  refresh tokens after ~7 days. If an unattended run fails with `errorType: "auth"`, just re-run
  `uv run tools/youtube_upload.py --auth`. Publishing the consent screen removes this (but
  keeps the verification private-lock until verified).
- **Headless servers.** `--auth` needs a browser + reachable localhost. Run it once on a
  machine with a browser, then copy `_internal/.youtube/token_<account>.json` to the server.
- **`publishAt` requires private at insert** — the tool enforces this automatically.
- **Custom thumbnails** require the channel to have a verified phone number, else a 403
  (non-fatal — the video still uploads; the tool logs a warning).
- **The token file is a secret.** `_internal/.youtube/` is gitignored; don't commit it.

---

### Scheduling a video that is already uploaded

`--publish-at` schedules at upload time. For a private video that is already up (uploaded for someone to watch
first), set the time with `videos.update` on `part=status`, keeping `privacyStatus: private`:

```python
yt.videos().update(part="status", body={"id": VIDEO_ID, "status": {
    "privacyStatus": "private", "publishAt": "2026-11-24T15:00:00Z",
    "embeddable": True, "publicStatsViewable": True, "license": "youtube", "selfDeclaredMadeForKids": False,
}}).execute()
```

Two things to know. Send every status field you want kept: an update that leaves out `embeddable` or
`publicStatsViewable` sets them to false. And a `videos.list` straight after the update can still return the old
status, so read the update's own response, or wait about 20 seconds before reading back.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing dependency: ...` | `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2` |
| `OAuth client secrets file not found` | Set `YOUTUBE_CLIENT_SECRETS_FILE` in `.env` (absolute path) or pass `--client-secrets` |
| `No cached credentials` | Run `uv run tools/youtube_upload.py --auth` once |
| `Refresh token ... is invalid` | Re-run `--auth` (7-day Testing expiry or revoked token) |
| `accessNotConfigured` | An API isn't switched on in the Cloud project. `--check` prints the page to switch it on; wait a minute or two after |
| `can't add captions (it lacks ...force-ssl)` | The token predates the captions scope: re-run `--auth`, then attach them with `--video-id` |
| `errorType: "quota"` | Daily quota hit — wait or request more quota |
| Video uploaded but stays `private` | Unverified-app lock — verify your OAuth app, or publish manually |
| `access_denied` in the browser | The Google account isn't a Test user on the consent screen — add it |

## Command reference and stats

Upload a finished render to YouTube via the Data API v3. Use the `/publish` command for the
guided workflow (it auto-fills title/description/tags from `project.json`), or call the tool directly.

```bash
# One-time login (opens a browser, caches a refresh token under _internal/.youtube/)
uv run tools/youtube_upload.py --auth

# Upload privately (the safe default)
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \
    --description-file DESCRIPTION.md --tags "ai,agents,explainer" --json-out

# Schedule a public go-live
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \
    --publish-at 2026-06-10T09:00:00Z --thumbnail out/thumb.png --json-out

# Validate everything without uploading (also reports auth readiness)
uv run tools/youtube_upload.py --video out/video.mp4 --title "Test" --dry-run --json-out

# Setup check: channel, scopes, Data/Analytics API on or off, with the fix for each (read-only)
uv run tools/youtube_upload.py --check --account personal

# Subtitles and chapters from the render: one SRT for --captions, chapter lines for the description
uv run tools/publish_kit.py --project projects/my-video --video out/video.mp4
```

Before uploading, and in a dry run, the tool prints notes on the metadata (`notes` in the JSON): no hashtags, no
link, no tag in the title, no `--category`, and a Short that may need its related video set by hand in Studio.
They are advice and never stop an upload.

The publish kit transcribes the render once (Scribe, about $0.01 for 90 s, or `--engine whisper`) and aligns it to
the scene scripts, so the subtitles are spelled as written and timed to the render, with no scene offsets to add
up. Chapters sit at each scene's cut, titled from `project.json` `scenes[].title` or the file names, under
YouTube's rules (first at 0:00, three or more, 10 s each). Re-runs reuse the saved transcript.

**Setup is OAuth, not an API key** (uploads act on behalf of a channel). See
`docs/youtube-upload.md` for the one-time Google Cloud Console walkthrough and
`YOUTUBE_CLIENT_SECRETS_FILE` in `.env`. Key realities:
- Default quota is 10,000 units/day; `videos.insert` costs ~1,600 units → **~6 uploads/day**.
- **Unaudited API projects *may* have public uploads force-locked to private**: but this
  mainly affects uploads to *other* people's channels. First-party uploads (your own project +
  channel + account) generally publish public/scheduled fine without the audit. The tool reports
  the actual returned privacy. "Testing"-mode refresh tokens expire after ~7 days (re-run `--auth`).
- Cached tokens live in `_internal/.youtube/` (gitignored, they grant channel-upload access).

**How it did:** `uv run --extra youtube tools/youtube_stats.py --account personal [--github]` reads views, likes and
comments with the same token, adds watch time, retention and traffic sources once the token has the Analytics scope
(`--auth`), and snapshots each run to `_internal/.youtube/stats/` so the next one shows the change. Reading the
numbers, and what has worked on Shorts, is in `.claude/skills/youtube-stats/`.
