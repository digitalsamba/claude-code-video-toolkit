# Uploading to YouTube

`tools/youtube_upload.py` uploads a rendered video to YouTube using the **YouTube Data API
v3**. The `/publish` command wraps it with metadata auto-filled from a project's `project.json`.

> **The one thing to know up front:** YouTube uploads act *on behalf of a channel*, so they
> require **OAuth 2.0**, not an API key. There is no key-only path. The setup below is a
> one-time ~10-minute click-through in Google Cloud Console.

---

## One-time setup

### 1. Create a Google Cloud project + enable the API
1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and create (or pick) a project.
2. **APIs & Services → Library → "YouTube Data API v3" → Enable.**

### 2. Configure the OAuth consent screen
1. **APIs & Services → OAuth consent screen.**
2. User type **External** → fill in app name + your email.
3. On **Test users**, add the Google account that owns the YouTube channel.
   - While the app is in "Testing", only listed test users can authorize it, and refresh
     tokens expire after ~7 days (see [Gotchas](#gotchas)).

### 3. Create the OAuth client
1. **APIs & Services → Credentials → Create Credentials → OAuth client ID.**
2. Application type: **Desktop app**.
3. Download the `client_secret_*.json`.

### 4. Point the toolkit at it
```bash
echo 'YOUTUBE_CLIENT_SECRETS_FILE=/absolute/path/to/client_secret.json' >> .env
```

### 5. Install dependencies
```bash
pip install -r tools/requirements.txt
# or just the three:
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
```

### 6. Log in once
```bash
python3 tools/youtube_upload.py --auth
```
A browser opens for consent. On success the refresh token is cached at
`_internal/.youtube/token_default.json` (chmod 600, gitignored). Every later upload reuses it
silently — no browser needed.

---

## Usage

```bash
# Upload privately (the safe default)
python3 tools/youtube_upload.py \
  --video out/video.mp4 \
  --title "My video title" \
  --description-file DESCRIPTION.md \
  --tags "ai,agents,explainer" \
  --category 28 \
  --json-out

# Schedule a public go-live (works for first-party uploads — see Gotchas re: the audit)
python3 tools/youtube_upload.py --video out/video.mp4 --title "My video" \
  --publish-at 2026-06-10T09:00:00Z --thumbnail out/thumb.png

# Validate everything without uploading (also reports whether auth is ready)
python3 tools/youtube_upload.py --video out/video.mp4 --title "Test" --dry-run --json-out
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
| `--category` | Numeric ID string. `22` = People & Blogs, `28` = Science & Tech |
| `--privacy` | `private` (default) / `unlisted` / `public` |
| `--publish-at` | ISO8601 UTC, e.g. `2026-06-10T09:00:00Z`. Forces `private` at insert, flips public at the time. |
| `--thumbnail` | Custom thumbnail (≤2MB, 1280×720). Requires a phone-verified channel. |
| `--captions` + `--captions-language` | Caption file (.srt/.vtt) + language code |
| `--playlist` | Playlist ID to add the video to |
| `--account` | Namespaces the cached token — use for multiple channels (e.g. `--account work`) |
| `--auth` | Interactive login only; caches a token and exits |
| `--dry-run` | Builds + prints the request body and checks auth, but uploads nothing |
| `--json-out` | Emits one machine-readable JSON line on stdout |

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
{"success": false, "error": "...", "errorType": "auth", "videoId": null}
```
`errorType` ∈ `auth | quota | validation | upload | http | thumbnail | captions`.
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
  `python3 tools/youtube_upload.py --auth`. Publishing the consent screen removes this (but
  keeps the verification private-lock until verified).
- **Headless servers.** `--auth` needs a browser + reachable localhost. Run it once on a
  machine with a browser, then copy `_internal/.youtube/token_<account>.json` to the server.
- **`publishAt` requires private at insert** — the tool enforces this automatically.
- **Custom thumbnails** require the channel to have a verified phone number, else a 403
  (non-fatal — the video still uploads; the tool logs a warning).
- **The token file is a secret.** `_internal/.youtube/` is gitignored; don't commit it.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing dependency: ...` | `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2` |
| `OAuth client secrets file not found` | Set `YOUTUBE_CLIENT_SECRETS_FILE` in `.env` (absolute path) or pass `--client-secrets` |
| `No cached credentials` | Run `python3 tools/youtube_upload.py --auth` once |
| `Refresh token ... is invalid` | Re-run `--auth` (7-day Testing expiry or revoked token) |
| `errorType: "quota"` | Daily quota hit — wait or request more quota |
| Video uploaded but stays `private` | Unverified-app lock — verify your OAuth app, or publish manually |
| `access_denied` in the browser | The Google account isn't a Test user on the consent screen — add it |
