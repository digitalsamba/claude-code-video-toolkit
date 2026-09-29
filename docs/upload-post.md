# Publishing to social platforms (Upload-Post)

`tools/upload_post.py` publishes a rendered video to **TikTok, Instagram (Reels or Stories),
YouTube (incl. Shorts), LinkedIn, Facebook, X, Threads, Pinterest and Bluesky** in one call,
through the [Upload-Post](https://upload-post.com) API. The `/publish` command wraps it with
metadata auto-filled from a project's `project.json`.

It complements `tools/youtube_upload.py` rather than replacing it:

| | `youtube_upload.py` | `upload_post.py` |
|---|---|---|
| Destinations | YouTube | TikTok, Instagram, YouTube, LinkedIn, Facebook, X, Threads, Pinterest, Bluesky |
| Auth | Your own Google Cloud OAuth client + browser login | One API key; accounts connected once in the Upload-Post dashboard |
| Runs on | Google's API quota (~6 uploads/day by default) | An Upload-Post plan (free: 10 uploads/month, no TikTok) |
| Best for | YouTube-only, fully first-party | Shorts/Reels/TikTok and cross-posting |

> **Disclosure:** Upload-Post is a hosted third-party service. Its free plan covers 10 uploads a
> month on every platform above except TikTok, which needs a paid plan. The toolkit only talks
> to it when you run this tool with a key you created.

---

## One-time setup

1. **Create an account** at [upload-post.com](https://upload-post.com).
2. **Create a profile and connect your accounts.** A profile groups the social accounts you
   post from (e.g. one per brand or client). Connect each platform you want to publish to.
3. **Create an API key** in the dashboard.
4. **Add both to `.env`:**

```bash
echo 'UPLOAD_POST_API_KEY=your_api_key' >> .env
echo 'UPLOAD_POST_USER=your_profile_name' >> .env
```

5. **Check it** — a dry run validates the key, the profile, and which of the platforms you ask
   for are actually connected, without posting anything:

```bash
uv run tools/upload_post.py --video out/video.mp4 --title "Test" \
    --platforms tiktok,instagram,youtube --dry-run --json-out
```

`authOk: true` and an empty `missingPlatforms` means you're ready. No extra Python deps are
needed (`requests` is already a core dependency).

---

## Usage

```bash
# Vertical short to TikTok + Reels + Shorts, now; waits for every platform's result
uv run tools/upload_post.py --video out/short.mp4 --title "How RAG works in 45s #ai" \
    --platforms tiktok,instagram,youtube --ai-generated --json-out

# Landscape explainer to LinkedIn + X, scheduled in your timezone
uv run tools/upload_post.py --video out/video.mp4 --title "Our new release" \
    --description-file projects/NAME/.publish-description.txt \
    --platforms linkedin,x --schedule 2026-10-01T09:00:00 --timezone Europe/London --json-out

# Send to TikTok drafts / keep YouTube unlisted while you review
uv run tools/upload_post.py --video out/short.mp4 --title "Draft" \
    --platforms tiktok,youtube --tiktok-draft --youtube-privacy unlisted

# Check an earlier upload (request_id or scheduled job_id)
uv run tools/upload_post.py --status <id> --json-out
```

### Options

| Option | Description |
|--------|-------------|
| `--video, --input` | Path to the video file |
| `--title` | Caption/title used on every platform (YouTube ≤100 chars) |
| `--description` / `--description-file` | Longer text for YouTube, LinkedIn, Facebook, Pinterest |
| `--platforms` | Comma-separated: `tiktok,instagram,youtube,linkedin,facebook,x,threads,pinterest,bluesky` (aliases `twitter`, `reels`, `shorts`) |
| `--user` | Upload-Post profile to post from (default `$UPLOAD_POST_USER`) |
| `--schedule` + `--timezone` | Publish later (ISO8601, ≤365 days ahead); timezone is IANA, default UTC |
| `--first-comment` | Posted as the first comment/reply once the post is live |
| `--ai-generated` | Self-disclose AI-generated content (TikTok, Instagram, YouTube, X labels) |
| `--thumbnail` | Custom thumbnail (YouTube, LinkedIn) |
| `--youtube-privacy` | `private` (default) / `unlisted` / `public` |
| `--youtube-category`, `--tags` | YouTube category ID (default `22`) and tags |
| `--tiktok-privacy` | `PUBLIC_TO_EVERYONE` / `MUTUAL_FOLLOW_FRIENDS` / `FOLLOWER_OF_CREATOR` / `SELF_ONLY` (default: the account's own) |
| `--tiktok-draft` | Send to TikTok drafts instead of posting |
| `--instagram-story` | Post to Stories instead of Reels |
| `--facebook-page-id`, `--linkedin-page-id` | Pick a Facebook Page / post as a LinkedIn company page |
| `--pinterest-board` | Board ID (required for Pinterest) |
| `--status ID` | Look up an earlier upload and exit |
| `--no-wait`, `--wait-timeout` | Don't poll / how long to poll (default 600 s) |
| `--dry-run` | Validate + check the account, no upload |
| `--json-out` | Single machine-readable JSON line on stdout |

### Output (`--json-out`)

```json
{
  "success": true,
  "status": "completed",
  "requestId": "7b2c2f5e-…",
  "results": [
    {"platform": "tiktok", "status": "completed", "url": "https://www.tiktok.com/@you/video/…", "postId": "…", "note": null, "error": null, "inbox": false},
    {"platform": "youtube", "status": "completed", "url": "https://www.youtube.com/watch?v=…", "postId": "…", "note": null, "error": null, "inbox": false}
  ],
  "timedOut": false,
  "platforms": ["tiktok", "youtube"]
}
```

Per-platform `status` is `completed`, `failed`, `retryable` (Upload-Post retries it), or
`skipped` (the profile has no account for that platform — nothing was posted there).
`success` is `false` if any platform failed. A private post has no public link: YouTube still
gets a `url` (visible to the channel owner), other platforms return `postId` and a `note`.
Scheduled uploads return `{"status": "scheduled", "jobId": …, "requestId": …}` right away;
`--status` accepts either id.

---

## How it behaves

- **One upload, many platforms.** The file is sent once; Upload-Post transcodes where a platform
  needs it and publishes everywhere in parallel.
- **Asynchronous with polling.** The API answers with a `request_id` as soon as the file is
  received; the tool polls the status endpoint every 10 s until all platforms finish or
  `--wait-timeout` passes. A timeout doesn't cancel anything — use `--status` later.
- **No double posts.** The tool generates the `request_id` itself and sends it as an
  `Idempotency-Key`. If the connection drops mid-upload it does **not** resend the file; it
  polls that same id to find out whether the upload arrived.
- **Safe defaults.** YouTube uploads default to `private` like `youtube_upload.py`. TikTok uses
  the account's own default privacy unless you pass `--tiktok-privacy`.
- **Aspect ratio matters.** TikTok, Reels and Shorts want vertical 9:16 — the
  `concept-explainer-short` template renders exactly that. Landscape renders suit YouTube,
  LinkedIn, X and Facebook.

---

## Troubleshooting

| Symptom (`errorType`) | Fix |
|-----------------------|-----|
| `auth` | `UPLOAD_POST_API_KEY` missing or wrong. Recreate the key in the dashboard. |
| `validation` | A flag is wrong (missing title, unknown platform, past `--schedule`, Pinterest without `--pinterest-board`), or none of the requested platforms is connected to the profile. The message says which. |
| `forbidden` | The plan doesn't allow it — e.g. TikTok on the Free plan. The message says why. |
| `quota` | Rate or plan limit reached — the message says which; wait or upgrade. |
| Platform `skipped` | Connect that platform to the profile in the Upload-Post dashboard. |
| TikTok `inbox: true` | TikTok delivered the video to the account's drafts instead of posting it live; open the TikTok app to publish. |
| Platform `failed` | The `error` field carries the platform's own reason (e.g. an expired connection — reconnect it in the dashboard). |

Full API reference: [docs.upload-post.com](https://docs.upload-post.com).
