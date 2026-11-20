---
name: youtube-stats
description: How published videos are doing on YouTube, and what to learn from it. Views, likes, comments and subscribers per video, plus watch time, retention and traffic sources (Shorts feed, search, external) with the Analytics scope, saved as snapshots so trends show; GitHub traffic beside it to see whether YouTube sends people to the repo. Use when asked how a video or the channel is doing, which title, cover or genre worked, whether to make more of something, or for a weekly stats check.
---

# YouTube stats

```bash
uv run --extra youtube tools/youtube_stats.py --account personal                 # every upload, newest first
uv run --extra youtube tools/youtube_stats.py --account personal --video 1hyDgC5Qg_o e4HW-tvi8-8 --since 7d
uv run --extra youtube tools/youtube_stats.py --account personal --github --json  # plus repo referrers
```

Each run saves a snapshot to `_internal/.youtube/stats/<account>/` and reports the change since the last one, so a
daily or weekly run is the trend. `--no-save` for a look without a snapshot.

## Setup

It reuses the token `youtube_upload.py --auth --account NAME` cached, so views, likes and comments work at once.
Watch time, retention and traffic sources need the YouTube Analytics API:

1. Google Cloud Console, the uploader's project: **APIs & Services → Library → YouTube Analytics API → Enable**.
2. **OAuth consent screen → Data access → Add or remove scopes**: add
   `https://www.googleapis.com/auth/yt-analytics.readonly`. It is a sensitive scope, the same class as the
   `youtube` scope the uploader already has, so the consent screen should behave as it did at the first login.
3. `uv run --extra youtube tools/youtube_stats.py --auth --account personal`: one consent that asks for the upload
   scopes and Analytics together and replaces the cached token, so the uploader keeps working.

Impressions and click-through rate are asked for separately: the metrics reference lists them, but the API has
refused them in the past. When it refuses, the report says so; Studio still shows them.

## Reading it

- **A Short lives or dies in the feed.** Traffic source `SHORTS` is the feed; `YT_SEARCH`, `EXT_URL` (links from
  the README, chat) and `SUBSCRIBER` are the rest. A Short with views only from `EXT_URL` was never pushed by the
  feed: change the cover frame and first second, not the title.
- **A Short's views stay on the Short unless it points somewhere.** Compare its views with the long video's and
  with `youtube.com` in the repo referrers. Give it a way on: the long video as its related video (set by hand in
  Studio; `youtube_upload.py` says when), and the link in its description and a pinned comment.
- **Average view percentage is the Short's real score.** Over 100% means people loop it. Under ~50% on a 60-90 s
  Short means the hook or the middle lost them; check where with Studio's retention graph.
- **Long videos: impressions × CTR × view duration.** A low CTR is the title or thumbnail; a short duration is the
  opening.
- **YouTube lags.** Analytics is 1-2 days behind, and a new Short's first 48 hours are the feed testing it; judge
  after a week.
- **The repo is the goal.** `--github` shows the last 14 days of repo referrers; `youtube.com` appearing there is
  the signal that a video sends people to the toolkit.

## Before the next Short

What the numbers above point at is nearly always the opening. Three things to settle while planning, each of
which `qc.py --render` then measures on the finished file (and `--layout` for the text):

- **The first 2 seconds.** Start on the action: movement, a strong picture or a line already being said. Cut the
  lead-in, the establishing shot and any pause before the first word.
- **Text from the first frame.** A short line that says what the viewer gets by staying. Frame 0 is also the
  cover, so the same line sells the video in a grid.
- **Something new every few seconds.** A cut, a push in, a new piece of text. The check reports how long the
  first shot runs and warns past 5 s.

Treat these as defaults to depart from on purpose, and let the retention graph say whether the departure
worked: a slow opening that people watch through is a finding worth writing below.

## What has worked on Shorts

Findings, not numbers: write each as what changed and what happened, with the dates, once there is a week of data.

- 2026-11-08, baseline: You're On Mute (`1hyDgC5Qg_o`) and The Great Return (`e4HW-tvi8-8`) public for about a
  day, 1 and 0 views; channel 26 subscribers. The repo got ~10k unique visitors in 14 days, mostly from Google,
  none from YouTube yet.
