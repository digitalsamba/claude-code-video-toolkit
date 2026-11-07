"""Refresh data/star_series.json: cumulative GitHub stars per day, from every stargazer's timestamp.

    uv run examples/data-chart-stars/scripts/fetch_stars.py [owner/repo]

Uses GITHUB_TOKEN if set (unauthenticated calls are limited to 60 an hour, one per 100 stars).
Stars that were later removed aren't counted, so totals can sit slightly under older snapshots.
"""
import collections
import json
import os
import pathlib
import sys
import urllib.request

repo = sys.argv[1] if len(sys.argv) > 1 else "digitalsamba/claude-code-video-toolkit"
headers = {"Accept": "application/vnd.github.star+json", "User-Agent": "video-toolkit-data-chart-stars"}
if os.environ.get("GITHUB_TOKEN"):
    headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

days: collections.Counter = collections.Counter()
page = 1
while True:
    url = f"https://api.github.com/repos/{repo}/stargazers?per_page=100&page={page}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        batch = json.load(r)
    if not batch:
        break
    days.update(s["starred_at"][:10] for s in batch)
    page += 1

total, series = 0, []
for d in sorted(days):
    total += days[d]
    series.append([d, total])

out = pathlib.Path(__file__).resolve().parent.parent / "data" / "star_series.json"
out.write_text(json.dumps(series), encoding="utf-8")
print(f"{repo}: {total} stars over {len(series)} days, {series[0][0]} to {series[-1][0]} -> {out}")
