"""youtube_stats.py's offline parts: durations, snapshot deltas and the text report (no API calls)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

HAS_GOOGLE = importlib.util.find_spec("googleapiclient") is not None


def video(vid: str, views: int, secs: int = 60) -> dict:
    return {"id": vid, "title": f"Video {vid}", "published": "2026-11-08T00:00:00Z", "privacy": "public",
            "syntheticMedia": True, "seconds": secs, "upToThreeMin": secs <= 180, "views": views, "likes": 1,
            "comments": 0}


@unittest.skipUnless(HAS_GOOGLE, "needs uv sync --extra youtube")
class YoutubeStatsTest(unittest.TestCase):
    def setUp(self):
        import youtube_stats
        self.ys = youtube_stats

    def test_iso_durations(self):
        self.assertEqual(self.ys.iso_seconds("PT1M30S"), 90)
        self.assertEqual(self.ys.iso_seconds("PT45S"), 45)
        self.assertEqual(self.ys.iso_seconds("PT1H2M3S"), 3723)
        self.assertEqual(self.ys.iso_seconds("P1DT1S"), 86401)
        self.assertEqual(self.ys.iso_seconds(""), 0)

    def test_deltas_against_previous_snapshot(self):
        prev = {"taken": "2026-11-07T00:00:00+00:00", "channel": {"subscribers": 20},
                "videos": [video("a", 10), video("b", 5)]}
        report = {"taken": "2026-11-08T00:00:00+00:00", "channel": {"subscribers": 26},
                  "videos": [video("a", 25), video("c", 3)]}
        self.ys.add_deltas(report, prev)
        self.assertEqual(report["videos"][0]["delta"]["views"], 15)
        self.assertNotIn("delta", report["videos"][1], "a new video has no delta")
        self.assertEqual(report["channel"]["subscribersDelta"], 6)
        self.assertEqual(report["since"], prev["taken"])

    def test_snapshots_round_trip_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertIsNone(self.ys.previous_snapshot(d))
            for t in ("2026-11-07T10:00:00+00:00", "2026-11-08T10:00:00+00:00"):
                self.ys.save_snapshot({"taken": t, "channel": {}, "videos": []}, d)
            self.assertEqual(self.ys.previous_snapshot(d)["taken"], "2026-11-08T10:00:00+00:00")

    def test_text_report_with_analytics_and_github(self):
        r = {"taken": "x", "since": "2026-11-07T00:00:00", "channel": {"title": "Chan", "subscribers": 26,
             "views": 100, "videos": 2, "subscribersDelta": 1},
             "videos": [dict(video("a", 25), delta={"views": 15, "likes": 0, "comments": 0}), video("b", 3, 400)],
             "analytics": {"window": ["2026-11-01", "2026-11-08"], "notes": ["impressions/CTR unavailable"],
                           "videos": {"a": {"averageViewDuration": 40, "averageViewPercentage": 66.7,
                                            "traffic": {"SHORTS": 20, "EXT_URL": 5}}}},
             "github": {"views": {"count": 10, "uniques": 7},
                        "referrers": [{"referrer": "youtube.com", "count": 3, "uniques": 2}]}}
        out = self.ys.text_report(r)
        self.assertIn("(+1 since", out)
        self.assertIn("(+15 views)", out)
        self.assertIn("6:40", out)
        self.assertIn("avg 40s watched (67%)", out)
        self.assertIn("from shorts 20, ext_url 5", out)
        self.assertIn("YouTube sent 2 unique visitors", out)
        self.assertIn("Note: impressions/CTR unavailable", out)

    def test_since_parsing(self):
        import datetime as dt
        self.assertEqual(self.ys.parse_since("7d"), dt.date.today() - dt.timedelta(days=7))
        self.assertEqual(self.ys.parse_since("2026-11-01"), dt.date(2026, 11, 1))


if __name__ == "__main__":
    unittest.main()
