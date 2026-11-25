"""copy_lint.py: the rules, what counts as prose in HTML and Markdown, and --diff on a throwaway git repo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import copy_lint  # noqa: E402

TOOL = ROOT / "tools" / "copy_lint.py"


class RulesTest(unittest.TestCase):
    def test_flags_dashes_and_stock_phrases(self):
        hits = [m.lower() for m, _ in copy_lint.lint_text("A seamless tool — it's worth noting it will elevate you")]
        self.assertIn("—", hits)
        self.assertIn("seamless", hits)
        self.assertIn("it's worth noting", hits)
        self.assertIn("elevate", hits)

    def test_plain_sentences_pass(self):
        self.assertEqual(copy_lint.lint_text("It writes the script, voices it and renders the MP4."), [])
        self.assertEqual(copy_lint.lint_text("A 50-second video, well-made, at 9:16."), [])  # hyphens are fine

    def test_video_job_uses_the_same_rules(self):
        import video_job
        self.assertIs(video_job.COPY_RULES, copy_lint.COPY_RULES)


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def write(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_html_skips_code_and_tags_but_reads_meta(self):
        p = self.write("a.html", '<head><meta name="description" content="A game-changer"></head>\n'
                                 '<p>Plain words.</p>\n<pre><code>ls -- --flag — ok</code></pre>\n'
                                 '<!-- a comment — fine -->\n<p class="seamless">tag attributes are not copy</p>\n')
        found = copy_lint.lint_file(p)
        self.assertEqual([(f["line"], f["match"]) for f in found], [(1, "game-changer")])

    def test_markdown_skips_fences_inline_code_and_link_targets(self):
        p = self.write("a.md", "Text with `--x — y` code.\n```\ncmd — arg\n```\n[docs](https://x/seamless)\n"
                               "A real dash — here.\n")
        found = copy_lint.lint_file(p)
        self.assertEqual([(f["line"], f["match"]) for f in found], [(6, "—")])

    def test_cli_exit_codes(self):
        ok = subprocess.run([sys.executable, str(TOOL), "--text", "Plain words."], capture_output=True)
        bad = subprocess.run([sys.executable, str(TOOL), "--text", "Delve in"], capture_output=True)
        self.assertEqual((ok.returncode, bad.returncode), (0, 1))

    def test_toolkit_brand_checks_product_names_and_spelling(self):
        rules = copy_lint.brand_rules("video-toolkit")
        self.assertEqual(copy_lint.lint_text("Make a voiceover with Claude Code, OpenCode and FFmpeg.", rules), [])
        hits = copy_lint.lint_text("Make a voice-over with claude code and Github.", rules)
        self.assertEqual({match for match, _reason in hits}, {"voice-over", "claude code", "Github"})

    def test_brand_without_copy_rules_falls_back_with_a_warning(self):
        result = subprocess.run([sys.executable, str(TOOL), "--brand", "default", "--text", "Delve in", "--json"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Applying general copy rules", result.stderr)
        self.assertFalse(json.loads(result.stdout)["ok"])


class DiffTest(unittest.TestCase):
    def test_only_added_lines_are_linted(self):
        with tempfile.TemporaryDirectory() as d:
            git = lambda *a: subprocess.run(["git", *a], cwd=d, check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.email", "t@example.org")
            git("config", "user.name", "t")
            Path(d, "README.md").write_text("Old line — with a dash.\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "base")
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
            Path(d, "README.md").write_text("Old line — with a dash.\nNew line, seamless.\n", encoding="utf-8")
            Path(d, "notes.md").write_text("A new file — dashed.\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "FEAT: a seamless change")
            r = subprocess.run([sys.executable, str(TOOL), "--diff", base, "--commits", base, "--json"], cwd=d,
                               capture_output=True, text=True)
            import json
            found = {(f["file"], f["line"], f["match"]) for f in json.loads(r.stdout)["findings"]}
            self.assertEqual(found, {("README.md", 2, "seamless"), ("notes.md", 1, "—"), ("--text", 1, "seamless")})
            self.assertEqual(r.returncode, 1)





class BrandRulesTest(unittest.TestCase):
    """A brand's own rules, from brands/<name>/copy.json."""

    def rules(self, data):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "brands" / "acme").mkdir(parents=True)
        (tmp / "brands" / "acme" / "copy.json").write_text(json.dumps(data))
        old, copy_lint.ROOT = copy_lint.ROOT, tmp
        try:
            return copy_lint.brand_rules("acme")
        finally:
            copy_lint.ROOT = old

    def test_prefer_spares_its_exceptions(self):
        rules = self.rules({"prefer": [{"avoid": "center", "use": "centre", "except": ["Account Center"]}]})
        self.assertEqual(copy_lint.lint_text("Open the Account Center.", rules), [])
        self.assertEqual([m for m, _ in copy_lint.lint_text("In the center of the page.", rules)], ["center"])

    def test_capitalised_phrase_matches_case_sensitively(self):
        rules = self.rules({"avoid": [{"phrase": "Zoom"}], "names": [{"avoid": "DS", "use": "Digital Samba"}]})
        self.assertEqual([m for m, _ in copy_lint.lint_text("You can zoom in. Hosted by DS.", rules)], ["DS"])
        self.assertEqual([m for m, _ in copy_lint.lint_text("Better than Zoom, DS-Free.", rules)], ["Zoom"])  # DS-Free is not DS

    def test_check_level_reports_without_failing(self):
        rules = self.rules({"avoid": [{"phrase": "unlimited", "level": "check"}]})
        hits = [{"reason": why} for _, why in copy_lint.lint_text("Unlimited calls.", rules)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(copy_lint.failing(hits), [])

    def test_shipped_brand_rules_load(self):
        for path in (ROOT / "brands").glob("*/copy.json"):
            self.assertTrue(copy_lint.brand_rules(path.parent.name))


if __name__ == "__main__":
    unittest.main()
