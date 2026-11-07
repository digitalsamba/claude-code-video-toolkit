# AGENTS.md

This is the [claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit) — an AI-native video production workspace. It's built around [Claude Code](https://claude.ai/code) and `CLAUDE.md`, with experimental bridges to other agent runtimes.

For full toolkit guidance — workflow, conventions, timing rules, design patterns, tool catalog — read **[CLAUDE.md](./CLAUDE.md)**. Everything in there applies equally to a Codex (or other-agent) session.

**What can this toolkit make, and which route fits?** [skills/video-toolkit/SKILL.md](./skills/video-toolkit/SKILL.md)
opens with a routes table (a short explainer as a one-prompt `job`; a product walkthrough rebuilt from an app's
source, a data chart, a sprint review or a marketing demo as a working `session` with the skill each names), and
`uv run tools/video_job.py catalog` lists every template, example build, brand and design source from the registry
with demo links.

## Codex setup

If you're using this repo from Codex, run the migration script once to install the toolkit's skills under `~/.codex/skills/`:

```bash
uv run scripts/migrate_to_codex.py --force
```

That installs 25 entries (11 toolkit skills + 13 command wrappers + 1 overview). The script can also sync the full `CLAUDE.md` content into a generated block at the end of this file — re-run `--force` after editing `CLAUDE.md` to keep the synced block fresh. Manual content above the generated block (i.e. everything you're reading now) is preserved.

To uninstall:

```bash
uv run scripts/migrate_to_codex.py --reset
```

See [README.md § Using with Codex](./README.md#using-with-codex) for details.
