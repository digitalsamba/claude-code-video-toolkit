# AGENTS.md

This is the [agent-video-toolkit](https://github.com/digitalsamba/agent-video-toolkit), an AI-native video production workspace. It started in [Claude Code](https://claude.ai/code), was called claude-code-video-toolkit until v0.31.0, and keeps its guidance in `CLAUDE.md`. Codex and OpenCode have each made a whole video with it unaided (`examples/agent-comparison-manim`).

For full toolkit guidance — workflow, conventions, timing rules, design patterns, tool catalog — read **[CLAUDE.md](./CLAUDE.md)**. Everything in there applies equally to a Codex (or other-agent) session.

**What can this toolkit make, and which route fits?** [skills/video-toolkit/SKILL.md](./skills/video-toolkit/SKILL.md)
opens with a routes table (a short explainer as a one-prompt `job`; a product walkthrough rebuilt from an app's
source, a data chart, a sprint review or a marketing demo as a working `session` with the skill each names), and
`uv run tools/video_job.py catalog` lists every template, example build, brand and design source from the registry
with demo links.

## Codex setup

Codex can use the toolkit instructions and skills directly from this checkout. If you want the skills and
command wrappers available across workspaces, install them under `~/.codex/skills/`:

```bash
uv run scripts/migrate_to_codex.py --force
```

That installs 25 entries (11 toolkit skills + 13 command wrappers + 1 overview). The script can also sync the full `CLAUDE.md` content into a generated block at the end of this file — re-run `--force` after editing `CLAUDE.md` to keep the synced block fresh. Manual content above the generated block (i.e. everything you're reading now) is preserved.

To uninstall:

```bash
uv run scripts/migrate_to_codex.py --reset
```

See [README.md § Using with Codex](./README.md#using-with-codex) for details.

If `uv run` fails because the Codex sandbox blocks its cache, use `uv cache dir` to find the cache and
allow that directory in the sandbox's writable roots. If you can't change those settings, run from the
toolkit root and use a cache under the git-ignored `scratch/` directory:

```bash
uv run --cache-dir scratch/uv-cache tools/video_job.py catalog
```

After installing the Python dependencies with `uv sync`, you can check local rendering without service
credentials or network access. From [examples/quick-spot](./examples/quick-spot/README.md), run
`uv run build.py` to render the silent `out.mp4`. If you use a workspace cache, point uv at the same
cache when installing dependencies and running the example.
