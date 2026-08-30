# Using with Kiro CLI

This toolkit is built around Claude Code assets in `.claude/` and `CLAUDE.md`, but it also ships a migration script for [Kiro CLI](https://kiro.dev/) — the sibling of `scripts/migrate_to_codex.py`.

```bash
uv run scripts/migrate_to_kiro.py --force
```

This does three things:

1. **Copies the toolkit skills into `~/.kiro/skills/`** — Kiro uses the same `SKILL.md` frontmatter format as Claude Code, so the domain-knowledge skills (remotion, ltx2, ideogram4, acestep, …) copy verbatim. The set is discovered dynamically from `_internal/toolkit-registry.json`, so newly added skills are picked up on the next run.
2. **Generates a wrapper skill per slash command** — Kiro invokes skills as `/name` slash commands with the same `$ARGUMENTS` placeholder Claude Code uses, so `/video`, `/setup`, `/scene-review`, etc. work identically. Each wrapper points at the original `.claude/commands/*.md` file (by absolute path) as the source of truth, so upstream command updates flow through without re-running the script.
3. **Generates `.kiro/steering/video-toolkit.md` from `CLAUDE.md`** — Kiro steering files are always-loaded context, the equivalent of `CLAUDE.md` in Claude Code. The content lives inside a managed marker block; manual content outside the block is preserved.

## Why skills install globally (Claude Code parity)

> **Claude Code walks up the directory tree** to find `.claude/`, so its commands work from any subdirectory of the repo (e.g. `projects/my-video/`). **Kiro resolves workspace `.kiro/` from the current directory only** — a workspace-scoped install would silently stop working the moment you `cd` into a project folder. Installing to `~/.kiro/skills` with absolute-path wrappers restores the Claude Code behavior: `/video` works from the repo root, any subdirectory, or anywhere else. The wrappers pin the toolkit's absolute path, so if you move or re-clone the repo, re-run the script with `--force`.

Prefer a repo-local install anyway? Use `--workspace-skills` — just remember it only loads when `kiro-cli` starts at the toolkit root.

> **Note:** if you already have a personal skill in `~/.kiro/skills` whose name matches a toolkit skill or command (e.g. `video`), the script refuses to touch it unless you re-run with `--force` — which overwrites it. Check for collisions first if you maintain your own skills.

## Usage

```bash
kiro-cli chat   # from anywhere
```

Then `/video`, `/setup`, `/brand`, etc. are available as slash commands (tab completion works: `/vid<Tab>`). When running from the toolkit root, Kiro also auto-loads the generated steering for full always-on context.

## Keeping it fresh

- After `CLAUDE.md` changes: re-run `uv run scripts/migrate_to_kiro.py --force` to refresh the steering file.
- After new skills/commands are added upstream: re-run with `--force` — the set is rediscovered from the registry.
- Command workflow edits need no re-run at all (wrappers read the originals at invocation time).
- Moved or re-cloned the repo: re-run with `--force` to refresh the absolute paths in the wrappers.

## Options

| Flag | Effect |
|------|--------|
| `--force` | Overwrite previously installed skills and refresh the steering block |
| `--dry-run` | Print the plan without writing anything |
| `--workspace-skills` | Install skills to the repo's `.kiro/skills` instead of `~/.kiro/skills` |
| `--reset` | Remove installed toolkit skills and the generated steering block |
| `--map-file` | Override `kiro/migration_map.json` (skip/rename skills and commands) |

## Removing

```bash
uv run scripts/migrate_to_kiro.py --reset
```

`--reset` removes the toolkit skills previously installed under `~/.kiro/skills` (or the workspace `.kiro/skills` with `--workspace-skills`) and removes the generated block from the steering file. It does not touch other skills you have installed and it does not remove manual steering content.
