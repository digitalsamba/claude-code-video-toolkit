# Using with Kiro CLI

This toolkit is built around Claude Code assets in `.claude/` and `CLAUDE.md`, but it also ships a migration script for [Kiro CLI](https://kiro.dev/) — the sibling of `scripts/migrate_to_codex.py`.

```bash
python3 scripts/migrate_to_kiro.py --force
```

This does three things:

1. **Copies the toolkit skills into `.kiro/skills/`** — Kiro uses the same `SKILL.md` frontmatter format as Claude Code, so the domain-knowledge skills (remotion, ltx2, ideogram4, acestep, …) copy verbatim. The set is discovered dynamically from `_internal/toolkit-registry.json`, so newly added skills are picked up on the next run.
2. **Generates a wrapper skill per slash command** — Kiro invokes skills as `/name` slash commands with the same `$ARGUMENTS` placeholder Claude Code uses, so `/video`, `/setup`, `/scene-review`, etc. work identically. Each wrapper points at the original `.claude/commands/*.md` file as the source of truth, so upstream command updates flow through without re-running the script.
3. **Generates `.kiro/steering/video-toolkit.md` from `CLAUDE.md`** — Kiro steering files are always-loaded context, the equivalent of `CLAUDE.md` in Claude Code. The content lives inside a managed marker block; manual content outside the block is preserved.

By default everything installs into the repository workspace (`.kiro/`, gitignored), so the toolkit context only loads when running Kiro from this repo. Use `--global-skills` to install the skills to `~/.kiro/skills` instead — steering always stays workspace-scoped.

## Usage

```bash
cd claude-code-video-toolkit
kiro-cli chat
```

Then `/video`, `/setup`, `/brand`, etc. are available as slash commands, and Kiro loads the toolkit instructions automatically (it reads both `AGENTS.md` and `.kiro/steering/` by default).

## Keeping it fresh

- After `CLAUDE.md` changes: re-run `python3 scripts/migrate_to_kiro.py --force` to refresh the steering file.
- After new skills/commands are added upstream: re-run with `--force` — the set is rediscovered from the registry.
- Command workflow edits need no re-run at all (wrappers read the originals at invocation time).

## Options

| Flag | Effect |
|------|--------|
| `--force` | Overwrite previously installed skills and refresh the steering block |
| `--dry-run` | Print the plan without writing anything |
| `--global-skills` | Install skills to `~/.kiro/skills` instead of workspace `.kiro/skills` |
| `--reset` | Remove installed toolkit skills and the generated steering block |
| `--map-file` | Override `kiro/migration_map.json` (skip/rename skills and commands) |

## Removing

```bash
python3 scripts/migrate_to_kiro.py --reset
```

`--reset` removes the toolkit skills previously installed under `.kiro/skills` (or `~/.kiro/skills` with `--global-skills`) and removes the generated block from the steering file. It does not delete other skills and it does not remove manual steering content.
