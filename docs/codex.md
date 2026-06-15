# Using with Codex

This toolkit is built around Claude Code assets in `.claude/` and `CLAUDE.md`, but it
also ships an experimental migration script for [Codex](https://openai.com/codex/) —
contributed by [@kimhoontae-gogo](https://github.com/kimhoontae-gogo) in
[#16](https://github.com/digitalsamba/claude-code-video-toolkit/pull/16).

```bash
python3 scripts/migrate_to_codex.py --force
```

This does two things:

1. **Installs Codex skills into `~/.codex/skills`** — one skill per directory found in
   `.claude/skills/` (domain knowledge: remotion, ltx2, ideogram4, acestep, …) plus the
   guided workflows from `.claude/commands/` (video, brand, voice-clone, …). The set is
   discovered dynamically, so newly added toolkit skills are picked up on the next run;
   the script prints exactly what it installed.
2. **Generates a Codex block in the repository root `AGENTS.md`**, derived from
   `CLAUDE.md`.

## The AGENTS.md block

- The script manages **only** its generated block inside `AGENTS.md`.
- Manual `AGENTS.md` content outside that block is preserved.
- The block is derived from `CLAUDE.md` — after `CLAUDE.md` changes, re-run
  `python3 scripts/migrate_to_codex.py --force` to refresh it.

## Removing

```bash
python3 scripts/migrate_to_codex.py --reset
```

`--reset` removes the toolkit skills previously installed under `~/.codex/skills` and
removes the generated Codex block from `AGENTS.md`. It does not delete other user
skills and it does not remove the rest of `AGENTS.md`.
