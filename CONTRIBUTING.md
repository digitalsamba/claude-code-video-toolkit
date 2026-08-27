# Contributing to claude-code-video-toolkit

Thank you for your interest in contributing! This toolkit is designed to help people create videos with Claude Code assistance.

## Ways to Contribute

### Report Issues
- Bug reports
- Feature requests
- Documentation improvements

### Submit Pull Requests
- Bug fixes
- New templates
- New skills or commands
- Documentation updates

## Development Setup

1. Fork and clone the repository
2. Set up your environment with [uv](https://docs.astral.sh/uv/) (creates `.venv/` and installs all locked dependencies):
   ```bash
   uv sync
   ```
3. Add your ElevenLabs API key to `.env`

## Project Structure

```
├── .claude/skills/     # Domain knowledge for Claude Code
├── .claude/commands/   # Guided workflow commands
├── tools/              # Python CLI tools
├── templates/          # Video templates
├── brands/             # Brand profiles
└── docs/               # Documentation
```

## Adding a New Template

1. Create a new folder in `templates/`
2. Include a working Remotion project
3. Add a `README.md` explaining the template
4. Register it in `_internal/toolkit-registry.json`
5. **Update documentation** (see checklist below)
6. Test with `npm run studio` and `npm run render`

## Adding a New Skill

1. Create a folder in `.claude/skills/`
2. Add `SKILL.md` with the skill definition
3. Optionally add `reference.md` for detailed docs
4. Register it in `_internal/toolkit-registry.json`
5. **Update documentation** (see checklist below)
6. Test by asking Claude Code questions about the domain

## Adding a New Command

1. Create a markdown file in `.claude/commands/`
2. Follow the existing command format
3. Register it in `_internal/toolkit-registry.json`
4. **Update documentation** (see checklist below)
5. Test by running the command in Claude Code

## Documentation Checklist

When adding or modifying commands, skills, or templates, update these files:

| What Changed | Update These Files |
|--------------|-------------------|
| New command | `README.md` (Commands table), `CLAUDE.md` (Commands section) |
| New skill | `README.md` (Skills table), `CLAUDE.md` (Skills Reference) |
| New template | `README.md` (Templates section), `CLAUDE.md` (Templates section) |
| New component | `CLAUDE.md` (Shared Components table) |
| New transition | `README.md` (Scene Transitions), `lib/transitions/README.md` |

If your change affects Codex compatibility, also update:

| What Changed | Update These Files |
|--------------|-------------------|
| Codex migration flow | `README.md` ("Using with Codex"), `docs/getting-started.md`, `scripts/migrate_to_codex.py` |
| Claude guidance source | `CLAUDE.md` and then re-run `uv run scripts/migrate_to_codex.py --force` to regenerate the Codex block in `AGENTS.md` |
| Generated resource list or warnings | `README.md` and `docs/getting-started.md` |

If your change affects Kiro CLI compatibility, also update:

| What Changed | Update These Files |
|--------------|-------------------|
| Kiro migration flow | `README.md` ("Using with Kiro CLI"), `docs/kiro.md`, `docs/getting-started.md`, `scripts/migrate_to_kiro.py` |
| Claude guidance source | `CLAUDE.md` and then re-run `uv run scripts/migrate_to_kiro.py --force` to regenerate the steering file in `.kiro/steering/` |

**Quick verification:** After adding a command, grep for it across docs:
```bash
grep -r "/your-command" README.md CLAUDE.md
```

## Code Style

- Use clear, descriptive names
- Comment complex logic
- Follow existing patterns in the codebase

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Update documentation if needed
5. Submit a PR with a clear description

### Integrations that depend on a single paid service or local software

If your integration only works against one paid API or a piece of software we can't bundle, keep
it in your own repo and open an issue to be listed under **Community add-ons** in the README.
In-tree integrations need at least one open or self-hostable path.

### Automated and AI-generated PRs

Automated or AI-generated PRs are welcome only when a human author responds to review and the
PR states why the toolkit needs the feature — cost, license, and how it differs from the tools
already here (e.g. a hosted alternative to a self-hosted generator must say when to choose it).
PRs with no human response within 14 days are closed without further review.

## Toolkit Tracking Files

| File | Purpose |
|------|---------|
| `_internal/ROADMAP.md` | What we're building (phases, current work) |
| `_internal/BACKLOG.md` | What we might build (unscheduled ideas) |
| `_internal/CHANGELOG.md` | What we built (historical record) |

For more details on the toolkit's evolution principles and local contribution workflow, see `docs/contributing.md`.

## Questions?

Open an issue for questions or discussions.
