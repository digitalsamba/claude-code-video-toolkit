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

### Integrations that depend on a hosted service or local software

An integration that needs a hosted third-party service, or software we can't bundle, lives in
your own repo. Open an issue with a link and a one-line description and we'll list it under
**Community add-ons** in the README.

The toolkit does carry a few hosted providers: ElevenLabs, 60db, Ideogram 4 and acemusic. Each is
there because a maintainer uses it, holds an account, and can run it against the live API when
something breaks. That is the bar, and a PR can't meet it on a maintainer's behalf — code we
can't run is code we can't keep working.

If you think a hosted service belongs in-tree, **open an issue before writing any code**. Say
what it does that the toolkit can't already do; another place to run a model we already run
isn't that. If a maintainer wants to adopt it, we'll take it from there.

### Design skills

Plug in your own design skill, or use ours: the toolkit's own source is `video-design` with the brand, and it bundles
no third-party design skill. `/design` and `/scene-review` take direction from whichever one the
user has installed (the README's "Design sources"), and `video-design` applies the video rules. To have yours
listed, open an issue with a link, its licence (it must allow use in published videos), the `design.md` it
produced for `examples/data-chart-stars`, and that example's contact sheet
(`uv run tools/contact_sheet.py examples/data-chart-stars/out/data-chart-stars.mp4`), so sources can be compared
on the same video. Listed skills stay in their own repos. Details: `.claude/skills/video-design/sources.md`.

### Brand profiles and new templates

If your video differs from an existing template only in colors, fonts, logo, narrator voice, or
timing values, you want a **brand profile**, not a new template. Templates take their palette and
voice from a brand (`voice.brand` in `config.json` resolves `brands/<name>/voice.json`), and
timings like `lead`/`tail`/`xfade` belong in your own project's `config.json`. Copying a template
to change those leaves you with a fork that no longer inherits fixes to the original.

Brand profiles work fine without being committed here, so keep yours local unless other people
would genuinely use it. A new template in-tree needs to differ in its composition code — new
scenes, structure, or rendering — not just its look.

### Automated and AI-generated PRs

Automated or AI-generated PRs are welcome only when a human author responds to review and the
PR states why the toolkit needs the feature — cost, license, and how it differs from the tools
already here. PRs with no human response within 14 days are closed without further review.
Integrations with hosted services follow the section above whoever, or whatever, wrote them.

## Toolkit Tracking Files

| File | Purpose |
|------|---------|
| `_internal/ROADMAP.md` | What we're building (phases, current work) |
| `_internal/BACKLOG.md` | What we might build (unscheduled ideas) |
| `_internal/CHANGELOG.md` | What we built (historical record) |

For more details on the toolkit's evolution principles and local contribution workflow, see `docs/contributing.md`.

## Questions?

Open an issue for questions or discussions.
