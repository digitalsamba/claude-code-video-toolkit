# Feedback & Evolution

This toolkit is designed to grow through use. Every interaction is an opportunity to improve.

## How to Contribute

### Local Users (working with Claude Code)

1. **Tell Claude** - Just say "I have feedback" or "this could be better"
2. **Claude captures it** - Ideas go to BACKLOG.md, bugs get fixed
3. **Patterns emerge** - Common needs become new features
4. **Submit upstream** - `gh pr create` to share with others

### Remote Contributors (via GitHub)

- **Issues**: [github.com/digitalsamba/claude-code-video-toolkit/issues](https://github.com/digitalsamba/claude-code-video-toolkit/issues)
  - Bug reports, feature requests, questions
- **Pull Requests**: Fork, improve, submit PR
  - New templates, skills, commands
  - Documentation improvements
  - Bug fixes

---

## Quick Captures

<!-- Claude: Add user feedback here during sessions. Format: - [date] [area] description -->

### Recent Ideas


### Reported Issues

- [2025-12-10] [claude-code] **Slash commands not loading** - ✅ RESOLVED. Caused by `/skill` command conflicting with built-in `Skill` tool. Renamed to `/skills` and all commands now load correctly. Consider reporting to Anthropic - naming collision between custom commands and built-in tools silently breaks command loading.


---

## Evolution Principles

### Commands Evolve
Commands start simple and grow based on real usage:
- `/video` started as `/new-sprint-video`, expanded to multi-template, multi-session
- `/brand` merged creation and listing into one entry point
- Patterns that work get documented; patterns that don't get removed

### Skills Mature
Skills progress through maturity levels:
- **draft** → **beta** → **stable**
- Each real-world use validates or improves the skill
- Reference docs grow from actual questions and edge cases

### Templates Generalize
Templates extract patterns from projects:
- Build a project, notice reusable patterns
- Extract to template, parameterize the specifics
- Share components via `lib/` when used across templates

### The Toolkit Learns
Every session teaches the toolkit something:
- What workflows are awkward? → Improve commands
- What questions keep coming up? → Add to skill docs
- What's missing? → Add to BACKLOG.md

---

## Feedback Categories

### 🐛 Bugs
Something doesn't work as expected.
→ Fix immediately or document in BACKLOG.md

### 💡 Ideas
New features, commands, or improvements.
→ Capture in BACKLOG.md under appropriate section

### 📝 Documentation
Missing or unclear guidance.
→ Update relevant SKILL.md, command, or docs/

### 🔧 Workflow
Process improvements, better defaults.
→ Update commands or CLAUDE.md

### 🎨 Templates
New template types or component ideas.
→ Add to BACKLOG.md → Templates section

---

## Contributing Improvements

Your `projects/` directory is gitignored - your local video work stays private. Only toolkit files (commands, skills, templates, docs) are shared.

### Quick Contribution (from your working copy)

```bash
# 1. Check what you're about to share (projects/ won't appear)
git status

# 2. Create a branch for your improvement
git checkout -b improve/description

# 3. Stage only toolkit files (projects/ is ignored automatically)
git add .claude/ templates/ lib/ docs/ _internal/

# 4. Commit
git commit -m "Improve: description"

# 5. Create PR
gh pr create --title "Improve: description" --body "..."
```

### Clean Contribution (fresh clone)

If you want to be extra careful:

```bash
# Clone fresh
git clone https://github.com/digitalsamba/claude-code-video-toolkit ~/toolkit-contrib
cd ~/toolkit-contrib

# Copy only the files you improved
cp -r /path/to/your/work/.claude/commands/improved-command.md .claude/commands/

# Commit and PR
git checkout -b improve/description
git add -A
git commit -m "Improve: description"
gh pr create
```

### What's Safe to Share

| Directory | Shared? | Contains |
|-----------|---------|----------|
| `.claude/commands/` | ✅ Yes | Slash commands |
| `.claude/skills/` | ✅ Yes | Skill documentation |
| `templates/` | ✅ Yes | Video templates |
| `lib/` | ✅ Yes | Shared components |
| `docs/` | ✅ Yes | Documentation |
| `tools/` | ✅ Yes | Python CLI tools |
| `brands/` | ⚠️ Careful | Only share generic brands |
| `projects/` | ❌ No | Your private video work |
| `assets/voices/` | ❌ No | Your voice samples |

Target: `github.com/digitalsamba/claude-code-video-toolkit`

---

## Session History

<!-- Claude: Log significant improvements made during sessions -->

| Date | Change | Area |
|------|--------|------|
| 2024-12-10 | Added evolution narrative to commands | commands |
| 2024-12-10 | Created FEEDBACK.md | meta |

