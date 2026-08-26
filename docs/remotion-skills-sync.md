# Remotion Skills: Official + Toolkit Split

## Why Two Skills?

The toolkit maintains two complementary Remotion skills:

| Skill | Path | Source | Content |
|-------|------|--------|---------|
| `remotion-official` | `.claude/skills/remotion-official/` | [remotion-dev/skills](https://github.com/remotion-dev/skills) `skills/remotion-best-practices/` | Core framework knowledge (hooks, animations, rendering, media, maps, captions, etc.) |
| `remotion` | `.claude/skills/remotion/` | This toolkit | Custom transitions, shared components, project conventions |

**Before the split**, our `remotion/SKILL.md` duplicated core Remotion documentation that quickly became outdated. The official skill repo is maintained by the Remotion team and stays current with framework releases.

## What Lives Where

### remotion-official (upstream-managed)

Upstream reorganised into one skill per topic in mid-2026 (`remotion-markup`, `remotion-maps`,
`remotion-captions`, `remotion-render`, …). We sync only `skills/remotion-best-practices/`: it is
the **router skill** whose `SKILL.md` links into a nested copy of every other skill
(`remotion-markup/REFERENCE.md`, `remotion-captions/REFERENCE.md`, …), so it is the complete,
self-contained knowledge set in one directory. It must remain the only `SKILL.md` in the tree —
the nested skills are deliberately shipped as `REFERENCE.md` so they don't register as separate
Claude Code skills.

- Animation APIs (`interpolate`, `spring`, easing)
- Composition registration and config
- Sequencing (`Sequence`, `Series`, `Loop`, `Freeze`)
- Media components (`Video`, `Audio`, `Img`, `OffthreadVideo`)
- Static files and assets
- Input props and async data loading
- CLI and programmatic rendering
- Lambda deployment
- Player component
- Captions, charts, 3D, fonts, Tailwind, etc.

### remotion (toolkit-managed)
- Custom transition library (`lib/transitions/`)
- Shared component catalog (`lib/components/`)
- Toolkit best practices and conventions
- Project timing guidelines
- Transition duration guidelines

## Automatic Sync

A GitHub Actions workflow (`.github/workflows/sync-remotion-skills.yml`) runs weekly to check for upstream changes:

1. Clones `remotion-dev/skills` (shallow)
2. Verifies `skills/remotion-best-practices/SKILL.md` exists — fails loudly with a directory
   listing if upstream re-organises again (update `UPSTREAM_SKILL_DIR` in the workflow)
3. Compares it against our `.claude/skills/remotion-official/`
4. Opens a PR if files have changed (title carries the upstream skill version + commit)

### Manual Sync

To sync manually:

```bash
# Clone upstream
git clone --depth 1 https://github.com/remotion-dev/skills.git /tmp/remotion-skills

# Copy into toolkit
rm -rf .claude/skills/remotion-official
cp -r /tmp/remotion-skills/skills/remotion-best-practices .claude/skills/remotion-official

# Commit
git add .claude/skills/remotion-official
git commit -m "Sync official Remotion skills ($(git -C /tmp/remotion-skills rev-parse --short HEAD))"
```

### Handling Sync PRs

When the automated PR arrives:
1. Skim the diff for any breaking changes
2. Grep the diff for `OffthreadVideo` / `@remotion/media` — the toolkit deliberately stays on
   `<OffthreadVideo>` / `<Audio>` from `remotion` (see CLAUDE.md "Media"); upstream teaches
   `@remotion/media`. That is a known, accepted divergence, not a reason to block the sync.
3. Check if new guidance overlaps with our toolkit skill (unlikely but possible)
4. Merge if everything looks good
