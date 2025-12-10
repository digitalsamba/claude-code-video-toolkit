---
description: Video projects - list, resume, or create new
---

# Video

Unified command for video projects. Scans for existing projects and offers to resume or create new.

## Entry Point Logic

On invocation, scan for projects and adapt:

### Step 1: Scan Projects

```
1. Glob projects/*/project.json
2. For each project found:
   - Read project.json
   - Check filesystem for actual assets (public/demos/*, public/audio/*)
   - Reconcile status vs reality
   - Calculate health (ready/blocked/stale/complete)
3. Sort by last updated (most recent first)
```

### Step 2: Present Options

**No projects found:**
```
No video projects found. Let's create one!

Which template would you like to use?
  1. Sprint Review - internal demos, release videos
  2. Product Demo - marketing, launches, features
```
→ Proceed to New Project Flow

**One project found:**
```
Found: **my-release-video** (sprint-review)
  Phase: assets
  Progress: 2 of 5 demos recorded
  Last worked: 2 days ago

  → Resume this project
  → Start a new project
```

**Multiple projects found:**
```
Found 3 video projects:

  1. **my-release-video** (sprint-review)
     Phase: assets - 2/5 demos recorded
     Last worked: 2 days ago

  2. **product-launch** (product-demo)
     Phase: audio - voiceover needed
     Last worked: 5 days ago

  3. **tutorial-v1** (sprint-review)
     ✅ Complete
     Rendered: 2 weeks ago

Which project? (or 'new' for a new project)
```

---

## Resume Flow

When resuming a project:

### Step 1: Read State

```
1. Read project.json
2. Read VOICEOVER-SCRIPT.md for context
3. Glob public/demos/* and public/audio/*
4. Compare to project.json expectations
```

### Step 2: Reconcile Filesystem

For each scene with `visual.asset`:
- If `status: "asset-needed"` but file exists → update to `"asset-present"`
- If `status: "ready"` but file missing → update to `"asset-missing"`, flag issue

For audio:
- Check if `public/audio/voiceover.mp3` exists
- Update `audio.voiceover.status` accordingly

### Step 3: Present Current State

```
Resuming: my-release-video (sprint-review)

## Scenes

| # | Scene | Type | Status |
|---|-------|------|--------|
| 1 | Title | title | ✅ Ready |
| 2 | Overview | overview | ✅ Ready |
| 3 | Dark Mode Demo | demo | ✅ Recorded |
| 4 | Login Flow | demo | ⬜ Needs recording |
| 5 | Stats | stats | ✅ Ready (slide) |
| 6 | Credits | credits | ✅ Ready (slide) |

## Audio

- Voiceover: ⬜ Not yet generated (script ready)
- Music: Optional

## Next Actions

1. **Record login flow demo** (Scene 4)
   Run `/record-demo` or provide external video

2. **Generate voiceover** (after demos complete)
   Run `/generate-voiceover`

Ready to record the login flow?
```

### Step 4: Update project.json

After reconciliation:
- Update any changed statuses
- Add session entry: `{ date: "YYYY-MM-DD", summary: "Resumed project" }`
- Update `updated` timestamp

### Step 5: Regenerate CLAUDE.md

Update the project's CLAUDE.md with current state (see CLAUDE.md Generation below).

---

## New Project Flow

When creating a new project:

### Phase 1: Setup

**Template Selection:**
```
Which template would you like to use?

| Template | Best For | Scene Types |
|----------|----------|-------------|
| Sprint Review | Internal demos, release videos | title, overview, demo, split-demo, summary, credits |
| Product Demo | Marketing, launches, features | title, problem, solution, demo, feature, stats, cta |
```

**Brand Selection:**

List available brands from `brands/` directory:
```
1. Read each brands/*/brand.json
2. Extract name and primary color
3. Present list with "Create new brand" option
```

```
Which brand?

  1. default - #3B82F6 (blue)
  2. digital-samba - #FF6B00 (orange)
  3. Create new brand → /brand
```

**Project Name:**
```
What should we call this project?
(This becomes the folder name, e.g., "v21-release" → projects/v21-release/)
```

### Phase 2: Content Gathering

```
What content should this video cover? You can:
- Paste release notes or bullet points
- Provide a URL (release notes, changelog, JIRA, GitHub)
- Describe what you want to show
- Any combination of the above
```

**If URL provided:**
- WebFetch the content
- Parse and summarize
- Ask clarifying questions

**Always ask:**
- "What's the main message or announcement?"
- "Who's the audience?" (internal team, customers, developers)
- "Any specific features to highlight?"
- "Anything to skip or de-emphasize?"

### Phase 3: Scene Planning

**Step 1: Propose scene breakdown**

Based on content, propose scenes:
```
Based on your input, here's a suggested scene breakdown:

  1. Title (5s) - "Version 2.1 Release"
     [SLIDE: Title with logo]

  2. Overview (15s) - Key highlights
     [SLIDE: 3 bullet points]

  3. Demo: Dark Mode (20s)
     [DEMO: Settings → toggle → show UI change]
     "One of our most requested features..."

  4. Stats (10s)
     [SLIDE: 3 features, 5 fixes]

  5. Credits (5s)
     [SLIDE: Team credits]

  Estimated total: ~55 seconds

Want to adjust any scenes?
```

**Step 2: Iterate**

Help refine:
- Add/remove scenes
- Reorder
- Adjust timing
- Change visual type

**Step 3: Confirm each scene**

| Aspect | Options |
|--------|---------|
| Type | title, overview, demo, split-demo, stats, credits, problem, solution, feature, cta |
| Visual | slide, playwright, external, screenshot |
| Narration | Draft voiceover text |
| Duration | Estimated seconds |

### Phase 4: Project Creation

1. **Copy template:**
   ```bash
   cp -r templates/{template}/ projects/{name}/
   ```

2. **Generate project.json:**
   Use schema from `lib/project/types.ts`:
   ```json
   {
     "name": "{name}",
     "template": "{template}",
     "brand": "{brand}",
     "created": "{ISO timestamp}",
     "updated": "{ISO timestamp}",
     "phase": "planning",
     "scenes": [...],
     "audio": {
       "voiceover": { "file": "audio/voiceover.mp3", "status": "needed" }
     },
     "estimates": { "totalDurationSeconds": N },
     "sessions": [{ "date": "YYYY-MM-DD", "summary": "Project created" }]
   }
   ```

3. **Generate brand.ts:**
   Read `brands/{brand}/brand.json` → generate `src/config/brand.ts`

4. **Write VOICEOVER-SCRIPT.md:**
   Full script with narration and asset markers

5. **Copy brand assets:**
   Logo from `brands/{brand}/assets/` to `public/images/`

6. **Generate CLAUDE.md:**
   Auto-generated project status (see below)

7. **Install dependencies:**
   ```bash
   cd projects/{name} && npm install
   ```

### Phase 5: Asset Guidance

After creation, actively guide asset creation:

```
Project created at: projects/v21-release/

You have 2 assets to create:

  ⬜ Scene 3: Dark mode demo (20s)
     → Record with /record-demo or provide external video

  ⬜ Scene 4: Stats dashboard screenshot
     → Capture with Playwright or provide image

Which would you like to start with?
```

**Track progress:**
- Update project.json as assets complete
- Update phase when appropriate:
  - All scenes defined → `phase: "assets"`
  - All assets present → `phase: "audio"`
  - Voiceover generated → `phase: "editing"`

---

## CLAUDE.md Generation

Generate/update the project's CLAUDE.md after any state change:

```markdown
# Project: {name}

**Template:** {template} | **Brand:** {brand} | **Phase:** {phase}
**Last Updated:** {relative time, e.g., "2 days ago"}

## Current Status

{PHASE_DESCRIPTIONS[phase]}

## Scenes

| # | Scene | Type | Visual | Status |
|---|-------|------|--------|--------|
| 1 | Title | title | slide | ✅ Ready |
| 2 | Dark Mode Demo | demo | demos/dark-mode.mp4 | ✅ Recorded |
| 3 | Login Flow | demo | demos/login.mp4 | ⬜ Needs recording |

## Audio

- Voiceover: {status} {details}
- Music: {status}

## Next Actions

1. {Priority action based on phase and blockers}
2. {Secondary action}

## Quick Commands

```bash
cd projects/{name}
npm run studio    # Preview in browser
npm run render    # Final render
```

## Session History

{List from sessions array}

---
*Auto-generated from project.json. Do not edit manually.*
```

---

## Phase Transitions

Update `phase` automatically based on state:

| Condition | New Phase |
|-----------|-----------|
| Scenes defined, script written | `planning` → `assets` |
| All scene assets present | `assets` → `audio` |
| Voiceover generated | `audio` → `editing` |
| User initiates render | `editing` → `rendering` |
| Render complete | `rendering` → `complete` |

---

## Integration with Other Commands

### /record-demo

After recording completes:
1. Move file to project's `public/demos/`
2. Update scene's `status: "asset-present"`
3. Update scene's `visual.asset` path
4. Add session entry
5. Regenerate CLAUDE.md

### /generate-voiceover

After generation:
1. Move file to project's `public/audio/`
2. Update `audio.voiceover.status: "present"`
3. Transition phase if appropriate
4. Add session entry
5. Regenerate CLAUDE.md

---

## Key Principles

1. **Scan first** - Always know what exists before asking what to do
2. **Reconcile reality** - Filesystem is truth, project.json is intent
3. **Track sessions** - Help Claude understand context across sessions
4. **Auto-generate CLAUDE.md** - Always-current human+Claude readable status
5. **Guide next action** - Never leave user wondering what's next
6. **Stay flexible** - Handle partial states, missing files, manual edits

---

## Evolution

This command evolves through use. If something's awkward or missing:

**Local improvements:**
1. Say "improve this" → Claude captures in `_internal/FEEDBACK.md`
2. Edit `.claude/commands/video.md` → Update `_internal/CHANGELOG.md`
3. Share upstream → `gh pr create`

**Remote contributions:**
- Issues: `github.com/digitalsamba/claude-code-video-toolkit/issues`
- PRs welcome for new features, bug fixes, documentation

History: `/new-sprint-video` → `/new-video` → `/video` (unified with resume support)
