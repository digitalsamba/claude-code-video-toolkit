# Project System

This directory contains the schema and documentation for multi-session video project tracking.

## Overview

Video projects span multiple Claude Code sessions. The project system provides:

1. **Structured state** via `project.json` in each project
2. **Filesystem reconciliation** - comparing intent vs reality
3. **Session continuity** - instant context on resume
4. **Auto-generated CLAUDE.md** - human+Claude readable status

## Project Lifecycle

```
planning → assets → audio → editing → rendering → complete
```

| Phase | Description | Exit Criteria |
|-------|-------------|---------------|
| `planning` | Defining scenes, writing script | VOICEOVER-SCRIPT.md complete, scenes defined |
| `assets` | Recording demos, gathering materials | All scene assets present |
| `audio` | Generating voiceover, music | Voiceover file exists |
| `editing` | Adjusting timing, previewing | Config matches assets, preview reviewed |
| `rendering` | Final render in progress | Output file generated |
| `complete` | Done | N/A |

## Schema

See `types.ts` for full TypeScript definitions. Key structures:

### project.json

```json
{
  "name": "my-release-video",
  "template": "sprint-review",
  "brand": "digital-samba",
  "created": "2024-12-09T10:30:00Z",
  "updated": "2024-12-10T15:45:00Z",
  "phase": "assets",

  "scenes": [...],
  "audio": {...},
  "estimates": {...},
  "sessions": [...]
}
```

### Scene Status

| Status | Meaning |
|--------|---------|
| `ready` | No asset needed (slides) or asset verified |
| `asset-needed` | Asset required but not created |
| `asset-present` | File exists, not yet verified |
| `asset-missing` | Was present but now missing (error) |

### Visual Types

| Type | Asset Required | How to Create |
|------|----------------|---------------|
| `slide` | No | Template generates |
| `playwright` | Yes | `/record-demo` command |
| `external` | Yes | User provides file |
| `screenshot` | Yes | Playwright or user provides |

## Filesystem Reconciliation

The project system follows these principles:

1. **project.json is intent** - What the user planned
2. **Filesystem is truth** - What actually exists
3. **Claude reconciles** - Updates status based on reality

### Reconciliation Logic

```
For each scene with visual.asset:

  If status = "asset-needed" AND file exists:
    → Update to "asset-present"
    → Suggest: "I found {file}, want to verify it in preview?"

  If status = "ready" AND file missing:
    → Update to "asset-missing"
    → Flag: "Asset {file} was removed, needs re-recording"

  If status = "asset-present" AND user confirms:
    → Update to "ready"
```

## Session History

The `sessions` array tracks work across Claude Code sessions:

```json
"sessions": [
  { "date": "2024-12-09", "summary": "Created project, planned 6 scenes" },
  { "date": "2024-12-10", "summary": "Recorded dark-mode and login demos" },
  { "date": "2024-12-11", "summary": "Generated voiceover, adjusted timing" }
]
```

This helps Claude understand context when resuming.

## Auto-Generated CLAUDE.md

Each project gets an auto-generated `CLAUDE.md` with:

- Current phase and status
- Scene checklist with ✅/⬜ markers
- Audio status
- Next actions
- Quick commands

This provides instant context even without running `/video`.

**Template:**

```markdown
# Project: {name}

**Template:** {template} | **Brand:** {brand} | **Phase:** {phase}
**Last Updated:** {relative_time}

## Scenes

| # | Scene | Type | Status |
|---|-------|------|--------|
| 1 | Title | title | ✅ Ready |
| 2 | Demo | demo | ⬜ Needs recording |

## Audio

- Voiceover: ⬜ Not generated
- Music: Optional

## Next Actions

1. {next_action_1}
2. {next_action_2}

## Commands

\`\`\`bash
npm run studio    # Preview
npm run render    # Final render
\`\`\`

---
*Auto-generated from project.json*
```

## Integration with Commands

### /video

The main entry point. Scans projects, offers resume or new.

### /record-demo

After recording, updates the scene's status:
- Sets `status: "asset-present"`
- Updates `visual.asset` path if needed
- Adds session entry

### /generate-voiceover

After generating:
- Sets `audio.voiceover.status: "present"`
- Updates phase to `audio` → `editing` if all assets ready
- Adds session entry

## Project Health

When scanning, projects are classified:

| Health | Condition |
|--------|-----------|
| `ready` | Can proceed to next phase |
| `blocked` | Missing required assets |
| `stale` | No updates in 7+ days, not complete |
| `complete` | Phase is "complete" |
