---
description: Generate AI voiceover from script
---

# Generate Voiceover

Help me generate a voiceover for a Remotion video project using ElevenLabs TTS.

## Project Integration

Before gathering configuration, check if we're in a project context:

1. **Check for active project:**
   - Look for `project.json` in current directory or parent `projects/*/`
   - If found, read it to understand project state

2. **Check review status:**
   If project has `phase: "review"` or no `reviewStatus` on scenes:
   ```
   ⚠️  Scene review not complete.

   Generating voiceover before review risks:
   - Narration that doesn't match visuals
   - Timing mismatches
   - Wasted API credits if script needs changes

   Run `/scene-review` first to verify each scene in Remotion Studio.

   Options:
   1. Run /scene-review first (recommended)
   2. Generate voiceover anyway (not recommended)
   ```

   Only proceed if user explicitly chooses option 2.

3. **If project context found and review complete:**
   ```
   I see you're working on: {project.name}

   Script: VOICEOVER-SCRIPT.md (ready)
   Audio status: ⬜ Not yet generated
   Review status: ✅ Complete

   Scenes requiring audio:
     - Overview (15s estimated)
     - Dark mode demo (20s estimated)
     - Login flow (10s estimated)
     - Summary (12s estimated)

   Total estimated: ~57 seconds of narration

   Ready to generate voiceover?
   ```

4. **After generation completes:**
   - Update `project.json`:
     - Set `audio.voiceover.status: "present"`
     - Transition `phase` if appropriate (review → audio → editing)
   - Add session entry
   - Regenerate project CLAUDE.md

---

## Your Tasks

1. **Detect Script Source**
   Look for a script in this order:
   - Check if `VOICEOVER-SCRIPT.md` exists in the current working directory
   - Check if `VOICEOVER-SCRIPT.md` exists in parent directories (up to 3 levels)
   - If not found, ask the user to provide the script text or file path

2. **Gather Configuration**
   Use the AskUserQuestion tool to collect:

   **Question 1 - Script Confirmation:**
   If a script file was found, show the first 100 characters and ask to confirm.
   Options:
   - Use this script
   - Enter different text
   - Specify different file path

   **Question 2 - Output Location:**
   Options:
   - `public/audio/voiceover.mp3` (default, relative to project)
   - Custom path (let user specify)

   **Question 3 - Voice Settings (optional):**
   Options:
   - Use defaults (stability: 0.85, similarity: 0.95)
   - Customize settings

3. **Execute Voiceover Generation**
   Run the voiceover tool:
   ```bash
   cd /Users/conalmullan/work/video
   python tools/voiceover.py \
     --script "SCRIPT_PATH" \
     --output "OUTPUT_PATH" \
     --json
   ```

   Or for custom text (pipe via stdin):
   ```bash
   cd /Users/conalmullan/work/video
   echo "USER_TEXT" | python tools/voiceover.py \
     --output "OUTPUT_PATH" \
     --json
   ```

4. **Report Results**
   Parse the JSON output and report:
   - File path of the saved MP3
   - Duration in seconds
   - Frame count at 30fps
   - The exact value to use in config: `durationSeconds: X`
   - Character count (for ElevenLabs quota awareness)

## Tool Location

- Voiceover tool: `/Users/conalmullan/work/video/tools/voiceover.py`
- Config: `_internal/toolkit-registry.json` (voice ID)
- API Key: `.env` file (`ELEVENLABS_API_KEY`)

## Voice Settings Reference

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| stability | 0.85 | 0-1 | Higher = more consistent, lower = more expressive |
| similarity | 0.95 | 0-1 | Higher = closer to original voice |
| style | 0.0 | 0-1 | Higher = more stylistic variation |
| speed | 1.0 | 0.5-2.0 | Speech speed multiplier |

## Script Format Tips

Share these tips with the user:
- Use `<break time="1.0s" />` for pauses (SSML-style)
- Keep sentences short for natural pacing
- Test with `--dry-run` flag first to check character count
- The voice ID is configured in `_internal/toolkit-registry.json`

## Error Handling

- If `ELEVENLABS_API_KEY` is missing, tell user to add it to `.env`
- If voice ID is missing, tell user to set `config.voiceId` in `toolkit-registry.json`
- If script file not found, offer to create a template

## Example Output

```
Voiceover generated successfully!

File: public/audio/voiceover.mp3
Duration: 45.2s (1356 frames @ 30fps)
Characters: 892

For your config:
  durationSeconds: 46
```

---

## Evolution

This command evolves through use. If something's awkward or missing:

**Local improvements:**
1. Say "improve this" → Claude captures in `_internal/BACKLOG.md`
2. Edit `.claude/commands/generate-voiceover.md` → Update `_internal/CHANGELOG.md`
3. Share upstream → `gh pr create`

**Remote contributions:**
- Issues: `github.com/digitalsamba/claude-code-video-toolkit/issues`
- PRs welcome for voice features, script formats, documentation
