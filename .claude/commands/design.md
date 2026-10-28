# Design Refinement

Focused design session for improving video scene visuals using the frontend-design skill.

## Purpose

Deep-dive visual refinement for slides and components:
- Elevate generic aesthetics to distinctive, memorable visuals
- Ensure visual narrative coherence across scenes
- Iterate on typography, color, motion, and atmosphere

## Usage

```
/design              # Discover project, list scenes, choose one
/design [scene-id]   # Jump directly to a specific scene
```

---

## Step 1: Discover Project Context

First, find and understand the current project:

1. **Locate the project**
   - Check if in a project directory (has `project.json` or template config)
   - If not, scan `projects/` for active projects
   - If multiple, ask user which project

2. **Read project configuration**
   - `project.json` if exists (scenes, phase, brand)
   - Or template config: `src/config/sprint-config.ts` or `src/config/demo-config.ts`
   - Brand/theme: `src/config/theme.ts`

3. **Build scene inventory**
   - List all scenes from config
   - Identify which are slide-type (designable) vs demo-type (video playback)
   - Note each scene's purpose from config content

**math-story-short projects are different.** Scenes live in `src/config/story-config.ts` and are
grouped by `visual` key; one component per group in `src/episode/visuals.tsx` receives the scene
boundaries as beats. Design there means: read the template's `CLAUDE.md` (layout archetypes
showcase / presenter / set piece, type scale, "objects, not icons", scene recipes), pick the
archetype per group, and check the result with `npm run stills` (one PNG per scene end, plus
`--frames` for mid-animation moments) rather than a video render. Brand fonts for it include a
display serif (Fraunces 900) for the hero numeral of each scene.

---

## Step 2: Identify Target Scene

If no scene specified in command, show discovered scenes:

```
## Design Refinement

**Project:** {project_name}
**Template:** {template_type}

Which scene would you like to refine?

| # | Scene ID | Type | Description |
|---|----------|------|-------------|
{list slide-type scenes discovered from config}

(Demo/video scenes are not shown - they display recorded content)

Enter scene ID or number:
```

---

## Step 3: Understand Scene Context

Before invoking frontend-design, gather specific context:

1. **Read the target slide component**
   - Find component file from scene type mapping
   - e.g., `src/components/slides/{Type}Slide.tsx`

2. **Read scene content from config**
   - What text/data does this scene display?
   - What's the intended message?

3. **Understand position in sequence**
   - What scene comes before? What mood does it leave?
   - What scene comes after? What mood should we set up?

4. **Check current mood/intent** (if defined in project.json)
   - If scene has `designNotes.mood`, use that
   - If not, infer from content or ask user

---

## Step 4: Establish Scene Mood

If the scene doesn't have a defined mood, ask the user:

```
## Scene Mood

What feeling should the "{scene_id}" scene evoke?

The scene contains: {brief content summary}
It follows: {previous scene summary}
It leads to: {next scene summary}

Suggested moods based on content:
- {suggestion_1}
- {suggestion_2}

Or describe the mood you want:
```

Common mood dimensions:
- **Energy**: calm ↔ energetic
- **Tone**: serious ↔ playful
- **Warmth**: cool/professional ↔ warm/inviting
- **Trust**: edgy/provocative ↔ safe/reliable

---

## Step 5: Invoke frontend-design Skill

Use the Skill tool to invoke `frontend-design`.

Provide context gathered:
- Scene purpose and content
- Target mood
- Adjacent scenes and their moods
- Brand constraints (colors, fonts)
- Technical constraints (Remotion/React, duration)

---

## Step 6: Iterative Refinement

Work with user in a back-and-forth loop:

1. **Propose changes** - Explain the design direction
2. **Implement** - Edit the component file
3. **Preview** - User checks Remotion Studio (hot-reload)
4. **Feedback** - User responds with what works/doesn't
5. **Refine** - Adjust based on feedback
6. **Repeat** until satisfied

**Common refinement areas:**
- Background (gradients, effects, atmosphere)
- Typography (font choice, size, animation)
- Color palette (mood, contrast, warmth/coolness)
- Motion (timing, easing, entrance/exit)
- Decorative elements (particles, shapes, glows)
- Visual hierarchy (what draws the eye)

---

## Step 7: Ensure Coherence

After refining a scene, consider its neighbors:

```
Does this scene:
- [ ] Flow naturally from the previous scene?
- [ ] Set up the next scene appropriately?
- [ ] Maintain consistent visual language with brand?
- [ ] Create appropriate contrast where needed?
```

If adjacent scenes need adjustment, offer to refine them too.

---

## Step 8: Document Changes

Update project.json (if exists) with refinement notes:

```json
{
  "scenes": [
    {
      "id": "{scene_id}",
      "designNotes": {
        "mood": "{established mood}",
        "refinements": "{description of visual changes made}"
      }
    }
  ]
}
```

Add session entry:
```json
{
  "sessions": [
    {
      "date": "YYYY-MM-DD",
      "summary": "Design refinement on {scene_id}: {brief description}"
    }
  ]
}
```

---

## Design Principles for Video

When working with frontend-design skill on video scenes:

1. **Video scale** - Everything needs to be larger than web UI (viewers are farther from screen). On 9:16
   shorts nothing readable goes under ~44px and each scene has one huge hero element (200px+)
2. **One layout per scene, not one layout per video** - vary full-bleed showcases, presenter scenes and
   set pieces; a stat tile only earns its place when its value changes on screen
3. **Motion matters** - Static designs that work on web may feel dead on video
4. **Time constraints** - Each scene has limited duration; don't over-animate
5. **Narrative arc** - Consider how visual intensity builds across scenes
6. **Contrast creates impact** - A calm scene makes the next dramatic scene hit harder

---

## Scene Type Reference

These are common scene types, but projects may have different ones:

| Type | Common Purpose | Design Considerations |
|------|----------------|----------------------|
| title | Opening, brand intro | Set the mood, establish visual language |
| overview | Agenda, summary | Clarity, scannable layout |
| demo | Show functionality | Clean chrome, focus on content |
| summary | Recap, stats | Data visualization, credibility |
| credits | Closing | Graceful exit, brand reinforcement |
| problem | Pain points | Tension (if narrative calls for it) |
| solution | Resolution | Relief, hope |
| cta | Call to action | Energy, memorability |

**Note**: Not all projects use all types. The command discovers what's actually in the project.

---

## Example Session

```
User: /design

Claude: Let me discover your project...

Found project: "q4-release-video"
Template: sprint-review
Brand: digital-samba

## Design Refinement

Which scene would you like to refine?

| # | Scene ID | Type | Description |
|---|----------|------|-------------|
| 1 | title | title | "What's New in v2.5" |
| 2 | overview | overview | 4 feature highlights |
| 3 | summary | summary | Sprint stats |
| 4 | credits | credits | Team acknowledgment |

(3 demo scenes not shown)

User: 1

Claude: Let me gather context for the title scene...

[Reads TitleSlide.tsx, sprint-config.ts, theme.ts]
[Reads overview scene to understand what follows]

The title announces "What's New in v2.5" for Digital Samba.
It's followed by an overview of 4 features.

What mood should this title scene set?
- Excited/energetic (big release energy)
- Professional/trustworthy (enterprise feel)
- Warm/welcoming (team celebration)

User: Warm and celebratory - this was a big team effort

Claude: Got it - warm, celebratory mood for a team achievement.

Invoking frontend-design skill...

[frontend-design provides direction]

I'll implement:
1. Warm gradient (brand orange softened with cream tones)
2. Gentle confetti particles (subtle, not overwhelming)
3. Smooth text reveal with slight bounce
4. Soft glow behind version number

[Makes changes]

Check Remotion Studio - how does it feel?
```
