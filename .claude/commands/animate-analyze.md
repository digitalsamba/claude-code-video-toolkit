# Animate Analyze

Interactive element identification for character animation. Claude Code analyzes each frame and the user confirms before writing the manifest.

## Purpose

Identify animatable elements in slide-based videos (NotebookLM, presentations):
- Characters, people, avatars
- Animals, creatures
- Environmental elements (water, fire, clouds, foliage)
- Decorative elements with natural motion potential

**Skip:** Text, logos, UI elements, charts - these should NOT be animated.

## Entry Point

User provides: video path (required)
Optional: output directory

```
/animate-analyze path/to/video.mp4
/animate-analyze path/to/video.mp4 --output-dir ./my-output
```

---

## Step 1: Setup and Scene Detection

```bash
# Activate venv
source .venv/bin/activate

# Create output directory
OUTPUT_DIR={provided or video_directory/animation_analysis}
mkdir -p "$OUTPUT_DIR/frames"

# Run scene detection and frame extraction
python tools/animate.py --input {video_path} --analyze --output "$OUTPUT_DIR/manifest.json" --output-dir "$OUTPUT_DIR"
```

**Tell the user:**
```
Detected {N} scenes in the video.
Frames extracted to: {OUTPUT_DIR}/frames/

I'll now analyze each frame and show you what I find.
You can approve, edit, or skip each scene's elements.
```

---

## Step 2: Load Initial Manifest

Read the generated manifest.json to get scene list:
- scene_id
- start_time, end_time, duration
- frame_path

---

## Step 3: Scene-by-Scene Analysis

For each scene in the manifest:

### 3a. Read the Frame

Use the Read tool to view the frame image:
```
Read: {frame_path}
```

### 3b. Analyze and Present

After viewing the frame, identify animatable elements and present to user:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Scene {N} of {total}

**Time:** {start_time}s - {end_time}s ({duration}s)
**Frame:** {frame_path}

### Elements I Found:

{For each identified element:}

**{N}. {element_id}** ({type})
- Description: {what it is}
- Location: {approximate position - left/center/right, top/middle/bottom}
- Suggested bbox: [{x}, {y}, {width}, {height}]
- Motion style: {suggested motion - e.g., "gentle breathing, slight weight shift"}
- Motion intensity: {subtle/medium/active}

{Or if no elements:}
No animatable elements found in this scene.
(Text-heavy slide, static background, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Options:**
1. Approve - accept these elements as-is
2. Edit - modify elements (add, remove, adjust bbox)
3. Skip - no animation for this scene
4. View frame - open the frame image in Finder
```

### 3c. Handle Response

#### Approve
- Add elements to manifest for this scene
- Record motion_bucket_id based on intensity:
  - subtle: 90
  - medium: 127
  - active: 180
- Proceed to next scene

#### Edit
Ask what to change:
- "What would you like to change?"
- Options: add element, remove element, adjust bbox, change motion style
- Update element list
- Show updated list for confirmation

#### Skip
- Leave elements array empty for this scene
- Add note: `"_skipped": true`
- Proceed to next scene

#### View Frame
```bash
open {frame_path}
```
Then return to options.

---

## Step 4: Summary and Write Manifest

After all scenes reviewed:

```
## Analysis Complete

| Scene | Elements | Status |
|-------|----------|--------|
{for each scene: | {N} | {count} elements | approved/skipped |}

**Total:** {total_elements} elements across {scene_count} scenes

Ready to write manifest?
1. Yes - write manifest.json
2. Review again - go back to a specific scene
3. Cancel - discard analysis
```

### Write Manifest

Update the manifest.json with all identified elements:

```python
# Manifest structure per scene
{
  "scene_id": N,
  "start_time": X.X,
  "end_time": Y.Y,
  "duration": Z.Z,
  "frame_path": "/path/to/frame.png",
  "elements": [
    {
      "element_id": "char_001",
      "type": "character",
      "description": "Illustrated person in meditation pose",
      "bbox": [100, 200, 300, 400],
      "animate": true,
      "motion_prompt": "gentle breathing, slight weight shift",
      "motion_bucket_id": 90
    }
  ]
}
```

---

## Step 5: Next Steps

```
## Manifest Written

Manifest: {OUTPUT_DIR}/manifest.json
Frames: {OUTPUT_DIR}/frames/

**Next steps:**

1. Review manifest (optional):
   cat {manifest_path} | jq '.scenes[] | {scene_id, elements: .elements | length}'

2. Fine-tune bboxes (optional):
   Open frames in an image editor to get precise coordinates

3. Run animation pipeline:
   python tools/animate.py --input {video_path} --manifest {manifest_path} --runpod --output-dir {OUTPUT_DIR}

Note: Animation requires RunPod endpoint. Run `python tools/animate.py --setup` if not configured.
```

---

## Analysis Guidelines

When analyzing frames, look for:

### Good Animation Candidates
- **Characters:** People, avatars, cartoon figures - breathing, weight shift, subtle movement
- **Animals:** Birds, pets, creatures - breathing, small movements
- **Water:** Rivers, oceans, puddles - rippling, flowing
- **Fire/Flames:** Candles, torches, fires - flickering, glow variation
- **Foliage:** Trees, plants, grass - wind sway, gentle movement
- **Clouds/Smoke:** Atmospheric elements - slow drift, morphing
- **Floating elements:** Particles, bubbles, leaves - gentle drift

### Skip These
- Text and titles
- Logos and branding
- UI elements (buttons, icons)
- Charts, graphs, diagrams
- Static geometric backgrounds
- Photos where motion would look unnatural

### Bbox Estimation
For a 1920x1080 frame:
- Estimate position visually (left third = x ~0-640, center = ~640-1280, right = ~1280-1920)
- Width/height based on element size relative to frame
- Include some padding around the element
- SAM2 will refine the mask, so rough bbox is fine

### Motion Intensity Guide
- **Subtle (90):** Breathing, micro-movements - for calm/meditative content
- **Medium (127):** Natural movement - default for most content
- **Active (180):** More dynamic movement - for energetic content

---

## Key Principles

1. **User confirms everything** - never write manifest without approval
2. **Show the frame** - user sees exactly what's being analyzed
3. **One scene at a time** - focused attention
4. **Skip is valid** - not every scene needs animation
5. **Rough bboxes OK** - SAM2 will refine with masks
6. **Err toward fewer elements** - quality over quantity
