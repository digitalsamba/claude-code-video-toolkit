# Example Projects

Curated showcase projects demonstrating toolkit capabilities.

## Available Examples

| Example | Stack | Contributor | Description | Complexity |
|---------|-------|-------------|-------------|------------|
| hello-world | Remotion sprint-review | — | Minimal 25s video — zero config, renders in 2 minutes | Beginner |
| quick-spot | moviepy + PIL | — | 15s ad-style spot with audio-anchored timeline. Runs with zero external assets. | Beginner |
| data-viz-chart | moviepy + matplotlib | — | Animated time-series chart with deterministic title and source attribution. Runs with included data file. | Beginner |
| data-chart-stars | Remotion (custom) + `DataChart` | [Digital Samba](https://digitalsamba.com) | 31s narrated chart video: star history drawn word by word with Lugh's voice, then monthly bars lit on cue. All assets committed. | Intermediate |
| blender-disk-panic | Blender shot spec | | 22.5s rigged character short from one JSON spec: 8 shots, 37 acting beats, cameras checked by geometry. Needs Blender; renders in about 15 minutes. | Intermediate |
| blender-product-shot | Blender shot spec (studio set) | | 12s product stage with no character: a picture on a floating card and on a phone, and an imported 3D model, in `reveal`, `orbit` and `product` shots framed from each prop's size. Needs Blender; all assets committed. | Beginner |
| blender-in-remotion | Remotion (custom) + Blender shot specs + `BlenderClip` | | 23.5s video with Blender elements: a `Terminal` scene cuts to itself on a 3D laptop and the camera pulls back, then a transparent character acts beside the beats it is acting. Needs Blender; specs and sound committed, clips built with `npm run blender`. | Advanced |
| ds-crt-stinger | LTX-2 + moviepy + PIL | — | 6s brand stinger — LTX-2 CRT LoRA footage + post-processed grunged logo | Intermediate |
| sky-blue-short | concept-explainer-short (moviepy) | — | 52s vertical 9:16 explainer short — Qwen3 VO + Ideogram cards + LTX b-roll + burned captions. All assets committed; re-renders with `uv run build.py`. | Intermediate |
| ds-release-649 | Remotion (custom) | [Digital Samba](https://digitalsamba.com) | 80s synthetic release-notes video. Product UIs rebuilt from source, scripted cursor and camera, Lugh voice. All assets committed. | Advanced |
| ds-free-schedule-demo | Remotion (custom) | [Digital Samba](https://digitalsamba.com) | 26s synthetic product demo with a SoulX narrator tile. All assets committed. | Intermediate |
| digital-samba-skill-demo | Remotion product-demo | [Digital Samba](https://digitalsamba.com) | Marketing video for Claude Code skill | Intermediate |
| sprint-review-cho-oyu | Remotion sprint-review | [Digital Samba](https://digitalsamba.com) | iOS sprint review for Digital Samba Mobile | Intermediate |

> **Note:** Remotion examples include configs and documentation but NOT large media files — see each example's `ASSETS-NEEDED.md` for what to create. The moviepy examples (`quick-spot`, `data-viz-chart`) are fully self-contained and run end-to-end with `uv run build.py`.

## Contributors

Thank you to these organizations and individuals for sharing their video projects:

| Contributor | Website | Examples Shared |
|-------------|---------|-----------------|
| Digital Samba | [digitalsamba.com](https://digitalsamba.com) | digital-samba-skill-demo, sprint-review-cho-oyu, ds-release-649, ds-free-schedule-demo, data-chart-stars |

*Want your project featured? Run `/contribute` and select "Share an example project".*

## Using Examples

**Remotion examples** (`hello-world`, `digital-samba-skill-demo`, `sprint-review-cho-oyu`, `ds-release-649`, `ds-free-schedule-demo`, `data-chart-stars`):

```bash
cp -r examples/example-name projects/my-project
cd projects/my-project
npm install
npm run studio
```

**moviepy examples** (`quick-spot`, `data-viz-chart`) — run in place, no copy needed:

```bash
cd examples/quick-spot   # or examples/data-viz-chart
uv run build.py         # produces out.mp4 in the example directory
```

These are fully self-contained references for the moviepy skill. Read the `build.py` and `README.md` in each.

## Adding Demo Assets

Examples don't include large media files (videos, audio). To run them:

1. **Record demos** - Use `/record-demo` to capture screen recordings
2. **Generate voiceover** - Use `/generate-voiceover` with the included script
3. **Add music** - Use `uv run tools/music.py` for background tracks

Each example includes a `ASSETS-NEEDED.md` documenting what to create.

## Contributing Examples

To share a project as an example:

1. Run `/contribute` and select "Share a template" (examples work similarly)
2. Or manually:

```bash
# Copy project to examples (without large media)
cp -r projects/my-project examples/

# Remove media files (these are gitignored anyway)
rm examples/my-project/public/demos/*.mp4
rm examples/my-project/public/audio/*.mp3

# Create asset documentation
# (describe what demos/audio are needed)

# Commit and PR
git add examples/my-project
git commit -m "Add example: my-project"
gh pr create
```

## Example Structure

```
examples/example-name/
├── README.md              # What this example demonstrates
├── ASSETS-NEEDED.md       # What media to create
├── src/
│   └── config/
│       └── *-config.ts    # Example configuration
├── public/
│   ├── demos/             # Empty (add your recordings)
│   ├── audio/             # Empty (add your voiceover)
│   └── images/            # Logos, screenshots (tracked)
└── VOICEOVER-SCRIPT.md    # Script for narration
```
