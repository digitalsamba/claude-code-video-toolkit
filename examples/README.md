# Example Projects

Curated showcase projects demonstrating toolkit capabilities.

## Available Examples

| Example | Template | Contributor | Description |
|---------|----------|-------------|-------------|
| digital-samba-skill-demo | product-demo | [Digital Samba](https://digitalsamba.com) | Marketing video for Claude Code skill |
| sprint-review-cho-oyu | sprint-review | [Digital Samba](https://digitalsamba.com) | iOS sprint review for Digital Samba Mobile |

> **Note:** Examples include configs and documentation, but NOT large media files. See each example's `ASSETS-NEEDED.md` for what to create.

## Contributors

Thank you to these organizations and individuals for sharing their video projects:

| Contributor | Website | Examples Shared |
|-------------|---------|-----------------|
| Digital Samba | [digitalsamba.com](https://digitalsamba.com) | digital-samba-skill-demo, sprint-review-cho-oyu |

*Want your project featured? Run `/contribute` and select "Share an example project".*

## Using Examples

```bash
# Copy an example to your projects directory
cp -r examples/example-name projects/my-project
cd projects/my-project

# Install and preview
npm install
npm run studio
```

## Adding Demo Assets

Examples don't include large media files (videos, audio). To run them:

1. **Record demos** - Use `/record-demo` to capture screen recordings
2. **Generate voiceover** - Use `/generate-voiceover` with the included script
3. **Add music** - Use `python tools/music.py` for background tracks

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
