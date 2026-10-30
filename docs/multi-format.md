# Compose one story for different screens

The `storyboard` template reads project scenes as data and exposes three
compositions: Storyboard at 1920×1080, StoryboardPortrait at 1080×1920 and
StoryboardSquare at 1080×1080. Text and pictures are rearranged, rather than
cropping a finished landscape video. Legacy templates continue to work as
before; they can adopt the shared `lib/formats.ts` layout helper individually.

Copy `templates/storyboard` into a project directory and run `npm ci`. Edit
project.json, then run the template's render commands. A generated props file
can replace project.json at render time. See the template README for fields.

Scenes support a title, onScreenText, image or video, scene narration audio,
word captions and narrator video. Caption size/margins and the narrator box
adapt with the canvas. Missing optional content uses the included visual card;
declared media files must exist. Keep text concise and check each format.
Titles are limited to 120 characters and onScreenText to 320 per scene. Split
longer material into scenes; the adapter reports a useful error before export.

Set the safeZones input prop to true in Studio to show dashed guides. Guides
reserve conservative room for social controls. These guides are configurable
design margins, not a live specification for every publishing platform.

Validation:

```bash
templates/storyboard/node_modules/.bin/tsc lib/formats.ts --target ES2020 --module commonjs --strict --outDir scratch/formats-ts
node --test tests/formats.test.cjs
cd templates/storyboard
npm run typecheck
npm run render:portrait
```
