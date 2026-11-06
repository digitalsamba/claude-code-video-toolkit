# Kits

A kit is everything one person or product needs for one-prompt videos, set up once in a working session and
used by every job after that. With a kit, someone can ask their agent "a post about the new autumn jars for
Friday" in a chat, and the agent writes a short spec that names the kit and a photo or two. No code gets written
per video.

```
kits/<name>/
├── kit.json      # what the job fills in when the spec doesn't say
├── brand.json    # optional: {"colors": {"primary", "accent", "bgDark", ...}} for this kit only
└── assets/       # their own photos and clips, used by "file" scenes
```

`kit.json` (every field optional):

| Field | Meaning |
| --- | --- |
| `description` | One line an agent can read back to the person ("small candle shop, warm and plain") |
| `brand` | A brand in `brands/` (Ideogram cards and voice use it). For a kit-only brand, put `brand.json` beside `kit.json` instead |
| `voice` | A voice preset name or a voice object, as in a spec |
| `format` | `9:16` or `16:9` |
| `backgrounds` | `auto` (default), `flux` or `local` (see `docs/agent-video-jobs.md`) |
| `music` | `false`, a prompt, or a music object |
| `assets` | The assets folder, relative to the kit (default `assets`) |
| `focus` | `{"candle.mp4": 0.85}`: where each asset's subject sits along the side a crop cuts, 0 (left or top) to 1 (right or bottom), default 0.5. A landscape clip in a 9:16 video keeps only its middle third, so set this for any subject off centre. A scene's own `visual.focus` overrides it |
| `design` | The look decided in the design session (`video-design` with taste or ours), so every one-prompt video keeps it. `fonts`: `{"display": {"path": "fonts/Display.ttf", "weight": "Medium"}, "body": ..., "caption": ...}`, files inside the kit so every host renders the same. `headline`: `font`, `color`, `scale`, `scrim`, `shadow`, `accentBar`, `sublineFont`, `case` (`as-written`, `sentence` or `upper`, applied whatever the agent typed). `captions`: `font`, `case` (`upper` or `as-written`), `pill`. `closing`: a sign-off card added to the end of every video, `{"seconds", "background", "color", "image", "lines": [{"text", "font", "size": "display" \| "title" \| "label" \| "note", "color", "gapBefore"}]}`. `fields`: the flat grounds of the design (`{"burgundy": {"ground": "#56140F", "ink": "#FFFCF1"}, ...}`), which turn on two more layouts (below); `layouts` limits which a kit allows. Every key is optional |
| `describe` | `{"candle.mp4": "a lit candle in a dark room, close"}`: what each asset shows. `catalog` lists every asset with this, so an agent that can't see the files can still choose one for each scene. Describe every file |

The kit's colours (its `brand.json`, else the named brand's) set the accent and the ground of the video:
headline accent bars, and the local backgrounds that `text` scenes and scenes with no visual get.

A kit's `voice` can be a clone: `{"provider": "qwen3", "refAudio": "voice/ref.wav", "refText": "exact transcript",
"maxWpm": 150, "minTempo": 0.8}`, with the reference inside the kit. A clone copies its reference's pace, so slow
a presenter-paced recording with `voice_prosody.py expand --tempo 0.8` first (CLAUDE.md, "Clone Prosody").

## Layouts: the same assets, set differently

A kit with a few assets makes the same-looking video every time if every scene is a clip full frame with a
headline over it. With `design.fields`, a scene can be set three ways (`"layout"` on the scene or its visual):

| Layout | What it shows |
| --- | --- |
| `full` | The photo or clip fills the frame, the headline over it (the default for `file` scenes) |
| `split` | The photo or clip on a stage (top 55% in 9:16, left half in 16:9), a flat field below it with the headline and captions on one left edge: the split-stage direction from taste |
| `type` | Words alone on a field, the headline large (the default for `text` scenes when the kit has fields) |

`"field": "<name>"` picks the ground (default: the first). Change it where the story turns, not every scene
(video-design: colour follows the story). In the one-line form: `split: candle.mp4 ; Let it settle @ cream`,
`text: Five minutes ; a warm drink @ gold`.

`catalog` also gives each asset a `usedRecently` count (how many of the last three videos made with this kit
showed it), so an agent can choose the others, and a kit whose headline `case` is `sentence` refuses a Title Case
title or headline before spend.

## Where kits live

The job looks in the agent workspace's `kits/` first (`$VIDEO_JOB_WORKSPACE/kits/<name>`), then in the toolkit's
own `kits/`. A runtime serving several people keeps each person's kit in its workspace; the toolkit doesn't know
about users. Everything in this folder except this README and `example/` is gitignored, because kits hold people's
photos and voices.

## Making one

In Claude Code, in the toolkit: make the brand (`/brand`) and, if they want their own voice, clone it
(`/voice-clone`); copy their photos and clips into `assets/` (portrait for 9:16, landscape for 16:9: a still
fills the frame and is cropped to it); write `kit.json`. Then try it:

```bash
uv run tools/video_job.py estimate --json --spec-json '{"title": "Test", "kit": "<name>", "scenes": [
  {"text": "A sentence about the first photo.", "visual": {"kind": "file", "path": "photo1.jpg", "headline": "HEADLINE"}},
  {"text": "A sentence about why it matters.", "visual": {"kind": "text", "headline": "TWO WORDS"}}]}'
```

`host_missing` in the answer lists anything this host can't do for the kit. An agent's one-line version:

```bash
uv run tools/video_job.py submit --kit <name> --title "Samhain" --output files/samhain.mp4 \
  --scene "The year turns dark now, and that is a kind of rest. :: file: standing-stones.mp4 ; Samhain" \
  --scene "Tonight, light one candle and let the day go. :: split: candle.mp4 ; Let it go @ cream"
```
