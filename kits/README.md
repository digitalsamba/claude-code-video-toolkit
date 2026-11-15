# Kits

A kit is everything one person or product needs for one-prompt videos, set up once in a working session and
used by every job after that. With a kit, someone can ask their agent "a post about the new autumn jars for
Friday" in a chat, and the agent writes a short spec that names the kit and a photo or two. No code gets written
per video.

```
kits/<name>/
├── kit.json      # what the job fills in when the spec doesn't say
├── brand.json    # optional: {"colors": {"primary", "accent", "bgDark", ...}} for this kit only
├── assets/       # their own photos and clips, used by "file" scenes
└── cast/         # optional: characters for Blender shot specs, one JSON file each
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
| `describe` | `{"candle.mp4": "a lit candle in a dark room, close"}`: what each asset shows. `catalog` lists every asset with this, so an agent that can't see the files can still choose one for each scene. Describe every file |

`catalog` also gives each asset a `usedRecently` count (how many of the last three videos made with this kit
showed it), so an agent can choose the others and two videos for one person don't look the same.

A kit's `voice` can be a clone: `{"provider": "qwen3", "refAudio": "voice/ref.wav", "refText": "exact transcript",
"maxWpm": 150, "minTempo": 0.8}`, with the reference inside the kit. A clone copies its reference's pace, so slow
a presenter-paced recording with `voice_prosody.py expand --tempo 0.8` first (CLAUDE.md, "Clone Prosody").

The kit's colours (its `brand.json`, else the named brand's) set the accent and the ground of the video:
headline accent bars, and the local backgrounds that `text` scenes and scenes with no visual get.

## A kit's cast

`cast/<id>.json` is a character for a Blender shot spec (`tools/blender_render.py`): a description of a person as
named choices and hex colours, as in `example/cast/maya.json`.

```json
{"id": "maya", "kind": "clay_person",
 "description": "Maya, the product lead: slim, brown skin, a black bob, round glasses, a green shirt",
 "body": {"build": "slim", "skin": "brown"},
 "hair": {"style": "bob", "color": "black"},
 "clothes": {"top": "shirt", "top_color": "#2E7D5B"},
 "accessories": ["glasses", "earrings"]}
```

A spec names the file and may change the costume for one video, which keeps the person the same across videos:

```json
"cast": [{"id": "maya", "from": "../../kits/example/cast/maya.json", "clothes": {"top_color": "#B23A48"}}]
```

The path is relative to the spec. `description` is for the agent to read back; the build ignores it. The fields
and their choices are in `.claude/skills/blender/SKILL.md`. The characters are cartoon caricatures, not
likenesses. `video_job.py` does not read `cast/` yet: it is for shot specs written in a session.

## Where kits live

The job looks in the agent workspace's `kits/` first (`$VIDEO_JOB_WORKSPACE/kits/<name>`), then in the toolkit's
own `kits/`. A runtime serving several people keeps each person's kit in its workspace; the toolkit doesn't know
about users. Everything in this folder except this README and `example/` is gitignored, because kits hold people's
photos and voices.

A `file` scene's path may only lead into the kit, the agent workspace, the directory `submit` runs from or the
toolkit. Anything else (an absolute path elsewhere, `../..`) is refused, so an agent taking requests from a channel
can't put an arbitrary file from the host into a video. Copy a file into the kit's `assets/` to use it.

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
  --scene "Tonight, light one candle and let the day go. :: file: candle.mp4"
```
