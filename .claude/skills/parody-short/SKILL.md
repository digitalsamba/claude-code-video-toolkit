---
name: parody-short
description: Comedy shorts about a group's own life - a team's launch, quarter end, office move or leaving do; a club's season; a class's year; a family's Christmas - or about a shared topic everyone recognises (video calls, the return to the office), told in a borrowed genre (80s action intro, creature feature, Dickens Christmas ad, heist, nature documentary) and starring the group's real, animated photos or an invented cast. Covers mining the group's own context for in-jokes (chat, email, calendars, shared docs, plus domain tools like a CRM, helpdesk or git), mapping a genre's beats onto the real event, costuming and animating photos (image_edit then LTX), and per-genre look and sound. Use when someone asks for a fun, parody, spoof, holiday or "for the giggles" video about their team, colleagues, club, class or family, a party or all-hands video, or wants people's photos animated into a story.
---

# Parody shorts

A short comedy for an audience that knows the people in it, or the situation: a team's big night told as an 80s
action-show intro, a Halloween creature feature, a Christmas TV ad. The group can be any group (a sales team, a
school staff room, a five-a-side club, a family) and the subject anything they lived through together. They land harder than anything else the toolkit makes, because the
audience is laughing at things they recognise. This skill is how to make one on purpose.

Built and learned on three videos for one team in November 2026 (an action-show intro of a release, its Halloween
sister cut, a Christmas ad of *A Christmas Carol*). They are not in `examples/` because they star real colleagues.
Then on two public shorts with an invented cast (a horror trailer about the video call that never ends, a wildlife
documentary about the return to the office), each in 16:9 and vertical, plus a brand edition in a product's own UI.

## Why they work (in order of weight)

1. **Real faces, moving.** The step change was animating the team's profile photos. Stills got smiles; the same
   people raising aviators or toasting with punch got laughs. Recognition is the engine. See `people.md`.
2. **Real in-jokes.** Lines found in the team's own channels beat any written gag: a colleague's catchphrase, the
   feature someone keeps cramming in, the thing someone has been saying all along. The context pass is worth more
   than script polish. See `context.md`.
3. **A genre whose beats fit the event.** The audience already knows the shape (three rules that get broken;
   ghosts of past, present and future; the team assembling), so the video only fills the beats. The tighter the
   mapping, the less the script has to explain. See `genres.md`.
4. **Permission to exaggerate.** These are not release notes. Once the person asking says facts don't matter,
   numbers can be inflated and timelines invented. Ask once, early; then stop fact-checking jokes.
5. **Affection is the line.** Everyone is the hero of a gag or in on it, never its victim. That keeps it warm, and
   it is also the safety rule (below).
6. **A look and sound per genre.** Its own fonts, overlay and grade (VHS tracking, film grain and letterbox,
   candlelight), an original song or score, a few sound effects, at most one narrator line in an ad. Nothing
   should look like a toolkit template.
7. **Short.** An ad is 60-90 s. A "show intro" can run 2 min. Past ~2:30 the middle sags; cut there first.
8. **Pick every clip.** About one LTX clip in three cuts to a stranger. One wrong face breaks the spell, so
   contact-sheet every clip and re-roll.

## Two kinds of cast

- **Real people** (the group's own photos): the strongest laughs, for an audience that knows them. Internal by
  default; public only with each person's written OK.
- **An invented cast** (FLUX portraits, or non-human characters): for a public video, or a shared topic where the
  audience recognises the situation rather than the people (video calls, the office kitchen). Recognition moves
  from the faces to the situation, so the situation has to be universal and specific: "you're on mute", the
  spreadsheet with 412 tabs, the jacket on the hot desk. See `people.md` step 0.
- **Cast once, use everywhere.** Generate the invented cast once in a shared folder and give each video its own
  costumes and clips from the same faces. Two videos with the same four people feel like a series, and a
  callback between them (video 2's last line is video 1's first) is a free laugh.

## Rules that are not optional

- **All affectionate.** Nobody in the group is the butt of a joke they'd mind. No performance, health,
  personal life or anything said in confidence. When in doubt, cut the line.
- **Real people stay internal.** Their likeness belongs to them. Make it for the group, share it link-only, and ask
  before posting anywhere public. Anyone in it who asks to be left out is left out.
- **Nothing sensitive on screen.** No customer or partner names, no hostnames, no secret values (a secret is a
  blurred bar or `••••`), no internal URLs. Rename real systems ("RELAY ONE").
- **Original words and music.** Borrow a genre's *form*, never a work's characters, names, lines, designs or theme
  tune. A public-domain story (Dickens, Grimm, Shakespeare) can be used directly; a film or TV show cannot, renamed
  or not. Write original lyrics for any song (`acestep` skill).
- **Quote only where the audience could have read it.** Public team channels, yes. DMs, no. See `context.md`.

## Before you start

- **GPU endpoints** (Modal, once): `uv sync --extra modal && uv run modal setup`, then
  `uv run tools/modal_setup.py flux2 image_edit ltx2 qwen3_tts`. LTX is gated: it needs a Modal secret with your
  Hugging Face token (`docs/modal-setup.md`). Music: a free `ACEMUSIC_API_KEY` in `.env` (`acestep` skill).
- **The project** is a plain Remotion project in `projects/<name>/`; no template fits, since nothing here should
  look like one. Copy `package.json` and `remotion.config.ts` from any template (the config makes `lib/` imports
  resolve), keep one composition per cut in `src/Root.tsx`, and put the cast's clips in `public/people/`. A shot
  is a `<Sequence>` on absolute frames with the person in it:

```tsx
import { LivingPhoto } from '../../../lib/components/LivingPhoto';
// plays the clip once and holds its last good frame; fit slows a slightly short clip to fill the shot
<Sequence from={120} durationInFrames={180}>
  <LivingPhoto src="people/sam-xmas.mp4" hold={140} fit={180} pos="center 30%" style={{ inset: 0 }} />
</Sequence>
```

## Workflow

| Step | What | Read |
|---|---|---|
| 1. Brief | Event, audience, cast (3-6 people), genre, length, "do facts matter?" | this file |
| 2. Context | Mine the group's channels and tools; write `public/bible.md` | `context.md` |
| 3. Map | Genre beats onto the event; one beat per person; pick lines from the bible | `genres.md` |
| 4. Cast | Photos or an invented cast, costume with `image_edit`, animate with LTX, contact sheet | `people.md` |
| 5. World | FLUX backgrounds (text-free), SFX, music or a song with original lyrics | `acestep`, `elevenlabs` |
| 6. Build | Remotion shots on absolute frames; `LivingPhoto` for the cast; genre overlay | `people.md`, `video-design` |
| 7. Check | Contact sheet of the render, `qc.py --render --fix`, `sync_check.py --cues`, then the safety rules again | `video-design/critique.md`, CLAUDE.md |
| 8. Cuts | Vertical, and a brand edition if wanted: new layouts over the same assets | `formats.md` |

**Shots, not scenes.** These are edited like ads: a fixed shot list with absolute frame ranges, a 10-20 frame
dissolve between shots, the song or score under everything. If there is narration, generate it first and anchor
to it (CLAUDE.md, Audio-Anchored Timelines); an ad with one closing narrator line can be music-led.

**Sound on a movement: measure, don't eyeball.** A sneeze, a jacket landing, a badge jingle: place each sound
with `tools/sync_check.py --clip` (where its peak meets the movement), list it in the project's `sync-cues.json`,
and run `sync_check.py --video ... --cues` on every render before showing it. A hand-placed sneeze was half a
second early; three other cues were 0.8-1.0 s off. If a clip freezes or cuts, end its sound there too.

**Text is the punchline carrier.** Speech bubbles, a stamp, a dance card, a tombstone, a typed label. Remotion
renders all of it (FLUX and LTX garble text). Give every on-screen gag a frame cue and a sound.

**Voices.** For a narrator nobody owns, design one with Qwen VoiceDesign from a written brief ("a warm,
middle-aged British woman narrating a wildlife documentary, hushed, a slow measured pace"; "a deep, gravelly movie
trailer announcer"): `qwen3_tts.py --design-instruct "..." --text "<a 20 s seed paragraph in the genre, not the
script>" --format wav`. VoiceDesign makes a new voice on every call, so design once, then clone that reference
for every sentence (three takes, `pick_takes.py`). Describe a genre, never a real person; picking a narrator the
genre's famous voice isn't (a woman for a wildlife film) keeps it clearly original.

**Cuts.** Plan the vertical cut and any brand edition from the start; they reuse every generated asset and cost
only a render each. See `formats.md`.

**Reuse across genres.** A cast animated once can be re-costumed for the next holiday: the photos, crops and
context bible carry over, and only the costumes, clips and world change.

## Cost (Modal, 2026-11)

LTX ~$0.08 a 5 s clip, image_edit ~$0.01 an edit warm (a cold start is ~10 min; send one edit first and wait),
FLUX ~$0.01 a background or portrait, ACE-Step song a few cents. A 90 s ad with five cast members and re-rolls is
about $2. Two 80-100 s shorts sharing one invented cast (9 costume edits, 16 clips, 2 designed voices) came to about
$2.50 together. The acemusic API can return 502s mid-batch; a failed take is skipped, so re-run for more.

## Related

`context.md`, `genres.md`, `people.md`, `formats.md` here; `ltx2`, `qwen-edit`, `acestep`, `elevenlabs`, `video-design`,
`remotion` skills; `lib/components/LivingPhoto.tsx`.
