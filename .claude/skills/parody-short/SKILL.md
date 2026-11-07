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
  audience recognises the situation rather than the people. Same pipeline from step 2 of `people.md`.

## Rules that are not optional

- **Colleagues only, all affectionate.** Nobody is the butt of a joke they'd mind. No performance, health,
  personal life or anything said in confidence. When in doubt, cut the line.
- **Internal use.** Real people's likeness belongs to them. Make it for the team, share it link-only, and ask
  before posting anywhere public. Anyone in it who asks to be left out is left out.
- **Nothing sensitive on screen.** No customer or partner names, no hostnames, no secret values (a secret is a
  blurred bar or `••••`), no internal URLs. Rename real systems ("RELAY ONE").
- **Original words and music.** Borrow a genre's *form*, never a work's characters, names, lines, designs or theme
  tune. A public-domain story (Dickens, Grimm, Shakespeare) can be used directly; a film or TV show cannot, renamed
  or not. Write original lyrics for any song (`acestep` skill).
- **Quote only where the audience could have read it.** Public team channels, yes. DMs, no. See `context.md`.

## Workflow

| Step | What | Read |
|---|---|---|
| 1. Brief | Event, audience, cast (3-6 people), genre, length, "do facts matter?" | this file |
| 2. Context | Mine the team's channels and tools; write `public/bible.md` | `context.md` |
| 3. Map | Genre beats onto the event; one beat per person; pick lines from the bible | `genres.md` |
| 4. Cast | Profile photos, costume with `image_edit`, animate with LTX, contact sheet | `people.md` |
| 5. World | FLUX backgrounds (text-free), SFX, music or a song with original lyrics | `acestep`, `elevenlabs` |
| 6. Build | Remotion shots on absolute frames; `LivingPhoto` for the cast; genre overlay | `people.md`, `video-design` |
| 7. Check | Contact sheet of the render, `qc.py --render --fix`, then the safety rules again | `video-design/critique.md` |

**Shots, not scenes.** These are edited like ads: a fixed shot list with absolute frame ranges, a 10-20 frame
dissolve between shots, the song or score under everything. If there is narration, generate it first and anchor
to it (CLAUDE.md, Audio-Anchored Timelines); an ad with one closing narrator line can be music-led.

**Text is the punchline carrier.** Speech bubbles, a stamp, a dance card, a tombstone, a typed label. Remotion
renders all of it (FLUX and LTX garble text). Give every on-screen gag a frame cue and a sound.

**Reuse across genres.** A cast animated once can be re-costumed for the next holiday: the photos, crops and
context bible carry over, and only the costumes, clips and world change.

## Cost (Modal, 2026-11)

LTX ~$0.08 a 5 s clip, image_edit ~$0.01 an edit warm (a cold start is ~10 min; send one edit first and wait),
FLUX ~$0.01 a background, ACE-Step song a few cents. A 90 s ad with five cast members and re-rolls is about $2.

## Related

`context.md`, `genres.md`, `people.md` here; `ltx2`, `qwen-edit`, `acestep`, `elevenlabs`, `video-design`,
`remotion` skills; `lib/components/LivingPhoto.tsx`.
