# things-worth-doing-short — guidance for Claude

Forked 2026-09-01 from `templates/concept-explainer-short` for the "Things Worth
Doing" men's content brand ([[Men's UGC Project]] in the Obsidian vault, and
[[Playbook - AI Agent Content Sprint (Things Worth Doing)]] for how a batch of
scripts/shot lists gets produced upstream of this template). Same Python/moviepy
pipeline, no Remotion, no npm — `gen_vo.py → gen_captions.py → build.py`, run
**from the project directory**. Differences from the base template are below;
everything not mentioned here works exactly like `concept-explainer-short`.

## What's different from concept-explainer-short

1. **One scene = one shot, not one narration beat.** The base template's own
   guidance ("~15s pattern interrupt") assumes few, longer scenes. Things Worth
   Doing's Production Standard requires visual movement every few seconds, so
   TWD scenes run ~2-4s each — map each shot in the sprint playbook's Shot List
   directly to one `scenes.json` scene, in order, one VO line per shot. A
   28-second video is normal at 7-9 scenes here, not 4.
2. **`visual` field gets a third real value: `"real"`.** `build.py` actually
   decides treatment by file **extension**, not by the `visual` string — it's
   documentation only. So a real filmed `.mp4` clip gets the exact same
   boomerang-loop treatment an LTX clip would. Tag it `"real"` in scenes.json
   purely so a human scanning the file can tell what still needs to be shot
   vs. generated — `build.py` doesn't care.
3. **Voice is ElevenLabs by default**, via `config.json → voice.brand:
   "things-worth-doing"` (patched into this fork's `gen_vo.py` — the base
   template didn't forward `--brand` to `tools/voiceover.py` at all, only
   scene text and qwen3-specific fields). Real voiceId is still a placeholder
   in `brands/things-worth-doing/voice.json` — see Gotchas.
4. **Palette is the proposed "field guide + documentary" default** in
   `config.json`/`brands/things-worth-doing/brand.json` — warm rust/olive
   accents, warm off-black/off-white, deliberately not neon or AI-futuristic
   (matches the brand's Visual Identity). Not yet confirmed by Adrian; safe to
   change, nothing else depends on the specific hex values.
5. **Real footage is expected, not optional.** TWD's playbook prioritizes real
   product/filmed footage over AI generation wherever the shot needs to show
   the actual product being used (see the playbook's visual hierarchy). Don't
   default to generating everything with LTX/Ideogram just because this
   template can — check each scene's per-shot production package first.

## Working on a project copy (same as base template)

1. Plan in `scenes.json` first — one scene per shot from the video's
   production package (VO Script + Shot List sections), not from scratch.
2. Hook discipline: scene 01 must earn the next few seconds — question or
   tension immediately, no throat-clearing.
3. `build.py` works at every stage — placeholders before assets, estimates
   before VO, silent before audio. Render early, render often; show the user
   intermediate renders rather than describing them.

## Review checklist before calling a video done

- Pull frames with ffmpeg at several timestamps and *look* at them: caption
  collisions, asset crops, placeholder cards left in.
- Check `gen_vo.py`'s per-scene wpm output; anything flagged FAST/SLOW that
  `maxWpm` didn't catch needs a script edit or retake.
- Run this video's package through the sprint playbook's QC gates and
  Red-Team pass again post-render, not just at pre-production — a render can
  surface problems (bad continuity, a shot that doesn't land) invisible in
  text form.

## Gotchas (this fork, so far)

- **`brands/things-worth-doing/voice.json`'s `voiceId` is still
  `"YOUR_VOICE_ID_HERE"`** — `tools/voiceover.py` will error rather than
  silently using a default. Replace it with a real ElevenLabs voice ID before
  running `gen_vo.py` for real.
- Inherited from the base template: clone pacing follows the reference
  recording, `voice.maxWpm` is a safety net not a fix; never burn whisper's
  own transcription (captions force-align to script text on purpose); moviepy
  2.x uses `with_*` methods and PIL for text.
- Keep toolkit-level fixes (improving this *template*) in
  `templates/things-worth-doing-short/`, not in a copied project under
  `projects/` — same separation the base template's CLAUDE.md establishes.
