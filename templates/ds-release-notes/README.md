# ds-release-notes: synthetic release-notes video template

A Remotion template for Digital Samba release-notes videos in which **nothing is screen-recorded**. The product's
screens are rebuilt from its source as static React, and a scripted cursor and camera perform each release item.
Every click is a frame number, so a scene re-renders identically and is updated by editing code.

It ships as the finished build of [release 650](https://www.digitalsamba.com/release-notes/release-650) (87s, six
items), so `npm run render` works before you change anything. For a new release you keep the kits, the stage, the
cards and the audio pipeline, and replace the scenes and the voice lines.

The method is the `synthetic-ui-demo` skill; this template is its release-notes shape.

## What's in it

| Path | What | Keep for the next release? |
|---|---|---|
| `src/Release.tsx` | Scene order, lengths, VO placement, SFX. Its header lists every phrase's length and start | Edit |
| `src/config.ts` | `RELEASE_NUMBER` and `RELEASE_MONTH`, shown on the title and outro cards | Edit first |
| `src/scenes/Cards.tsx` | Title card (item list), outro, scene label | Edit the item list |
| `src/scenes/*.tsx` | One file per release item: UI state, cursor path, clicks and camera as frame constants | Replace |
| `src/Stage.tsx`, `src/Cursor.tsx` | Browser window, camera keyframes, cursor (the project-local version of `lib/components` Stage) | Keep |
| `src/room/` | In-call room kit: chrome, left toolbar, tiles, participants panel, menus, dialogs, toasts, icomoon glyphs | Keep, extend |
| `src/ac/` | DS Free account-center kit: shell, nav, buttons, fields, switches, chips | Keep, extend |
| `public/people/`, `public/*.svg`, `public/banner.jpg` | FLUX.2 portraits (labelled AI-generated on screen), logos | Keep |
| `public/audio/scenes/` | Lugh's phrases: `.mp3`, `.txt`, and `.captions.json` word timings | Replace |
| `public/audio/sfx/` | click, typing, pop, chime, whoosh, reveal | Keep |

## Making the next release

Copy the template to `projects/release-<number>` (or let `/video` do it), set `RELEASE_NUMBER` and
`RELEASE_MONTH` in `src/config.ts` and the `ITEMS` list in `src/scenes/Cards.tsx`, then:

0. **Start from Lugh's brief if there is one.** Lugh (longarm) traces each item to code and writes a brief:
   the click path, the exact UI strings, states, the commit and Jira key, and the narration split into
   phrases. Read it with `ssh longarm cat /opt/longarm/lugh/memory/release-videos/release-<N>.md`. It covers
   steps 1 and 2; still check its quoted strings against `origin/master`, and settle its "Open questions"
   before building.
1. **Read the release notes and trace each item to code.** Send one subagent per app (`react-frontend`,
   `react-dashboard`, `react-account-center`), reading `origin/master`, to return the exact strings, control order,
   defaults and the commit for each item. Local checkouts are often stale. Never paraphrase UI text.
2. **Write one short line per item, split at sentences.** One `.txt` per phrase in `public/audio/phrases/`
   (`01a_...`, `01b_...`). Short phrases come out cleaner than paragraphs and give you the gaps.
3. **Generate three takes and pick.** Clone from the expressive reference, not the raw one (see Voice below):
   ```bash
   for t in 1 2 3; do cp -r public/audio/phrases public/audio/takes/t$t; \
     uv run tools/voiceover.py --provider qwen3 --cloud modal --language English \
       --ref-audio assets/voices/lugh-ref-expressive.wav --ref-text "<its transcript>" \
       --temperature 0.5 --scene-dir public/audio/takes/t$t; done
   uv run tools/pick_takes.py --takes-dir public/audio/takes --output public/audio/scenes --json
   uv run tools/captions.py public/audio/scenes          # word timings for click placement
   ```
4. **Build the scenes.** Copy the closest existing scene file. Lay the screen out at fixed positions on the 1440x810
   canvas, export the rects that matter, and put every cursor keyframe, click, camera move and UI state switch in
   the constants at the top. Put each click on the word that names it with `cue()` (`src/cues.ts`), and anchor the
   cursor and camera keyframes to those cues, so a re-timed phrase moves the whole scene with it;
   `src/scenes/FreeCustomLink.tsx` shows the pattern. The other 650 scenes still carry hand-set frames.
5. **Size each scene** to its phrases plus a beat and update the header comment in `Release.tsx`.
6. **Check stills before rendering.** Render a still at every click frame and tile them; a guessed coordinate is
   usually 20-50px off once the camera zooms.
7. **Render and normalise.**
   ```bash
   npm run render
   cd ../.. && uv run tools/qc.py --render projects/<name>/out/release-notes.mp4 --fix
   ```

## Voice

Lugh is a Qwen3-TTS clone. The raw reference (`assets/voices/lugh-ref.wav`) is a flat, fast read, and a clone copies
both, so `assets/voices/lugh-ref-expressive.wav` is the same clip with its pitch contour expanded 2.0x and slowed
15% (`tools/voice_prosody.py expand`). Clones from it land at 145-165 wpm with normal narration intonation and need
no stretching. Check any voice with `tools/voice_prosody.py measure`: under 2 st sd is flat, 3-4 st is narration.

The reference transcript: "Good morning. I've had a look through last night's tickets. There's one that needs your
attention from the key meeting team. Nothing urgent, but I'd like to flag it before we move on to the support queue".

## Run it

```bash
cp -r templates/ds-release-notes projects/release-651
cd projects/release-651   # then set RELEASE_NUMBER and RELEASE_MONTH in src/config.ts
npm install
npm run studio
npm run render
```

## 3D device shots (planned, not built)

A design note for using Blender shot specs here (`.claude/skills/blender/`). Nothing in the template needs Blender
today.

- **What for:** at most one device reveal per release, on an item that is about a device. The flat scene cuts to
  itself on a phone or tablet on a studio stage and the camera pulls back, which shows "this is on a tablet" in a
  way the browser window can't.
- **How:** render the scene with Remotion at 30 fps, name it as a studio prop's `screen.video` in a spec with
  `"view": "Standard"` (so the product's colours survive the cut) and the backdrop at the field colour `#eef2f8`,
  film it with a `reveal` shot, `render --clips`, and hard-cut to the clip on the frame `skip_frames` names. The
  clip starts on a `cue()` frame, as the scenes' clicks do.
- **Optional by construction:** the scene plays its flat version when `public/blender/clips.json` isn't there, so
  the template renders without Blender. The clip is its own `npm run blender` step: 2 to 3 minutes for a
  4-second shot on an M4.
- **Waiting on:** a tablet prop and an exposure range for light backdrops (both on the roadmap). Pilot it in a
  project copy first, on release 650's "Virtual backgrounds on Android tablets" scene.
- **Not planned:** a character for the narrator, or 3D on the title and outro cards. The first is the brand's
  decision; the second is decoration, which the design rules leave out.
- **Limits to know:** `qc.py --layout` can't measure UI text inside a clip, so check the reveal's last frame by eye.

## What's real and what isn't

| Real (from the product code) | Invented |
|---|---|
| Layout, colours, icons, toolbar and tab order, control order | People: FLUX.2 portraits, labelled "AI-generated" on screen |
| Every UI string: labels, help text, notices, warnings | Names, emails, call details, shared content |
| Defaults and behaviour | Phone and tablet chrome (status bar, address bar) |
