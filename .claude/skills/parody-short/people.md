# The cast: from profile photo to animated character

Profile photo → square-up crop → costume (`image_edit`) → animate (`ltx2.py --input`) → contact sheet → play
it in Remotion with `LivingPhoto`. Each step has one failure that cost a re-roll; they're listed with it.

## 0. An invented cast (no real people)

For a public video, generate the cast with FLUX at the size LTX animates, in webcam framing:

```bash
uv run tools/flux2.py --cloud modal --no-open --width 704 --height 768 --seed 7 --output cast/dana.png --prompt \
  "Webcam video call still of a cheerful woman in her mid forties with short grey-streaked dark hair and round \
tortoiseshell glasses, wearing a mustard cardigan, at a home desk with a bookshelf and a plant behind her, head and \
shoulders, centred, facing the camera, looking into the lens, soft natural window light, photorealistic photograph, \
no text, no logos"
```

- Make each person distinct at a glance: age, hair, one strong colour in their clothes, their own room (a guitar,
  a fridge covered in magnets, framed certificates). Four people is enough for a call grid.
- Tile them into one sheet and check them together before animating anything.
- Keep the cast in a shared folder (`projects/cast/`) and copy into each video; costumes and clips are per video.
- Skip step 1: the portraits are already 704x768.

## 1. Photos

Profile photos from the team's chat (`slack_read_user_profile` gives the image URL; take the largest size) or
photos the person asking supplies. Real people's photos are for internal use (SKILL.md, rules).

Crop every photo to the same **704x768 portrait**, the size LTX animates:

```bash
ffmpeg -y -i people/sam.jpg -vf "scale=704:768:force_original_aspect_ratio=increase,crop=704:768" people/sam-in.jpg
```

Plain `crop` fails on a square or small source; the `scale ... increase` first is what makes it work on any input.

## 2. Costume (optional)

```bash
uv run tools/image_edit.py --cloud modal --no-open --input people/sam-in.jpg --seed 7 --output people/sam-xmas.png \
  --prompt "Keep the same close-up head and shoulders framing, facing the camera. Dress him as a jolly Victorian \
Ghost of Christmas Present: a long green fur-trimmed robe and a holly wreath crown, warm firelight, a Victorian hall \
behind. Keep exactly the same person with an identical face and identical hair, do not add or remove glasses, \
beard or stubble, photorealistic photograph"
```

- **The keep-clause matters.** Listing features to keep ("keep his glasses and beard") *adds* them to people who
  don't have them. Use the wording above, which names no feature the person lacks. The reverse also happens: a
  person who *does* wear glasses can lose them in a new setting even with "do not add or remove glasses". For
  them, name the glasses ("she keeps wearing her round tortoiseshell glasses ... the same glasses"). Name what
  the person has; never name what they lack.
- Small props in hands (a mug, papers) come out ghosted or smeared. A flat, held prop (a clipboard against the
  chest) survives.
- Costume plus setting in one edit; keep framing "head and shoulders, facing the camera" so LTX gets a face.
- Modal cold-starts take ~10 min (the tool's 600 s timeout can fail the first call). Send one edit, wait for it,
  then run the rest warm (~10 s each).
- Re-crop the output to 704x768 (step 1) before animating.

## 3. Animate

```bash
uv run tools/ltx2.py --cloud modal --no-open --input people/sam-xmas-in.jpg --width 704 --height 768 \
  --num-frames 121 --seed 5 --output people/sam.mp4 \
  --prompt "Static camera, medium shot, framing stays exactly the same, no zoom, no cuts. \
The same man in the same green robe throws his head back and laughs heartily, then spreads his arms wide. \
The same background, the same person, the same face and clothes throughout, no text"
```

- **Feed the uncropped portrait.** A tight 16:9 crop of a face made LTX hard-cut to a different person in a
  different room. Frame the close-up in Remotion (`objectFit: cover`, `objectPosition`) instead.
- **Never** "opening credits", "camera pushes in", "title": they cause cuts, garbled text and push-ins.
- One clear, fun action per clip (lowers aviators, finger guns, a toast, tips a hat, a bouncer head-shake). A
  gentle action survives better than a big one.
- **Keep legs out of it.** A full-body shot where the legs move (feet up, crossing legs, a kick) grew a third leg.
  Hands and faces hold up; legs don't. Say "both feet flat on the floor, legs still" in a full-body prompt.
- 121 frames at 24 fps is ~5 s, about 145 frames at 30 fps in Remotion.
- Seed 5 was most reliable; seed 11 was the usual re-roll.

## 4. Check every clip

```bash
uv run tools/contact_sheet.py people/sam.mp4 --count 6 --cols 6    # 6 evenly spaced frames, one row
```

Then run it once without `--count`: its scene detection should find no cut in a 5 s clip. If it finds one, that's
the cut to a stranger.

Look for: a different face, a cut to another room, garbled text, a face that melts at the end. About one clip in
three needs something. Fix it in the cheapest way:

| Problem | Fix |
|---|---|
| Cuts to a stranger late in the clip | `hold` the frame before the cut (`LivingPhoto hold={105}`) |
| Smears or melts late in the clip | `hold` before it, the same way (it showed in 2 of 9 headset clips around 3.3 s) |
| The best moment is near the end (a sneeze, a punchline face) | freeze on it with `hold`, and time the shot's gag to the hold frame |
| Garbled text or a glitch in the first second | `trim` past it (`trim={40}`) |
| Stranger from the start, or the action is wrong | re-roll: another seed, a gentler action |
| An extra limb or a wrong action mid-clip | extend from a frame **before** it (`ltx_extend.py --at`) with a calmer prompt; the bad part never plays |
| The shot is longer than the clip | extend it (below); never loop or ping-pong it |

## 4b. Make it as long as the shot

An LTX clip is ~5 s and a shot is often 7-12 s. Two ways to fill the gap both failed: holding the last frame read
as a pause, and playing the tail back and forth read as a loop. Continue the clip instead:

```bash
uv run tools/ltx_extend.py --input people/sam.mp4 --at 4.8 --times 2 --output people/sam-long.mp4 \
  --prompt "A fixed view that never moves: the framing stays exactly the same, no zoom, no pull back, no cuts. A close medium shot of the same man, who keeps laughing ... no text"
```

Two things decide whether a continuation keeps the person (out of 21 extensions, 2026-11-07):

- **Never continue from a close-up.** A frame where the person has leaned right into the lens has no room left in
  it, and LTX invents a new scene: 2 of 2 cut to a stranger in a studio. Pick a frame from before the lean.
- **Don't say "tripod".** In a webcam scene "locked off tripod" got a tripod and a camera drawn into the frame, in
  a studio; "static camera" or "a fixed view" asks for the same thing without the object. For webcam shots write "a fixed webcam view that never moves: the framing stays exactly the same, no
  zoom, no pull back, no cuts". A continuation also tends to pull back to a wide shot; "a close medium shot of
  his upper body" holds it.

It takes the frame at `--at` (the last good one), runs image-to-video from it and joins the two with a short
cross-fade; `--times 2` chains another. The seam doesn't show, because the continuation starts from the clip's own
frame. Contact-sheet the result: a continuation can drift or cut at its own end, so set `hold` to the shot length
and stop before it. ~$0.08 an extension.

Measured on one shot (Marcus at home, 11 s): holding gave 63% frozen frames; a ping-pong tail gave 29% of frames
revisiting an earlier frame; an extended clip gave 0% frozen and 0% revisits.

## 5. Play it in Remotion

```tsx
import { LivingPhoto } from '../../../lib/components';

<Sequence from={shotStart} durationInFrames={shotLength}>
  <LivingPhoto src="people/sam.mp4" pos="center 20%" style={{ inset: 0 }} filter="sepia(.2)" />
</Sequence>
```

- Wrap it in a `<Sequence>` that starts at its shot, or the clip plays from frame 0 of the composition and is
  over before it appears.
- It plays forward once and holds its last good frame (`hold`, default 145) with a slow push-in; `after="freeze"`
  holds dead still, for a freeze-frame gag. It never plays backwards.
- `fit={shotLength}` slows a slightly short clip to fill the shot, down to 0.8x (faces look slowed below that).
  Anything shorter needs `ltx_extend.py` (4b). `rate` sets the speed outright (b-roll at 0.5 is natural slow motion).
- **No still photos in a call grid.** A tile that never moves looks broken next to moving ones. Give every cast
  member an idle clip (listening, blinking, small slow nods; one LTX clip each) and use it wherever they aren't
  the one acting.
- `qc.py --render` warns "motion stops dead" when part of the frame was moving for 3 s and then stops while the
  video goes on: the check for exactly this.
- Key props and stamps (a cigar, a "MERGED ✓", a freeze-frame title) to the hold frame, so they land on the still
  face rather than a moving one.
- Render to a new file name every time; the person may still want the previous cut.
