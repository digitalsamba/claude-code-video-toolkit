---
name: explainer-script
description: Turn raw knowledge (an article, notes, data, a topic) into a short storytelling script for a concept-explainer-short video, 15 to 60 seconds, Thai or English. Picks the one idea, extracts the key points, builds hook, explanation, payoff, close, budgets words to seconds and writes scenes.json. Use when someone wants a short educational or infographic-style clip scripted, key points pulled out of a source, or a 30-second knowledge video planned. Triggers include script for a short, summarise into a video, key points, 30 second explainer, infographic video script, storytelling script.
---

# Explainer script

Takes knowledge in, hands `scenes.json` to `templates/concept-explainer-short`. The template makes the
video; this skill decides what the video says.

```
source (article, notes, data, topic) → one idea → key points → beats → words per beat → scenes.json → template
```

## 1. One idea

A 30 second clip carries one idea. Write it as a sentence the viewer could repeat to a friend:
"The sky is blue light, scattered." If the source holds three ideas, pick one and offer the others as
follow-up clips. Never squeeze them in.

## 2. Extract the key points

From the source keep only:
- **The hook fact:** the surprising or counter-intuitive line (a number, a contradiction, a question).
- **Two or three supporting points,** each one a claim plus its reason, in plain words.
- **The payoff:** what the viewer can now see, do or decide.

Drop caveats, history, names and everything the idea survives without. Keep a number only if the video
will say it aloud and show it. Check each figure against the source; never invent one. If the source gives
no figure, write without one.

## 3. Beats and budget

| Beat | Seconds (30s clip) | Job |
|---|---|---|
| Hook | 0 to 4 | Question or tension in the first sentence. No greeting, no "today we will". |
| Points | 4 to 24 | One scene per point, claim then reason. |
| Payoff | 24 to 30 | The idea restated shorter, plus one thing to do or notice. |

Scenes in `scenes.json` are 3 to 4 for 30 seconds. The build adds about 0.3s start, 0.4s lead and 0.8s
tail around the voice (`config.json → timing`), so aim the narration at 26 to 27 seconds for a 30 second clip.

**Word budget**
- English: narration seconds × 2.4 words, so about 65 words for 27 seconds.
- Thai: Thai has no spaces between words, so count characters (without spaces), not words. A trial
  with the macOS Kanya voice measured about 9 characters per second (237 characters took 26.1s, 8.5 to
  10 per scene), so about 235 characters for 27 seconds. Other voices differ, often faster: generate
  the voice, read the per-scene seconds `gen_vo.py` prints, and rescale before rewriting.

## 4. Write it to be heard

- One short sentence at a time; the shortest that holds the thought.
- Spoken words, not written ones. Read it aloud; if you breathe mid-sentence, cut it.
- Show the concrete thing before the abstract name.
- Each scene ends on a word worth lingering on, not a filler.
- No stock phrases. Run the result through `tools/copy_lint.py` (see below).

### Thai

- **Put spaces between phrases** (after about every 2 to 5 syllables, at natural breaths). The template
  splits `text` on spaces to time and chunk the karaoke captions, so a Thai line with no spaces becomes one
  giant caption. Spaces also give the voice its pauses.
- Write the way it is spoken: ภาษาพูด, not ภาษาเขียน. Avoid royal or formal vocabulary unless the topic needs it.
- Write numbers as digits (`30%`, `3 เท่า`) so the voice and the caption agree, and spell out any acronym the
  voice may mis-read.
- Keep a few English loanwords as the audience says them; do not force a translation.

Example, 4 scenes about why the sky is blue (spaces mark the phrase breaks):

```
01 hook    ท้องฟ้าสีฟ้า ทั้งที่แสงแดดเป็นสีขาว แล้วสีฟ้ามาจากไหน
02 point   แสงแดดมีทุกสี แต่โมเลกุลในอากาศ กระเจิงแสงสีฟ้าได้ดีกว่าสีแดงมาก
03 point   ตอนเย็นแสงต้องเดินทางผ่านอากาศหนาขึ้น สีฟ้าหายไปก่อน เหลือแต่สีส้มกับสีแดง
04 payoff  ท้องฟ้าไม่ได้มีสีฟ้า แต่เป็นแสงสีฟ้าที่ถูกกระเจิงมาจากทุกทิศ
```

## 5. Write scenes.json

Follow the template's format exactly (`id`, `slug`, `visual`, `asset`, `text`). Alternate Ideogram cards
(anything with legible text, numbers or a diagram) with LTX motion (atmosphere), and leave `asset` paths
for the visuals step. Optional `titleCard` and per-scene `overlay` (`headline`, `subline`) are in the template
README.

Beside it, set the project's `config.json` for the language:

```json
{
  "font": "/System/Library/Fonts/Supplemental/Krungthep.ttf",
  "captions": { "language": "th", "maxWords": 3, "maxChars": 22 }
}
```

- `font` is a path to a TTF or TTC that has the script (Thai here); `build.py` tries it before its defaults
  (the defaults have no Thai glyphs, so captions would draw as boxes). Pillow here has no Raqm, so it does not
  shape stacked Thai marks: Thonburi shows dotted circles under every tone mark, while macOS's Krungthep
  (bold), Ayuthaya and Sathu draw correctly. Check a still of a caption before trusting any other font.
- `captions.language` is the whisper language code (`th`, `en`). The captions still burn your script text;
  whisper only gives the timing.
- A Thai voice: the built-in `speaker` may not speak Thai well; use a cloned Thai reference
  (`/voice-clone`) or ElevenLabs with a Thai-capable voice, and check by ear before building visuals.

## 6. Hand off to the toolkit

```bash
cp -r templates/concept-explainer-short projects/<name>
# put the scenes.json and config.json above into projects/<name>/
uv run tools/copy_lint.py projects/<name>/scenes.json     # from the toolkit root
cd projects/<name>
uv run build.py                 # placeholders render at once: judge the beat lengths
# then visuals (toolkit root), and in the project directory:
uv run gen_vo.py && uv run gen_captions.py && uv run build.py
uv run tools/qc.py --project projects/<name>              # from the toolkit root, then qc.py --render
```

Details for every step are in the template's `README.md` and `docs/timing.md`, `docs/voiceover.md`,
`docs/qc.md`. After `gen_vo.py`, read the per-scene seconds and the total: over 30 seconds means cutting
words, not speeding the voice up.

## Checklist before handing over

- [ ] One idea, stated in one sentence.
- [ ] Every figure traced to the source.
- [ ] Hook has the tension in its first sentence.
- [ ] Total narration within the budget for the target length.
- [ ] Each scene reads aloud in one or two breaths.
- [ ] Thai: spaces at every phrase break, `font` and `captions.language` set.
- [ ] `copy_lint.py` passes on `scenes.json`.
