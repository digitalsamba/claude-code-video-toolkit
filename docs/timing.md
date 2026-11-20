# Timing

How long scenes run, how to budget words, and how to keep picture and narration together after the audio is made. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Timing Sync (after voiceover)

```bash
uv run tools/sync_timing.py                          # Dry run comparison
uv run tools/sync_timing.py --apply                  # Update config (1s default padding)
uv run tools/sync_timing.py --apply --padding 1.5    # Custom padding
uv run tools/sync_timing.py --voiceover-json vo.json # Use voiceover.py output
uv run tools/sync_timing.py --json                   # Machine-readable output
```

Knows sprint-review v1/v2 and product-demo configs, and falls back to a **generic mode** for any
config that pairs `*AudioFile` with `*DurationSeconds` fields (e.g. `tickets[]` with intro/outro
narration), so new template shapes work without touching the tool.

## Video Timing

Timing is critical. Keep these guidelines in mind:

## Pacing Rules
- **Voiceover drives timing**: Narration length determines scene duration
- **Reading pace**: ~150 words/minute (2.5 words/second) for standard narration
- **Demo pacing**: Real-time demos often need 1.5-2x speedup (`playbackRate`)
- **Transitions**: Add 1-2s padding between scenes
- **FPS**: All videos use 30fps (frames = seconds × 30)

## Speaking Rate Tiers

| Pace | WPM | Use When |
|------|-----|----------|
| Slow | 120-130 | Technical explanations, complex concepts |
| Standard | 140-160 | General narration, demos, overviews |
| Fast | 160-180 | Energetic intros, recaps, CTAs |

## Narration Density by Scene Type

| Scene Type | Duration | Narration Density | Notes |
|------------|----------|-------------------|-------|
| Title | 3-5s | 0-10% | Logo + headline, let visuals breathe |
| Overview | 10-20s | 70-90% | 3-5 bullet points, narration-heavy |
| Demo | 10-30s | 30-50% | Let the demo speak, narrate key moments only |
| Stats | 8-12s | 70-90% | Read out highlights, skip obvious numbers |
| Credits | 5-10s | 0-20% | Quick fade, maybe a closing line |
| Problem/Solution | 10-15s | 80-90% | Narration drives the story |
| CTA | 5-10s | 60-80% | Clear call to action, leave a beat at end |

## Word Count Budgeting

Before writing scripts, budget words per scene:

```
Target duration × 2.5 = word budget (at standard pace)
Pause seconds × 2.5 = words to subtract from budget

Example: 15s scene with a 1s pause
  15 × 2.5 = 37 words budget
  1 × 2.5 = 3 words for pause
  Available: ~34 words of narration
```

Use `[pause 1.0s]` markers in scripts. Each second of pause costs ~2-3 words from the budget.

## Timing Calculations
```
Script words ÷ 150 = voiceover minutes (estimate)
Raw demo length ÷ playbackRate = demo duration
Sum of scenes + transitions = total video
```

## When to Check Timing
- **During scene planning**: Budget word counts per scene before writing
- **After writing script**: Count words per scene, compare to budget
- **After generating audio**: Run `sync_timing.py` to compare actual vs estimated
- **Before rendering**: Ensure `durationInFrames` matches actual audio for each scene

## TTS Duration Drift (The Real Timing Problem)

TTS engines do NOT consistently produce 150 WPM output. In practice:
- **ElevenLabs** tends to compress pauses and speed through short sentences. A 50s script may produce 40-45s of audio.
- **Qwen3-TTS** varies by speaker and tone preset. Ryan at "professional" tone speaks ~10% faster than "warm."
- **Short scenes drift more**: a 5-second scene might be off by 30%, while a 30-second scene is off by 10%.

**The feedback loop after TTS generation:**

1. Generate per-scene audio files
2. Run `uv run tools/sync_timing.py` to compare actual vs config durations
3. Run `uv run tools/sync_timing.py --apply` to update config automatically
4. For demo scenes: recalculate `playbackRate = rawDemoDuration / actualNarrationDuration`
5. Re-preview in Remotion Studio before rendering

**Common drift patterns and fixes:**

| Problem | Symptom | Fix |
|---------|---------|-----|
| Audio shorter than scene | Dead air / awkward silence at end | Reduce `durationInFrames` to match audio |
| Audio longer than scene | Narration cut off | Increase `durationInFrames` or trim script |
| Demo too fast for narration | Viewer can't follow | Decrease `playbackRate` or cut narration |
| Demo too slow for narration | Waiting for demo to catch up | Increase `playbackRate` (1.5-2x typical) |
| Pauses lost in TTS | Script felt spacious, audio feels rushed | Add explicit `<break time="1s"/>` in SSML or extend scene padding |
| Cloned voice rushes | Scenes land >170 wpm; retakes also fast (temperature doesn't help) | Pace is inherited from the reference audio. Use `--max-wpm 165` (pitch-preserving atempo clamp, both voiceover.py and qwen3_tts.py; if a take still lands fast it hit the 0.85x floor, so add `--min-tempo 0.8`), and record clone references as 12-25s of varied sentences *at narration pace* |

**Pacing QC:** voiceover.py and qwen3_tts.py report `wpm` and a `pacing` label
(fast/slow/ok) per generated file, check these before composition. Comfortable
narration is 140-160 wpm; flag anything over ~170. `--max-wpm` turns the check
into an automatic fix.

## Fixing Mismatches
- **Voiceover too long**: Speed up demos, trim pauses, cut content
- **Voiceover too short**: Slow demos, add scenes, expand narration
- **Demo too long**: Increase `playbackRate` (1.5x-2x typical)
- **Demo too short**: Decrease `playbackRate`, or loop/extend

## Audio-Anchored Timelines (the prevention approach)

`sync_timing.py` is reactive, it fixes drift after the fact. You can prevent drift entirely by **generating the audio first, then anchoring visuals to known timestamps** instead of estimating durations upfront.

**The pattern:**

1. Write the script and split into per-scene segments
2. Generate per-scene VO files: `voiceover.py --scene-dir public/audio/scenes --json`
3. Read the actual durations from the JSON output
4. Anchor every visual element to absolute timestamps in the timeline

This is especially clean for Python/moviepy builds where each clip carries its own `start=` parameter:

```python
# Audio-anchored scene timeline (25s total):
#   Scene 1 tired      0.3 → 3.74  (audio 3.44s)
#   Scene 2 worries    4.0 → 8.88  (audio 4.88s)
#   Scene 3 introduce  9.1 → 11.90 (audio 2.80s)

text_clip("TIRED OF",     start=0.5,  duration=1.2)
text_clip("THIRD-PARTY",  start=1.0,  duration=1.8)
vo_clip("01_tired.mp3",   start=0.3)
vo_clip("02_worries.mp3", start=4.0)
```

The comment block at the top is the source of truth. Every `start=` references it. Drift is impossible because durations aren't being estimated, they're being read from the rendered audio.

**Trade-off vs. `<Series>`-style auto-chaining:**

| Approach | Best for | Downside |
|----------|----------|----------|
| Audio-anchored absolute starts | Tight ad-style edits, sub-30s spots, anything with exact timing | Manual bookkeeping when re-timing a scene |
| `<Series>` / auto-chained durations | Long-form sprint reviews where adjacent scenes flex | Drift compounds across the timeline; needs `sync_timing.py` to recover |

For Remotion projects you can mix the two: use `<Sequence from={...}>` with absolute frames for tight sections and let `<Series>` handle the rest. For pure-Python builds (`build.py` + moviepy), audio-anchored is the natural default.
