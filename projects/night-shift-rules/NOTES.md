# Night Shift horror — production notes

Handoff for the next session. Built with the `youtube-longform` template and skill.

## Status (2026-10-26)

- `scenes.json` / `SCRIPT.md`: story 1, "The Gas Station Rules" — 20 scenes, 652 words,
  ~4.9 min. Script approved in principle; **no assets generated, no credits spent**.
- Placeholder preview renders with `uv run build.py --preview`.
- Next: owner is adding Modal + acemusic keys to the environment, then a fresh session
  deploys the toolkit's Modal tools and produces the long version below.

## Plan: "3 Night Shift Horror Stories Animated" (~25 min)

| # | Story | ~Length | Premise |
|---|-------|---------|---------|
| 1 | The Gas Station Rules | 8 min | Current script, expanded: more rules broken, longer middle act |
| 2 | The Hospital Night Nurse | 8 min | Room 314's call button keeps ringing — that patient was discharged three years ago |
| 3 | The Motel Night Desk | 8 min | Room 9's guest checks in every night at 3:33 a.m., same name, same wet handprint on the counter |

One chapter per story act; stories are self-contained so each can also post alone.

Generation (Modal free tier, ~$6–9 of the $30/month):
- Narration: Qwen3-TTS (`config.json → voice`, provider qwen3, cloud modal), deep slow male
  storyteller tone, `maxWpm` ~145 for horror pacing.
- Stills: FLUX.2 1920x1080, every prompt starts with `styleNote` from scenes.json.
- Motion: ~12 LTX-2 clips (1024x576, 161 frames) for the scare beats — car with no
  headlights, dripping nozzle, wet man's smile, the dawn loop, plus one per new story.
- Music: ACE-Step dark ambient drone (`--preset ambient`, ~180s, looped + ducked).
- Thumbnail: "RULE 7" style text card (Ideogram or gpt-image-2 if no Ideogram key).

Alternative already priced: ElevenLabs connector (free 10k credits) — narration ≈1 credit
per character (voice "Cedric M – Slow-Burn Storyteller", id BQOei2tk6QCBMHQWPhbj), Recraft
v4.1 Flash images ≈48 credits each. Only enough for ~5–8 min; free plan is non-commercial
and needs attribution.

## Research (vidIQ outliers + keywords, Oct 2026)

- Breakout format: **first-person worker stories** — "3 True Babysitter Horror Stories
  Animated" (32x), "6 Creepy Hospital Encounters Shared by Nurses" (244x), "4 … Alone at
  Night Animated" (26x). Mostly 25–40 min anthologies or multi-hour compilations.
- Hook pattern (from the babysitter breakout's transcript): "My name is X and I've been
  working as …" → "That changed the night I …" by 0:18 → money reason to stay → uncanny
  human detail ("smiles stretched a little too wide") → locked exits / dead phones →
  sticky epilogue image. No intro, no CTA.
- Keywords: "animated horror stories" ~242K/mo (+27%), "true horror stories animated"
  +62%, "scary horror stories" +138%. Don't title fiction as "True".
- Keep protagonists faceless/silhouetted — hides image-model inconsistency and reads scarier.
