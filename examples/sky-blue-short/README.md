# sky-blue-short — concept-explainer-short showcase

A complete worked example of the **concept-explainer-short** template: a 52-second
9:16 vertical short answering *"Why is the sky blue?"* — every asset AI-generated,
total cloud cost ≈ $0.80.

All source assets are committed, so you can study or rebuild it immediately:

```bash
cd examples/sky-blue-short
python3 build.py          # re-render from committed assets → out/short.mp4
```

## How it was made

| Piece | Tool | Notes |
|-------|------|-------|
| 4-scene plan + narration | `scenes.json` | hook → concept card → payoff → CTA |
| Voiceover | `gen_vo.py` → Qwen3-TTS (Modal) | built-in speaker Ryan, `tone: warm`, one batch call; pacing QC reported 125–152 wpm, no clamping needed |
| Sky + sunset b-roll | `tools/ltx2.py` | 576×1024, 161 frames, seed 11 — prompts in the table below |
| Scattering diagram + CTA cards | `tools/ideogram4.py` | JSON captions in `captions/02_scattering.json` / `04_cta.json`, `--resolution 1440x2560` |
| Music bed | `tools/music_gen.py` | 60s ambient, looped + ducked by build.py |
| Caption timing | `gen_captions.py` | whisper word timestamps force-aligned to the script |

LTX prompts used:

- `01_hook` — "Low angle looking straight up at a brilliant deep blue sky, wispy
  white clouds drifting slowly overhead, sun flaring at the edge of the frame
  with soft sunbeams, peaceful atmosphere, cinematic, vertical composition"
- `03_sunset` — "A glowing orange and red sunset over the ocean horizon, the sun
  low and golden, clouds lit pink and amber, gentle waves catching warm light,
  slow camera drift, cinematic"

## Things this example demonstrates

- **Card ↔ motion alternation**: LTX clip → Ideogram card → LTX clip → Ideogram
  card, a pattern interrupt roughly every 13 seconds.
- **Caption-safe card design**: the scattering card was regenerated once to keep
  its bottom fifth clear — burned captions occupy y≈1640+, so card content must
  end above that. (The first version had a tagline there; the caption pill
  covered it.)
- **Pacing QC in practice**: built-in speakers pace naturally; voice *clones*
  inherit pace from their reference and often need `voice.maxWpm` — see the
  template README.

The `captions/*.json` Ideogram caption files are committed alongside the
`words_*.json` timing files as documentation of how each card was authored.
