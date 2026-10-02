---
name: ideogram4
description: Prompting patterns for Ideogram 4.5 / 4 text-to-image (plus 4.5 precise edit) — best-in-class in-image text rendering and exact color/layout control via structured JSON captions. Use when generating images that need legible on-image text (title cards, thumbnails, logos, signage, CTAs), precise brand colors, or controlled spatial layout, or when fixing a detail in an existing card without regenerating it. Triggers include title slide image, thumbnail with text, on-image text, legible text in image, brand color palette image, bounding-box layout, precise edit, Ideogram.
---

# Ideogram 4 Skill

Text-to-image generation with **Ideogram 4.5** (the newest model and the tool's default) and
**Ideogram 4** (9.3B, open-weight, released June 2026). Their superpower is **best-in-class
in-image text rendering** — Ideogram 4 beats much larger models (FLUX.2 dev 32B, Qwen-Image 20B,
Hunyuan 80B) at rendering legible signage, logos, captions, and multi-line text — plus **exact
color-palette and bounding-box control**. Ideogram 4.5 adds **precise edit**: describe one
change and every pixel the edit doesn't touch is copied exactly from the source image.

That advantage is **locked behind a structured JSON caption format**. A plain-text prompt gets
you FLUX-level results and misses the entire point of using this model. This skill teaches
Claude to act as the "magic prompt" expander — turning a user's casual request into the JSON
caption Ideogram 4 was trained on. Ideogram 4.5 accepts the same caption, so one format covers
both models.

> **Backend:** The toolkit uses Ideogram's **hosted v2 API** (not self-hosted weights). The API
> takes a single `prompt` field that accepts either natural language or a structured JSON
> caption; a valid caption is used as is and skips server-side magic prompt. So everything this
> skill teaches applies directly — Claude builds the caption, the tool serializes it into
> `prompt`. Paid API plans include a **commercial license**, which the self-hostable weights
> (non-commercial) do not — that's why we use the API.
> Cost: Ideogram 4 is $0.03 (turbo) / $0.06 (default) / $0.10 (quality) per image. Ideogram 4.5
> is priced by quality tier and output size (1K vs 2K) — run the tool with `--dry-run` for an
> exact, free quote of the request you're about to send.

## When to Use This Skill

Reach for Ideogram 4 (over FLUX.2) when the image needs:
- **Legible on-image text** — title cards, thumbnails, lower-thirds backgrounds, signage, logos,
  quote cards, CTAs with a headline baked in
- **Exact brand colors** — hex color-palette conditioning, per-element
- **Controlled layout** — bounding boxes place text/objects in specific regions
- **Multilingual text** in the image

Use **FLUX.2** instead when: the image has no critical text, or you just want a fast atmospheric
background. FLUX takes plain natural-language prompts; Ideogram
wants JSON. See `tools/flux2.py`.

## The One Thing to Get Right

**Always emit a structured JSON caption, not a plain sentence.** The model is trained
*exclusively* on JSON captions that name every element explicitly. Claude is a better expander
than Ideogram's free hosted magic-prompt (their own docs note the shipped one "is not the same
used in production"), so build the caption yourself using this skill rather than passing raw text.

Minimal valid caption:

```json
{"high_level_description":"A sailboat at sunset on calm water.","style_description":{"aesthetics":"serene, warm, golden hour","lighting":"golden hour backlighting","photo":"wide angle, f/8","medium":"photograph","color_palette":["#FF6B35","#F7C59F","#004E89"]},"compositional_deconstruction":{"background":"Calm ocean at low horizon with orange-pink sky.","elements":[{"type":"obj","desc":"White triangular sail silhouetted against the setting sun."}]}}
```

Full schema, strict key-ordering rules, and the bbox coordinate system are in **`prompting.md`**.
Worked title-card / thumbnail / quote-card examples are in **`examples.md`**.

## Quick Reference — `tools/ideogram4.py`

> Thin wrapper over Ideogram's hosted v2 API. Needs `IDEOGRAM_API_KEY` in `.env`
> (key from developer.ideogram.ai). `--json` serializes the caption into the API's `prompt`
> field (a valid caption skips server-side magic prompt — Claude is the expander); `--prompt`
> posts plain text, which magic prompt expands (`--magic-prompt auto|on|off`).

```bash
# Hand-authored JSON caption (the recommended path for text/layout) — Claude writes caption.json
uv run tools/ideogram4.py --json caption.json --output title.png

# Caption from stdin (Claude can pipe it directly)
cat caption.json | uv run tools/ideogram4.py --json - --output title.png

# Plain prompt — Ideogram's server-side magic prompt expands it (weaker; prefer --json)
uv run tools/ideogram4.py --prompt "Title card: 'AI ENGINEERING REVIEW' bold white on dark" --output title.png

# Inject brand hex colors into the caption's palette (JSON mode)
uv run tools/ideogram4.py --json caption.json --brand digital-samba --output cta.png

# Quality tier + size (Ideogram 4.5, the default model)
uv run tools/ideogram4.py --json caption.json --quality high --resolution 2048x2048 --output slide.png

# Free price quote for the exact request (validated, nothing generated or billed)
uv run tools/ideogram4.py --json caption.json --resolution 1440x2560 --dry-run

# Ideogram 4 instead of 4.5
uv run tools/ideogram4.py --model 4 --json caption.json --speed quality --resolution 1440x2560 --output card.png

# Precise edit (4.5): fix one detail, keep everything else pixel-exact
uv run tools/ideogram4.py --edit title.png --prompt "Change the headline colour to #FF6B35. Keep everything else the same." --output title_v2.png
uv run tools/ideogram4.py --edit shot.png --mask mask.png --reference logo.png \
    --prompt "Put this logo on the mug" --output shot_logo.png
```

**Per-model options** — the tool exits with a clear error if you pass a knob the model lacks:

| Option | `--model 4.5` (default) | `--model 4` |
|--------|-------------------------|-------------|
| Quality / speed | `--quality low\|medium\|high` (default `high`) | `--speed turbo\|default\|quality` (no FLASH on v2) |
| `--resolution` (alias `--size`) | `auto` or `WxH`: multiples of 32, ≤ 2048×2048 px total, ≤ 6:1 (e.g. `1440x2560`, `2048x2048`, `1440x2880`) | one of the 4.0 presets (e.g. `1440x2560`, `2048x2048`, `2560x1440`) |
| `--edit` (precise edit) | yes — output keeps the input's size, so no `--resolution` | no |
| `--seed`, `--num-images` (1-8), `--magic-prompt`, `--copyright-detection`, `--dry-run` | yes | yes |

**Precise edit notes:** the prompt is the edit instruction (natural language works best — say
what to change *and* "keep everything else the same"). `--mask` is **black = edit, white = keep**
(the opposite of many inpainting tools) and must match the input's width and height. Up to 4
`--reference` images guide the edit (3 with a mask). Default quality for edits is `medium`;
`very_low` is the cheapest and only valid for edits. JPEG/PNG/WEBP, ≤ 25MB each.

## Key Files

- `prompting.md` — full JSON schema, strict key ordering, bbox coordinate system, palette rules
- `examples.md` — worked captions for title cards, thumbnails, quote cards, brand CTAs

## Video Production Fit

Ideogram 4's niche in the toolkit is **slides and thumbnails with baked-in text**, where FLUX and
LTX-2 fail (both render garbled text). Natural pairings:

| Use case | Why Ideogram 4 |
|----------|----------------|
| Title-card / CTA background **with headline text** | Legible text + exact brand hex colors in one pass |
| YouTube/social **thumbnail with a punchy phrase** | Big readable text is its strongest suit |
| Quote card / stat card | Multi-line text + layout control via bboxes |
| Signage/logos inside a product-demo scene | In-image text other models can't render |
| Fix a typo or colour on a card you already like | Precise edit (`--edit`) changes only that detail — no reroll |

Then feed the still into Remotion (`<OffthreadVideo>`/`Img`) or animate it with `tools/ltx2.py --input`.
