# Qwen-Edit Parameters

## Available Parameters

| Parameter | CLI Flag | Default | Range | Notes |
|-----------|----------|---------|-------|-------|
| `num_inference_steps` | `--steps` | 8 | 4-50 | More = higher quality, slower |
| `guidance_scale` | `--guidance` | 1.0 | 1.0-7.0 | Higher = follows prompt more strictly |
| `negative_prompt` | `--negative` | "" | text | Things to avoid |
| `seed` | `--seed` | random | int | For reproducibility |

## Steps (`--steps`)

Controls number of denoising passes.

| Value | Use Case | Inference Time |
|-------|----------|----------------|
| 4 | Quick preview, Lightning LoRA | ~10s |
| 8 | Default, good balance | ~18s |
| 16 | Higher quality, finer detail | ~34s |
| 25+ | Diminishing returns | 45s+ |

**Recommendation:** Start with 8. Use 16 for final renders if detail matters.

## Guidance Scale (`--guidance`)

Controls how strictly the model follows your prompt.

| Value | Behavior |
|-------|----------|
| 1.0 | Default, more creative freedom |
| 3.0 | Stronger adherence, more dramatic |
| 4-5 | Recommended max for Qwen-Edit |
| 7.0+ | Risk of artifacts, over-saturation |

**Note:** Qwen-Edit uses lower guidance than typical diffusion models. Don't go above 5-6.

**Recommendation:** Start with 1.0. Try 3.0 for more literal interpretation.

## Negative Prompt (`--negative`)

Critical for avoiding artifacts. Always consider using one.

### Common Negative Prompts

**For portraits/reframing:**
```
blur, blurry, ghostly, distortion, abstract, haze, fog, out of focus
```

**For zoom/scale changes:**
```
black borders, letterbox, vignette, dark edges, frames, cropped
```

**For viewpoint changes:**
```
silhouette, dark figure, no face, shadow, faceless, artifacts, distortion, black
```

**General quality:**
```
deformed, disfigured, bad anatomy, extra limbs, blurry, low quality
```

## Seed (`--seed`)

Use for reproducibility when iterating on prompts.

```bash
# First attempt
python tools/image_edit.py --input photo.jpg --prompt "..." --seed 12345

# Same seed, different prompt - compare results
python tools/image_edit.py --input photo.jpg --prompt "..." --seed 12345
```

## Cost vs Quality Tradeoffs

| Profile | Steps | Guidance | Time | Cost |
|---------|-------|----------|------|------|
| Quick preview | 4 | 1.0 | ~10s | ~$0.001 |
| Default | 8 | 1.0 | ~18s | ~$0.002 |
| Quality | 16 | 1.0 | ~34s | ~$0.004 |
| Dramatic | 8 | 3.0 | ~18s | ~$0.002 |
| Best | 16 | 3.0 | ~34s | ~$0.004 |

(Costs based on L4 GPU @ $0.34/hr)

## Experimental Parameters

These are supported by the handler but not yet exposed in the CLI:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_fp8` | true | FP8 quantization (false = BF16 full quality) |
| `auto_resize` | true | Auto-resize input for optimal processing |
| `true_cfg_scale` | 4.0 | Internal CFG scale |

To expose these, edit `tools/image_edit.py`.
