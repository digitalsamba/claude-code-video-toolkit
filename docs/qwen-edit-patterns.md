# Qwen-Image-Edit Patterns & Learnings

Documented from session on 2026-01-01. Milestone 1.2 of video generation pipeline.

## Model Overview

**Model:** Qwen/Qwen-Image-Edit-2511
**Pipeline:** QwenImageEditPlusPipeline (diffusers)
**Task:** Identity-preserving image editing

## GPU Requirements

| Mode | VRAM | GPU Options | Cost/hr |
|------|------|-------------|---------|
| BF16 (full) | ~40GB | A6000, L40S, A100 | $0.89-2.17 |
| FP8 | 24GB | L4, RTX 4090 | $0.34 |
| Quantized | 12-17GB | RTX 3090 | $0.20 |

**Current setup:** A100 80GB (overkill - could use 48GB)

## Cost Estimates

- **Per image:** ~$0.012 (20s inference on A100)
- **Per 1000 images:** ~$12
- **With L4 + FP8:** Could reduce to ~$0.004/image

## Inference Settings

```python
pipe(
    image=[input_image],
    prompt=prompt,
    negative_prompt=" ",
    num_inference_steps=8,      # 4-8 recommended
    guidance_scale=1.0,
    true_cfg_scale=4.0,
    generator=generator,
)
```

## Prompt Patterns That Work

### Character Transformations
```
"{Character name}, wearing {costume details}, {pose/action}, {background}, {lighting style}, {aesthetic keywords}"
```

**Examples that worked well:**
- `"James Bond 007, wearing a tuxedo with bow tie, holding a martini glass, casino background, suave and sophisticated, cinematic lighting, spy thriller aesthetic"`
- `"Neo from The Matrix, wearing long black leather trench coat, dark sunglasses, green digital rain code background, cyberpunk aesthetic, dramatic pose"`
- `"Forrest Gump running across America, long beard, plaid shirt, running shorts, dusty desert highway, golden hour, inspirational"`

### Professional/Corporate
```
"{Role}, {attire}, {setting}, {lighting}, {mood keywords}"
```

**Examples:**
- `"Tech CEO presenting on stage at a conference, modern blue lighting, professional speaker, sleek tech backdrop"`
- `"Professional video call host in a modern home studio, ring light reflection in glasses, clean minimal background, webcam perspective"`

### Cinematic Scenes
```
"{Scene description}, {costume}, {environment}, {lighting}, {film style}"
```

## What Works Well

1. **Strong visual references** - Iconic costumes, recognizable scenes
2. **Lighting keywords** - "cinematic", "dramatic", "golden hour", "neon"
3. **Specific clothing** - "tuxedo with bow tie", "leather trench coat", "flight suit"
4. **Environment details** - "casino background", "desert highway", "ancient temple"
5. **Mood/aesthetic** - "spy thriller", "cyberpunk", "1920s Birmingham"

## What Doesn't Work Well

1. **Too dark scenes** - Jedi temple scene lost identity
2. **Vague prompts** - Need specific visual details
3. **Extreme transformations** - Corporate headshot drifted from identity
4. **Missing context** - Just saying "Stranger Things" isn't enough

## Identity Preservation Tips

1. Keep lighting balanced (not too dark)
2. Preserve glasses if subject wears them (mention in prompt if needed)
3. Action poses work better than static
4. Avoid prompts that would naturally change face shape

## Output Characteristics

- **Input:** Any size (auto-resized)
- **Output:** ~768x1376 (portrait) or similar
- **Format:** PNG
- **Processing:** ~20s per image at 8 steps

## RunPod Deployment

### Template Config
```
GPU: AMPERE_80 (A100 80GB) - can reduce to AMPERE_48 or L4
Container Disk: 120GB
Image: ghcr.io/conalmullan/video-toolkit-qwen-edit@sha256:...
```

### Handler Pattern
```python
# Load directly from HuggingFace (handles caching)
from diffusers import QwenImageEditPlusPipeline

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

### Key Learnings
1. Let diffusers handle model caching (don't use snapshot_download to local dir)
2. Set HF_HOME to network volume for persistent cache
3. Pre-load pipeline at startup for faster first request
4. Cold start: ~3 min, Warm: ~20s inference

## Test Script

```bash
# Basic test
python tools/test_qwen_edit.py --image photo.jpg --prompt "description" --steps 8

# With seed for reproducibility
python tools/test_qwen_edit.py --image photo.jpg --prompt "description" --seed 42
```

## Sample Results

| Prompt Theme | Identity | Quality |
|--------------|----------|---------|
| James Bond (tux, casino) | Excellent | Excellent |
| Neo (Matrix, leather coat) | Good | Excellent |
| John Wick (tactical, neon) | Excellent | Excellent |
| Top Gun (flight suit) | Excellent | Excellent |
| Forrest Gump (running) | Good | Excellent |
| Indiana Jones (fedora, whip) | Good | Excellent |
| Peaky Blinders (flat cap) | Good | Excellent |
| Corporate headshot | Drifted | Good |
| Jedi (dark temple) | Lost | Poor |

## Next Steps

1. **Reduce GPU cost** - Switch to 48GB or 24GB with FP8
2. **Pipeline integration** - Connect to Wan I2V for video generation
3. **Batch processing** - Multiple edits per request
4. **LoRA support** - Custom style fine-tuning

## Files Created

```
/Users/conalmullan/work/video/
├── conor_bond.png
├── conor_neo.png
├── conor_wick.png
├── conor_maverick.png
├── conor_gump.png
├── conor_indy.png
└── conor_shelby.png
```

## References

- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Diffusers QwenImage Pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)
- [RunPod Serverless Pricing](https://docs.runpod.io/serverless/pricing)
