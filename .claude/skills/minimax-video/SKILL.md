---
name: minimax-video
description: Generate text-to-video and image-to-video clips with the MiniMax hosted video API. Use for cinematic b-roll, animated stills, product shots, social clips, camera-controlled motion, and cloud video generation without deploying a GPU endpoint.
---

# MiniMax Video Generation

Generate video clips through the hosted MiniMax video API with `tools/minimax_video.py`.
The tool creates an asynchronous task, polls its status, retrieves the generated file,
and saves the final MP4.

Set `MINIMAX_API_KEY` in `.env` before running the tool.

## Quick Reference

```bash
# Text-to-video in the global region
python3 tools/minimax_video.py \
  --prompt "A cinematic aerial shot over a coastal city at golden hour" \
  --output coast.mp4

# Image-to-video from a local first frame
python3 tools/minimax_video.py \
  --prompt "Slow camera push-in, natural wind moving the trees" \
  --input first-frame.png \
  --output animated.mp4

# China region
python3 tools/minimax_video.py \
  --region cn_zh \
  --prompt "A clean studio product reveal with soft reflections" \
  --output reveal.mp4

# Fast image-to-video model
python3 tools/minimax_video.py \
  --model MiniMax-Hailuo-2.3-Fast \
  --input first-frame.png \
  --prompt "The subject turns toward the camera" \
  --output fast.mp4
```

## Modes

| Mode | Required input | Recommended model |
|------|----------------|-------------------|
| Text-to-video | `--prompt` | `MiniMax-Hailuo-2.3` |
| Image-to-video | `--prompt` and `--input` | `MiniMax-Hailuo-2.3` |
| Fast image-to-video | `--prompt` and `--input` | `MiniMax-Hailuo-2.3-Fast` |

`--input` accepts a local JPG, PNG, or WebP under 20 MB, a public image URL, or an
image data URL. Local images are encoded as data URLs before submission.

## Regions

| Flag | API region |
|------|------------|
| `--region global_en` | Global English API, default |
| `--region cn_zh` | China API |

Keep the region consistent with the account that issued `MINIMAX_API_KEY`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | required | Video content and motion description |
| `--input` | none | First-frame image for image-to-video |
| `--output` | required | Output MP4 path |
| `--model` | `MiniMax-Hailuo-2.3` | Video generation model |
| `--duration` | `6` | `6` or `10` seconds |
| `--resolution` | `768P` | `512P`, `768P`, or `1080P`, subject to model limits |
| `--region` | `global_en` | `global_en` or `cn_zh` |
| `--no-prompt-optimizer` | off | Disable server-side prompt optimization |
| `--fast-pretreatment` | off | Reduce prompt optimization time on supported models |
| `--generation-timeout` | `900` | Overall async generation timeout in seconds |
| `--poll-interval` | `10` | Task status polling interval in seconds |
| `--json-out` | off | Print a machine-readable result line |

Ten-second generation uses `768P`. The fast model is image-to-video only.

## Prompting

Describe the scene, subject motion, camera motion, lighting, and visual style in that
order. Keep the action achievable within the requested duration.

```text
A ceramic perfume bottle on black stone, mist drifting across the surface,
[Push in], soft rim lighting, premium product commercial, shallow depth of field.
```

Supported camera instructions can be placed in square brackets, including:

- `[Truck left]` or `[Truck right]`
- `[Pan left]` or `[Pan right]`
- `[Push in]` or `[Pull out]`
- `[Pedestal up]` or `[Pedestal down]`
- `[Tilt up]` or `[Tilt down]`
- `[Zoom in]` or `[Zoom out]`
- `[Shake]`, `[Tracking shot]`, or `[Static shot]`

Use no more than three simultaneous camera instructions. For sequential moves, place
the instructions in the prompt in the order they should occur.

## Image-to-Video Guidance

- Use a sharp first frame with a clear subject and intentional composition.
- Describe motion that follows naturally from the still image.
- Specify camera movement separately from subject movement.
- Avoid asking for major identity, wardrobe, or scene changes in a short clip.
- Use `--no-prompt-optimizer` when precise camera instructions must remain unchanged.

## Troubleshooting

### API key error

Confirm `MINIMAX_API_KEY` is present in `.env` and that `--region` matches the account.

### Generation timeout

Increase `--generation-timeout`. The task remains associated with the API account even
if the local polling process stops.

### Unsupported duration or resolution

Use `6` seconds for `1080P`, or switch to `768P` for `10` seconds.

### Image rejected

Use JPG, PNG, or WebP under 20 MB. The short edge must be greater than 300 pixels and
the aspect ratio must stay between 2:5 and 5:2.
