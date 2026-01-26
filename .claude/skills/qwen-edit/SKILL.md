---
name: qwen-edit
description: AI image editing prompting patterns for Qwen-Image-Edit. Use when editing photos while preserving identity, reframing cropped images, changing clothing or accessories, adjusting poses, applying style transfers, or character transformations. Provides prompt patterns, parameter tuning, and examples.
---

# Qwen-Image-Edit Skill

AI-powered image editing using Qwen-Image-Edit-2511 via RunPod serverless.

**Status:** Evolving - learnings being captured as we experiment

## When to Use This Skill

Use when the user wants to:
- Edit/transform photos while preserving identity
- Reframe cropped images (fix cut-off heads, etc.)
- Change clothing, add accessories
- Change pose (arm positions, hand placement)
- Apply style transfers (cyberpunk, anime, oil painting)
- Adjust lighting/color grading
- Add/remove objects
- Character transformations (Bond, Neo, etc.)

## When NOT to Use

- **Background replacement (single image)** - creates cut-out artifacts, halos
- **Face swapping** - cannot preserve identity from reference
- **Outpainting** - can't extend canvas reliably

## Use With Care

- **Multi-image compositing** - CAN work with explicit identity anchors (see examples.md for prompt patterns). Requires describing distinctive features (hair texture/color, ethnicity, outfit) and using guidance ~2.0
- **Camera angle changes** - Inconsistent results. Vertical angles (low/high) work better than rotational (three-quarter view)

## Quick Reference

```bash
# Basic edit
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"

# With negative prompt (recommended)
python tools/image_edit.py --input photo.jpg \
  --prompt "Reframe as portrait with full head visible" \
  --negative "blur, distortion, artifacts"

# Style transfer
python tools/image_edit.py --input photo.jpg --style cyberpunk

# Background (use cautiously - often fails)
python tools/image_edit.py --input photo.jpg --background office

# Higher quality
python tools/image_edit.py --input photo.jpg --prompt "..." --steps 16 --guidance 3.0

# Multi-image composite (identity-preserving)
python tools/image_edit.py --input person.jpg background.jpg \
  --prompt "The [ethnicity] [gender] with [hair description] from first image is now in [scene] from second image. Same [features], [outfit]." \
  --negative "different ethnicity, different hair color, different face shape, generic stock photo" \
  --steps 16 --guidance 2.0
```

## Key Files

- `prompting.md` - Prompt patterns and structure
- `examples.md` - Good/bad examples from experiments
- `parameters.md` - Tuning steps, guidance, negative prompts

## Tool Location

`tools/image_edit.py` - CLI wrapper for RunPod endpoint

## Related Docs

- `docs/qwen-edit-patterns.md` - Character transformation patterns
- `.ai_dev/qwen-edit-research.md` - Research notes
