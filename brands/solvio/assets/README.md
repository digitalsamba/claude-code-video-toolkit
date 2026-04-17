# Solvio Brand Assets

Place brand assets here:

- `solvio-logo.png` — main logo (transparent PNG, ~512x512)
- `solvio-logo-light.png` — light variant for dark backgrounds
- `solvio-favicon.png` — favicon (64x64)

## Generating logo with FLUX.2

```bash
python tools/flux2.py \
  --prompt "Minimalist tech logo for 'Solvio', bright lime green #9FE600 on black, geometric, modern, transparent background, vector style" \
  --cloud modal \
  --output brands/solvio/assets/solvio-logo.png
```
