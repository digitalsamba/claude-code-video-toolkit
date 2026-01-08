# SadTalker Integration Progress

**Branch:** `experiment/sadtalker`
**Last updated:** 2026-01-07 00:45
**Status:** ✅ WORKING

## Completed
- [x] Dockerfile with baked-in models (~1.5GB)
- [x] handler.py with R2 integration and chunking (45s chunks)
- [x] tools/sadtalker.py CLI with --setup
- [x] README.md documentation
- [x] Build and push Docker image to ghcr.io
- [x] Deploy endpoint via --setup (working)
- [x] Fixed BFM_Fitting.zip extraction (was creating nested dirs)
- [x] Added hub.zip face detection models
- [x] Added facexlib detection models
- [x] Added verbose error logging to handler.py
- [x] **Fixed numpy 2.0 compatibility** (constraints file + sed patches)
- [x] **Fixed trans_params ValueError** (dtype=object fix from PR #859)
- [x] **Fixed output video detection** (check direct files, not just subdirs)
- [x] **Tested with short audio (15s)** - works!
- [x] **Tested with long audio (2.3 min, 4 chunks)** - chunking works!

## NumPy 2.0 Incompatibility - RESEARCHED FIX

**Problem**: SadTalker AND its dependencies (facexlib, gfpgan, basicsr) use deprecated numpy APIs removed in numpy 2.0:
- `np.float` → use `float` or `np.float64`
- `np.int` → use `int` or `np.int64`
- `np.VisibleDeprecationWarning` → use `DeprecationWarning`

**Root cause**: The original `pip install "numpy<2"` was only in the first pip command. The second pip install (gfpgan/basicsr/facexlib) would upgrade numpy to 2.0. Additionally, sed patches only fixed SadTalker code, not the dependencies in site-packages.

**Research sources**:
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
- [SadTalker GitHub issues](https://github.com/vladmandic/automatic/issues/1076)

**Solution (implemented)**:
1. **Constraints file**: `echo "numpy<2" > /app/constraints.txt` then use `-c /app/constraints.txt` for ALL pip installs
2. **Verification step**: Assert numpy version starts with "1."
3. **Expanded sed patches**: Patch both SadTalker AND dependencies in site-packages (facexlib, basicsr, gfpgan)

**Current Dockerfile approach:**
```dockerfile
# Constraints file enforces numpy<2 across ALL installs
RUN echo "numpy<2" > /app/constraints.txt
RUN pip install --no-cache-dir -c /app/constraints.txt gfpgan basicsr facexlib realesrgan

# Verify numpy version
RUN python -c "import numpy; v=numpy.__version__; assert v.startswith('1.')"

# Patch SadTalker AND dependencies (belt-and-suspenders)
RUN find /app/SadTalker -name "*.py" -exec sed -i 's/np\.float\b/float/g' {} \; && \
    find /usr/local/lib -name "*.py" -path "*/facexlib/*" -exec sed -i 's/np\.float\b/float/g' {} \;
```

## Performance Benchmarks

| Audio Duration | Chunks | Processing Time | Output Size |
|---------------|--------|-----------------|-------------|
| 15s | 1 | ~3.5 min | 1.1 MB |
| 2.3 min (138s) | 4 | ~14.2 min | 9.8 MB |

**Average:** ~3.5 minutes per 45-second chunk

## Optional Enhancements
- [ ] Template swapping support
- [ ] Higher resolution (512px) option testing
- [ ] Batch processing support

## Resume Instructions

```bash
# Switch to branch
git checkout experiment/sadtalker

# Rebuild Docker image with numpy patches
cd docker/runpod-sadtalker
docker build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-sadtalker:latest .
docker push ghcr.io/conalmullan/video-toolkit-sadtalker:latest

# Delete old endpoint/template and recreate
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('RUNPOD_API_KEY')
endpoint_id = os.getenv('RUNPOD_SADTALKER_ENDPOINT_ID')
requests.post('https://api.runpod.io/graphql',
    json={'query': f'mutation {{ deleteEndpoint(id: \"{endpoint_id}\") }}'},
    headers={'Authorization': f'Bearer {api_key}'})
requests.post('https://api.runpod.io/graphql',
    json={'query': 'mutation { deleteTemplate(templateName: \"video-toolkit-sadtalker\") }'},
    headers={'Authorization': f'Bearer {api_key}'})
"

# Setup fresh endpoint
python3 tools/sadtalker.py --setup

# Test with SadTalker's example image
curl -sL "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/full_body_1.png" -o /tmp/test_portrait.png
ffmpeg -y -i some_audio.mp3 -ss 0 -t 15 /tmp/test_audio_15s.mp3
python3 tools/sadtalker.py --image /tmp/test_portrait.png --audio /tmp/test_audio_15s.mp3 --output /tmp/test.mp4
```

## Key Files
- `docker/runpod-sadtalker/Dockerfile` - Docker build with numpy patches
- `docker/runpod-sadtalker/handler.py` - RunPod worker with verbose logging
- `tools/sadtalker.py` - CLI tool

## Current RunPod Resources
- **Endpoint ID:** `efkkbuyp7aupl8`
- **Template ID:** `11dbptpqjo`
- **Image:** `ghcr.io/conalmullan/video-toolkit-sadtalker:latest`

## Alternative Approaches to Try
1. Use an older base image with numpy 1.x pre-installed
2. Fork SadTalker and apply numpy 2.0 fixes upstream
3. Use a different talking head model (e.g., Wav2Lip, LivePortrait)

## See Also
- `.ai_dev/sadtalker-integration.md` - Full integration plan (gitignored)
