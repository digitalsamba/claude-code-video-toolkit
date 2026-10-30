# Production capability delivery

Implementation branch: `feat/advanced-video-production`.
Base: `e3253a7`, including the previous production PR's three CI fixes.
Published to the `AsharibAli/claude-code-video-toolkit` fork for draft review.
Depends on upstream PR #142, which supplies the production workflow foundation.

Approaches were compared against GitHub projects before coding. See
[the research record](../docs/advanced-production-research.md) for decisions,
upstream links and model/license boundaries, and
[the usage guide](../docs/advanced-production.md) for commands and input formats.

| Capability | Initial implementation commit | Reused engine |
| --- | --- | --- |
| Speech-aware mixing and timed effects | `f75ad67` | FFmpeg sidechaincompress, adelay, amix |
| Complete-sentence highlights | `245643d` | Whisper-compatible timestamps and FFmpeg cuts |
| Subject tracking and reframing | `bcc4278` | OpenCV CSRT and bundled Haar face detector |
| Multi-speaker dubbing | `4eb569a` | WhisperX/pyannote interface, existing TTS, FFmpeg; optional Demucs |
| Character reference shot plans | `c2a724d` | Existing multi-reference FLUX and LTX image-to-video |
| Reusable data videos | `cb4c8d4` | Matplotlib and FFmpeg |
| Source-backed script/project content | `bda268c`, `840adf4` | Local quotes, CSV cells, JSON pointers and content hashes |
| Background replacement and attached graphics | `f35bfec` | FFmpeg chromakey, OpenCV, optional recurrent RVM ONNX |

Later focused commits add validation, import cleanup, discovery, documentation,
and an offline production graph. `7e98cd5` prevents highlight padding from
including the next sentence's speech and rejects nonfinite/oversized numbers.

## Verified locally on 2026-10-31

- 102 Python tests passed, including 28 additional capability/CLI tests.
- Both optional RVM tests ran against the actual published FP32 model, including
  sequential state reuse and a complete background-replacement video.
- The eleven-task example completed, produced six videos with expected
  dimensions and audio, passed final QC, and resumed all eleven unchanged tasks.
- The example also validated character image-to-video dependencies and source
  evidence without invoking providers.
- The repository's undefined-name lint gate, new-tool import/unused-name checks,
  `uv lock --check`, and Git whitespace checks passed.
- Registry registration and generated agent guidance were checked for idempotence.

The actual model used was `rvm_mobilenetv3_fp32.onnx` from the official
[RVM release](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0),
SHA-256 `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828`.
Weights and test environments are outside tracked source.

Run the combined local checks with:

```bash
uv sync --extra vision
uv run python scripts/verify_features.py --capabilities
```

For the two optional RVM checks, also install `--extra matting` and set
`VIDEO_TOOLKIT_RVM_MODEL` to that local model. Otherwise those checks skip.
The new CI configuration exercises the offline graph on Windows and Ubuntu
without downloading model weights. Check the draft PR for current remote CI
results. Existing Remotion/browser code was unchanged and its checks were not
rerun locally during this addition.

## Inference and quality limits

Paid FLUX/LTX/TTS requests were not made. Transport tests exercise the existing
FLUX/LTX request and media decoding paths with simulated responses. They prove
reference/seed propagation, not visual identity quality. Reference-conditioned
shots need visual review.

WhisperX/pyannote and Demucs inference were not executed. Their optional local
integrations follow the upstream APIs; tests use supplied diarized segments,
utterance recordings and background stems. Speaker labels, separation quality
and replacement text need review. Dubbing synchronizes audio, not mouth pixels.

Highlight ranking is deterministic first-pass editing. Tracking can lose a
subject through cuts or occlusion. Evidence checks locate sources and compare
values; they do not establish semantic truth. These limits are documented at
each tool's usage boundary.
