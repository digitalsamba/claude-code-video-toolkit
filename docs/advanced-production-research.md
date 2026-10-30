# Production capability research

Research checked on 2026-10-31, before implementation. These are integrations
with existing engines. No upstream source code or model weights are vendored.

| Capability | Projects compared | Decision |
| --- | --- | --- |
| Highlights | [Auto-Editor](https://github.com/WyattBlue/auto-editor), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Audio activity alone can cut sentences; visual scene cuts do not identify useful speech. Use Whisper word timestamps and FFmpeg cuts, with sentence boundaries, repetition removal and explicit keyword ranking. No claim of editorial understanding. |
| Reframing | [OpenCV tracking](https://github.com/opencv/opencv_contrib/tree/4.x/modules/tracking), [Norfair](https://github.com/tryolabs/norfair) | Use OpenCV CSRT for a selected subject and its bundled face detector for automatic speaker framing. Norfair is useful for multiple identities but requires an additional detector. |
| Audio mixing | [FFmpeg](https://github.com/FFmpeg/FFmpeg), [Shotcut](https://github.com/mltframework/shotcut) | Use FFmpeg's sidechaincompress, amix and adelay directly. Avoid introducing a second editor or audio engine. |
| Dubbing | [WhisperX](https://github.com/m-bain/whisperX), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [Demucs](https://github.com/facebookresearch/demucs), [MuseTalk](https://github.com/TMElyralab/MuseTalk) | Import diarized WhisperX segments, map stable speaker IDs to existing TTS providers, fit utterances with FFmpeg atempo. Preserve a supplied background stem or optionally separate one with Demucs. Dialogue synchronization changes audio timing, not mouth pixels. MuseTalk is a separate GPU/model stack and is not bundled. |
| Character consistency | [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [Diffusers](https://github.com/huggingface/diffusers), [FLUX.2](https://github.com/black-forest-labs/flux2) | Reuse tools/flux2.py multi-reference editing. Lock a character description, wardrobe, visual style, reference hashes and shot seeds. Avoid a second diffusion runtime. References improve consistency; they do not guarantee identical identity. |
| Data videos | [Manim](https://github.com/ManimCommunity/manim), [bar_chart_race](https://github.com/dexplo/bar_chart_race), [Matplotlib](https://github.com/matplotlib/matplotlib) | Generalize the existing Matplotlib dependency into CSV/JSON line and bar animation, with data-derived narration and provenance. Manim adds system dependencies; bar_chart_race specializes in wide ranking data. |
| Evidence | [RAGChecker](https://github.com/amazon-science/RAGChecker), [LlamaIndex](https://github.com/run-llama/llama_index) | Use explicit local document quotes and dataset cell references with hashes and script coverage checks. Deterministic evidence checks do not establish semantic truth. Paraphrases require human review rather than a hidden paid LLM judge. |
| Compositing | [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), [rembg](https://github.com/danielgatis/rembg), [FFmpeg](https://github.com/FFmpeg/FFmpeg) | Use FFmpeg chromakey or supplied alpha for ordinary compositing, optional recurrent RVM ONNX inference for moving presenters, and reuse subject tracks for attached graphics. RVM carries temporal state; frame-independent rembg may flicker. |

## Dependency and model boundaries

- OpenCV is Apache-2.0, Matplotlib uses its BSD-style license, ONNX Runtime is MIT.
- Auto-Editor is Unlicense; PySceneDetect is BSD-3-Clause; Norfair is BSD-3-Clause.
- WhisperX is BSD-2-Clause; pyannote.audio and Demucs code are MIT. Their model
  weights and access conditions are separate. Diarization may need a Hugging Face
  token and acceptance of the selected model's conditions.
- FLUX.2 klein 4B uses Apache-2.0; other FLUX variants have different model terms.
  The existing configured endpoint determines the model. No cloud inference is
  needed for preview, validation, or offline verification.
- RVM is GPL-3.0. Its optional user-supplied ONNX model is loaded locally, never
  redistributed here. Review its model and distribution terms for your use.
- FFmpeg build licensing depends on enabled libraries. Use the installed build.

## Verification policy

Exercise actual FFmpeg media, tracker motion, chart frames, audio timing and
compositing pixels locally. Keep model inference checks separate from offline
tests. Do not report a paid TTS/FLUX request, downloaded model, or diarization as
tested unless it was actually run. All work stays on the local branch until the
user requests a push.
