# Check an intentionally silent video

Some motion graphics and visual-only deliverables intentionally have no audio.
QC still rejects missing or digitally silent audio by default. Use this explicit
flag only when silence is part of the deliverable:

```bash
uv run tools/qc.py --render output.mp4 --allow-silent
```

The flag changes only the audio-presence/silence finding to information.
Picture checks, missing-file errors, scene timing and other QC checks still run.
It does not add audio or suppress errors for an unreadable file.

This isolates the small QC option previously bundled in #142/#143. It uses the
installed FFmpeg/ffprobe and adds no packages, service, model or API cost.
Toolkit code is MIT; FFmpeg licensing follows its installed build.

```bash
uv run python -m unittest discover -s tests/python -p test_qc_silent.py -v
```
