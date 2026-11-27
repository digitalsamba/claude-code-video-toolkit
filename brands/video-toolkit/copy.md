# Video toolkit copy

Write for someone making a video. Name the input, the action and the result. Say what a tool can produce and give measured costs or timings when they help the reader choose a workflow.

- The name is Agent Video Toolkit. Use `agent-video-toolkit` for the repository slug and “the toolkit” when the project is already clear. It was Claude Code Video Toolkit until v0.31.0.
- Keep product names in their published form. Write Claude Code, Codex, OpenCode, GitHub, ElevenLabs and FFmpeg.
- Write voiceover and timestamp as one word. Use 1080p and 30 fps for render settings.
- Check claims about free use or offline operation against the exact workflow. For costs, give the provider, units and scope of the estimate.

```bash
uv run tools/copy_lint.py --brand video-toolkit narration.txt
```
