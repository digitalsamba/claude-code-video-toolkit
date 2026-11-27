# Video jobs from OpenCode

[OpenCode](https://opencode.ai/) can write specs for short explainers and call the toolkit's existing MCP server. The setup below was tested on Windows with OpenCode 1.19.6 and the hosted DeepSeek V4.1 Flash route. Its [model weights](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) are public.

## Connect the toolkit

Install `uv`, FFmpeg and OpenCode, then clone the toolkit. Create an `opencode.json` file in a separate working directory and replace `/path/to/agent-video-toolkit` with the absolute path to your checkout. Windows paths can use forward slashes.

```json
{
  "mcp": {
    "toolkit": {
      "type": "local",
      "command": [
        "uv", "run", "--locked", "--directory", "/path/to/agent-video-toolkit",
        "--extra", "mcp", "tools/video_mcp.py"
      ]
    }
  },
  "permission": {
    "*": "deny",
    "read": "allow",
    "external_directory": "deny",
    "toolkit_video_estimate": "allow",
    "toolkit_video_status": "allow",
    "toolkit_video_result": "allow"
  }
}
```

This configuration allows OpenCode to estimate costs and inspect jobs while you control submissions from the CLI. For other OpenCode versions, follow the [MCP configuration for your release](https://opencode.ai/docs/mcp-servers/).

Run `opencode mcp list` from the directory containing `opencode.json` and check that `toolkit` is connected. Start OpenCode in that directory and select a model that supports tool calls. The hosted model used here was `opencode/deepseek-v4.1-flash-free`. Use `opencode models` to check availability.

## Estimate and render

Ask OpenCode to call `toolkit_video_estimate` with a [video job spec](agent-video-jobs.md#the-spec). Ask it to show the tool's `estimate_usd` and `host_missing` fields. The estimate works without media provider keys. To render, configure the keys called for by the spec, such as `ELEVENLABS_API_KEY` for ElevenLabs narration and Scribe captions.

Save the spec as `spec.json` and run these commands from the toolkit checkout. Replace `JOB_ID` with the ID from `submit` and choose a cap that covers your estimate.

```bash
uv run --locked tools/video_job.py estimate --spec spec.json --json
uv run --locked tools/video_job.py submit --spec spec.json --max-usd 0.15 --json
uv run --locked tools/video_job.py status JOB_ID --json
uv run --locked tools/video_job.py result JOB_ID --json
```

You can also ask OpenCode to use `toolkit_video_status` and `toolkit_video_result` with that same ID. The result gives the local MP4 path after the job finishes. See the [video job guide](agent-video-jobs.md) for job recovery, supported providers and other production routes.
