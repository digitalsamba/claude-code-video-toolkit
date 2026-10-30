# Visual storyboard review

Start the local review page with a project directory or project.json:

```bash
uv run tools/review.py projects/my-video
```

Open the printed localhost URL. Select a scene to see its referenced picture or
video, narration, duration, voice preview and missing-asset warnings. A declared
`render.file` is seeked to the scene start. An optional public-relative
`thumbnailFile` supplies a storyboard thumbnail. Scene audio comes from
`audio.file` or `audioFile`, relative to public/.

Leave a note or approve a scene. Notes made while reviewing video include the
current playback time. Approvals identify the exact scene and asset bytes you
reviewed. A later edit marks the approval stale; notes remain available. Shared
project settings, source/config changes and final render changes invalidate
all affected approvals conservatively. Refresh the page after external edits.

The page reads the latest build QC report from `.video-toolkit/reports/`, or
`out/qc.json`. Findings with a numeric timestamp offer a playback shortcut.
This is a review screen; scene editing and rendering remain in the toolkit and
Remotion Studio. A missing preview is displayed explicitly.

Comments and approvals are saved atomically in `.video-toolkit/review.json`.
The server binds to 127.0.0.1 and serves only referenced media, not arbitrary
project files. It does not send review content to a hosted service. Stop it
with Ctrl+C. Do not expose the local server as a public collaboration site.

Verification:

```bash
uv run python -m unittest discover -s tests -p test_review.py -v
node --check tools/review_ui/review.js
```
