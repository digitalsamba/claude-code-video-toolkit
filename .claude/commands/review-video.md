---
description: Open a local visual storyboard to review scenes, voice previews, quality findings and approvals
---

# Review a video

1. Resolve the requested project and read docs/review.md.
2. Validate project.json, then start `uv run tools/review.py PROJECT` in a
   persistent terminal. Read the actual printed localhost URL.
3. Present the URL so the user can select scenes, preview referenced media,
   leave timestamped notes and approve individual versions. Keep the server
   running while the review is active.
4. Read `.video-toolkit/review.json` when acting on notes. Approval fingerprints
   refer to the reviewed content; changes make old approvals stale. Never
   manufacture approvals or mark unseen scenes approved.
5. For visual edits, use the existing /design or /scene-review workflow and
   Remotion Studio. Refresh the storyboard after changes. Stop the server when
   the user finishes the review.
