# Brief: how a brief becomes a video

Make a short video that explains how this toolkit turns a written brief into a finished video. Work in this
repository (the toolkit root) as you would for a real video, using its docs, tools and skills.

## The video

- **Audience:** a developer who has just found the repo and wonders what happens between "write a brief" and
  "get an MP4". They use a coding agent but have never made a video with one.
- **Length and format:** 45 to 60 seconds, 16:9, 1920x1080, 30 fps.
- **The idea:** a diagram of the stages a video goes through in this toolkit, built step by step while the
  narration names each stage. What the stages are is for you to find in the repo; say only what is true of
  it today.
- **Manim:** at least the diagram scene is drawn with Manim (`tools/manim_render.py`, `lib/manim`). Read
  `docs/manim.md` before writing a scene. LaTeX is not installed, so no formulas.
- **Everything else** (other scenes, template or plain composition, music or none) is your call.
- **Brand and voice:** the `video-toolkit` brand and its default voice.
- **Public copy:** everything in it is public, so the narration and on-screen words follow the "Public copy"
  section of `CLAUDE.md` and pass `tools/copy_lint.py`.

## Rules for this run

- Work in `projects/agent-test-<your agent name>/` and change nothing outside it.
- Spend at most $1 on paid services in total. Say what you spent.
- Don't upload, publish, commit, push or open a pull request.
- If a toolkit tool or doc gets in your way, don't fix the toolkit: work around it and write it down.
- Ask the person only when you are blocked or a choice is really theirs.

## Hand back

1. The final MP4 in your project folder, checked with the toolkit's QC.
2. `NOTES.md` in your project folder: the route you took and why, each command that failed and what you did
   about it, what you spent, how long it took, and anything in the toolkit that was missing, wrong or confusing.
