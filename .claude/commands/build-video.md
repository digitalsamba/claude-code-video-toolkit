---
description: Preview, execute or resume a production build with selective regeneration and cost limits
---

# Build a video

1. Find the requested project's project.json and read docs/build.md. Validate
   existing manifests with `uv run tools/project.py validate PROJECT --json`.
2. Run `uv run tools/build.py --project PROJECT --json` to inspect the task
   graph, reused outputs and cost forecast. It does not execute commands.
3. Use a saved production block or explicit `--plan` for custom templates.
   For language/campaign versions, read docs/variants.md and preview its generated
   plan. For estimate limits/profiles, read docs/budgets.md. Never silently treat
   unknown hosted costs as free.
4. When the user's request authorizes execution, run the same command with
   `--run` and any requested `--budget-usd`/`--profile`. Keep status updates tied
   to actual task output. An ordinary retry resumes checkpoints; use
   `--force-task ID` for a deliberate retake or remote model change.
5. Inspect the returned success/completed/resumed/failed fields and the actual
   final artifact. A failed QC/task is incomplete work. Report its diagnostic
   and fix the cause before continuing.

Do not append publishing tasks. Publishing remains a separate user action.
Do not delete checkpoint state to hide failures or force a clean run.
