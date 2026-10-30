# Resumable production builds

`tools/build.py` runs declared commands in dependency order and saves a checkpoint after each task. An interrupted or failed build resumes completed tasks on its next run. The default is a read-only preview.

```powershell
uv run tools/build.py --project projects/my-video --json
uv run tools/build.py --project projects/my-video --run
```

Projects can place a `production` block in `project.json`, or use a separate plan:

```powershell
uv run tools/build.py --project projects/my-video --plan production.json --run
```

The plan path is relative to the command's working directory. File paths *inside* the plan are relative to `--project`. An example plan is at `examples/production-plan.json`; it uses Python only and makes no network calls. Copy it into an empty project and run it to try failure-free checkpointing.

## Task format

```json
{
  "version": 1,
  "tasks": [
    {
      "id": "voice-intro",
      "command": ["{python}", "{toolkit}/tools/voiceover.py", "--script", "{project}/public/audio/scenes/01_intro.txt", "--output", "{project}/public/audio/scenes/01_intro.mp3", "--provider", "qwen3", "--json"],
      "inputs": ["public/audio/scenes/01_intro.txt"],
      "outputs": ["public/audio/scenes/01_intro.mp3"],
      "timeoutSeconds": 1800
    }
  ]
}
```

- `command` is an argv array, executed without a shell. `{python}`, `{toolkit}`, and `{project}` are substituted in each argument. Shell operators are ordinary arguments. Use executable paths rather than shell built-ins or platform batch launchers.
- `dependsOn` lists task IDs. Missing IDs, duplicate IDs, and cycles fail before execution.
- `cwd` defaults to the project root. Paths for working directories and outputs must stay inside the project, including through symlinks.
- `inputs` declares project files, directories, or glob patterns. SHA-256 content hashes include directory membership and missing optional files. Changing a file rebuilds its task and every dependent task; other branches resume. Touching a file without changing its bytes does not trigger work.
- `toolkitInputs` declares files, directories, or globs under the toolkit root, for shared components or brand settings. Existing Python/JavaScript command source files are fingerprinted automatically. Declare imported helpers and other external inputs explicitly. `environmentInputs` names environment variables whose hashed values should invalidate an output; plaintext values are never stored. Provider-side model changes cannot be detected locally: use `--force-task ID` after a remote model upgrade.
- `outputs` lists nonempty files that must exist after success. `stdoutFile` optionally writes captured stdout to a project-relative file and counts as an output. A task with no outputs always reruns. Outputs cannot be shared by multiple tasks.
- `timeoutSeconds` defaults to 1800. Failed commands, timeouts, missing outputs, and interruptions stop the graph and keep earlier checkpoints.
- `estimatedCostUsd` can annotate a task; this runner does not enforce a budget or claim provider billing accuracy.

Checkpoint data lives in `.video-toolkit/build-state.json`. Writes are atomic, and an OS lock prevents simultaneous runs for one project. A process crash releases the lock. Successful outputs are content-verified before reuse. In-place tools such as timing sync record their inputs after success, so their own config update does not trigger an endless rebuild. Old checkpoints without fingerprints rebuild once. Preserve invalid checkpoint files for diagnosis; the runner reports them rather than silently discarding them. Use `--restart` to rerun all tasks, or `--force-task ID` to rerun just that task and its dependents. Use `--run --dry-run` to force a preview with no state writes or commands.

## Built-in Remotion plan

For the `storyboard` template, the built-in adapter reads narration directly
from project.json. It creates one script and generation task per scene, reuses
explicitly supplied scene audio, measures the final audio lengths, validates
referenced media, prepares render props and runs the chosen format plus QC.
Changing one scene's narration leaves the other scene scripts reusable. Projects
with no planned narration/audio pass `--allow-silent` to QC explicitly.

For sprint-review and product-demo projects using `src/config/sprint-config.ts` or `src/config/demo-config.ts`, the runner connects existing toolkit commands:

1. Generate voiceover separately for each existing `public/audio/scenes/*.txt` script.
2. Optionally generate captions with `--captions` (`whisper` by default; install `uv sync --extra whisper`).
3. Sync timing to generated audio.
4. Run pre-render QC, render via the project's installed Remotion CLI, then run final QC.

```powershell
uv run tools/build.py --project projects/my-video --builtin --provider qwen3 --captions --json
uv run tools/build.py --project projects/my-video --builtin --provider qwen3 --captions --run
```

The built-in plan reads the project's plain `remotion render ...` package script and brand. Install the project's Node dependencies first. Supply existing scene scripts and configured voice credentials; generation can incur provider charges. Previewing never calls providers. Existing QC errors stop rendering. No publishing step is added. MoviePy and custom templates can use explicit task plans with their existing build commands.

Verify locally without cloud calls:

```powershell
python -m unittest discover -s tests -p test_build.py
```
