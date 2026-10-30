# Validated project manifests

`project.json` records a video's planned scenes and workflow metadata. Validate it
before an agent, build tool or editor updates it:

```bash
uv run tools/project.py validate projects/my-video
uv run tools/project.py validate projects/my-video/project.json --json
```

These commands are read-only. They do not install dependencies, render media,
contact providers or require assets to exist. Errors identify the field, such as
`/scenes/1/durationSeconds: must be greater than 0`. Exit status is `0` for a valid
manifest and `1` for file or validation errors. `--json` writes one result object
to stdout, including an `errors` array when validation fails.

## Version 1 and existing projects

The canonical schema is [`lib/project/schema.json`](../lib/project/schema.json).
Only `name`, `template` and `scenes` are required, so a project can start with:

```json
{
  "schemaVersion": 1,
  "name": "Product walkthrough",
  "template": "product-demo",
  "engine": "remotion",
  "scenes": []
}
```

Existing files without `schemaVersion` load as version 1. The loader preserves
unknown fields at every level, including template settings, design notes and
publishing metadata. It returns a copy and does not change the file. There is no
automatic rewriting of template configs, scripts or media.

Preview migration, then explicitly write the version field if desired:

```bash
uv run tools/project.py migrate projects/my-video --json
uv run tools/project.py migrate projects/my-video --write --json
```

The first command is a dry run. `--write` validates first and atomically replaces
the JSON file. It preserves field values but reformats JSON with two-space
indentation. Running it again produces the same file. Unsupported schema versions
fail instead of being silently downgraded.

## Scenes, audio and output settings

Scene IDs must be unique, nonempty strings. Scene and visual types are open so
templates can define `chapter`, `ticket`, `synthetic-ui`, `ideogram` or other kinds.
Each scene needs `id` and `type`. Duration, visuals and narration can be added
during planning:

```json
{
  "id": "schedule",
  "type": "demo",
  "durationSeconds": 8.5,
  "visual": {"type": "synthetic-ui", "instructions": "Show the scheduling dialog"},
  "narration": "Choose a time and invite your team.",
  "scriptFile": "public/audio/scenes/schedule.txt",
  "audio": {"file": "audio/scenes/schedule.mp3", "status": "present"}
}
```

`audioFile` and `captionsFile` are also supported per scene. Audio, visual assets
and caption paths follow existing templates and are relative to `public/`.
`scriptFile`, `render.entry`, `render.file`, top-level `output`, publishing files
and production task paths are relative to the project directory. Python templates
can resolve their audio relative to their own asset root instead; the loader never
guesses which files a template uses.

Optional project settings include:

- `phase`, including the `review` phase, plus `brand`, timestamps and sessions.
- `audio.voiceover`, `audio.music` and `audio.sfx`, including `per_scene` voiceover mode.
- `engine`, `composition` and `output`, retained for existing project layouts.
- `format`, as a string such as `"9:16"`, or an object with `width`, `height`, `fps`
  and `aspectRatio`.
- `render`, with `engine`, `entry`, `composition`, `file` and optional dimensions/timing.
- `publish`, with title, description, tags, thumbnail, captions, privacy and upload results.

Present durations and frame rates must be positive finite numbers. Start times,
estimates and costs must be nonnegative. Timestamps use ISO date-time strings with
a timezone; session dates use `YYYY-MM-DD`. The validator does not equate estimates
with rendered duration, require reviewed scenes, or measure audio. Continue using
`tools/qc.py` for media checks.

Local paths cannot be absolute, contain a drive or URL, or include `..` components.
Both slash styles are accepted. Legacy visual asset patterns such as `images/*.png`
are preserved. New production outputs must name files without glob patterns.

## Python API

Tools in `tools/` can import the module directly:

```python
from project_manifest import load_project, save_project, validate_project

project = load_project("projects/my-video")  # Directory or manifest filename.
project["phase"] = "review"
save_project("projects/my-video", project)
```

`validate_project(data)` returns a validated deep copy and supplies
`schemaVersion: 1` if absent. `ProjectValidationError.errors` is a list of
`{"path": "/scenes/0/id", "message": "..."}` objects using JSON Pointer paths.
File IO errors propagate as `OSError`; the CLI reports them without a traceback.

`resolve_project_path(root, value, allow_root=False)` validates and resolves a
reference, rejecting symlinks that escape the supplied root. Pass the project
directory for render/script paths, or `project_dir / "public"` for public assets.
Use this helper at the point of file access, because lexical schema validation
cannot know the filesystem root or whether symlinks have changed.

`save_project` validates before writing and replaces the file on the same volume.
It cleans up temporary files on failure. This prevents partial writes, but does
not merge simultaneous edits. Callers must coordinate writers or add an explicit
revision check before exposing concurrent editing.

The standard-library validator implements exactly the JSON Schema keywords used
by the canonical file. Its `project-*` formats add local path rules. A test rejects
new validation keywords until the interpreter supports them. External editors can
use the schema for types and completions; the Python API also checks duplicate
scene/task IDs and production dependency references/cycles.

## Production extension

An optional task graph gives the build tool explicit argv, inputs and outputs:

```json
{
  "production": {
    "version": 1,
    "budgetUsd": 1,
    "tasks": [
      {
        "id": "qc",
        "command": ["{python}", "{toolkit}/tools/qc.py", "--project", "{project}", "--json"],
        "cwd": ".",
        "inputs": ["public/audio/scenes/*.mp3"],
        "outputs": [],
        "stdoutFile": "out/qc.json",
        "timeoutSeconds": 120,
        "estimatedCostUsd": 0
      }
    ]
  }
}
```

Tasks require unique slug-like IDs, a nonempty argv array and an `outputs` array.
`dependsOn` lists existing task IDs and must form an acyclic graph. `inputs` may
contain globs, `cwd` may be `"."`, and `outputs`/`stdoutFile` must be local file
paths. Empty outputs are allowed for commands that must run every time. The build
tool defines placeholder expansion, execution, output caching and cost enforcement;
manifest validation never executes these commands.

## Verification

```bash
uv run python -m unittest discover -s tests -p "test_project_manifest.py" -v
```

Tests cover legacy data preservation, custom scenes, numbers and dates, path
containment, production graphs, atomic failures and the actual CLI from outside
the toolkit directory. They run without API keys or network access.
