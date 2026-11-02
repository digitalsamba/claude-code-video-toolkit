# Local asset version history

Regenerating an image or audio file can replace a take worth keeping. This command records immutable copies with content hashes and searchable metadata, and restores selected versions without requiring project.json or a build runner.

Uses Python standard-library hashing and filesystem publication. No dependencies, hosted services, models or API cost. Toolkit code is MIT. Stored asset copies consume local disk space and are not automatically pruned.

## Usage

```bash
uv run tools/asset_history.py --project projects/my-video --json record public/title.png --kind image
uv run tools/asset_history.py --project projects/my-video --json list --query ocean
```

Records live under `.video-toolkit/assets/` inside the selected project. Recording requires an existing asset within that project. Include optional `--scene`, `--kind` and `--metadata` to make records searchable. `restore VERSION_ID` verifies content hashes and refuses to replace existing content unless `--force` is explicit. Record the current file before using `--force` if you want to retain it. Use `restore VERSION_ID --output public/alternate.png` to keep both files.

History is opt-in. It neither changes the project schema nor hooks into generation or `/video`. Copies are local and can be large. This is version recovery, not a remote backup service.

## Implementation reference

This extracts the existing content-addressed asset store from #142. It is a manual local file utility, with no automatic build hooks or schema migration.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_asset_history.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
