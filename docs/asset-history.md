# Asset versions and provenance

Save generated images, audio, and video before replacing them. The history stores
the actual bytes plus the prompt, model version, voice, seed, source, or other
metadata you supply. Search those records and restore any version later.

Run these commands from the toolkit root:

```bash
uv run tools/asset_history.py --project projects/my-video record public/images/title.png --kind image --scene intro --metadata scratch/title-metadata.json
uv run tools/asset_history.py --project projects/my-video list --query "blue sea"
uv run tools/asset_history.py --project projects/my-video --json inspect VERSION_ID
uv run tools/asset_history.py --project projects/my-video restore VERSION_ID --output public/images/title-alternative.png
```

`--metadata` reads a JSON object, for example:

```json
{"prompt": "Blue sea at dawn", "model": "flux2-klein-4b", "seed": 42, "source": "original generation"}
```

Asset paths and restore outputs are relative to the project. An absolute path
must still be inside that project. Restoring to an existing file requires
`--force`. Omit `--output` to restore to the original recorded path.

Snapshots live under `.video-toolkit/assets/objects/`, named by SHA-256. Each
version has its own JSON record, so concurrent generators do not overwrite a
shared index. Identical bytes share one snapshot while retaining separate
metadata. Restores verify the snapshot checksum before writing. Keep this
directory with the project when moving or backing up its history.

Recording is explicit: running a generator alone does not automatically create
a version. Python integrations can import `record_asset(project_root, asset,
metadata=..., kind=..., scene_id=...)` from `tools/asset_history.py`. Record an
existing output before regeneration, then record its replacement after success.

`list` and `inspect` do not create files. `--json` emits JSON for every command,
including errors. The history records supplied model settings; a seed does not
guarantee that a hosted model can reproduce the same output later.

Verification:

```bash
uv run python -m unittest discover -s tests -p test_asset_history.py -v
```
