# Local template and provider contracts

Use local manifests to make extensions discoverable and testable. This kit
does not install third-party code or create hosted accounts. Existing built-in
tools keep their current CLIs; adapters provide a stable boundary for new ones.

```bash
uv run tools/extensions.py list
uv run tools/extensions.py validate examples/providers/local-card/adapter.json
uv run tools/extensions.py scaffold --name my-template --output scratch/my-template
```

Templates declare version, name, engine, entry, formats, and a Remotion
composition ID when applicable in `template-manifest.json`. The validator checks
the entry actually exists and stays inside the template. The scaffold produces
a runnable Python/FFmpeg template; run its build.py for an offline video.

Providers declare a name, capabilities, required environment variables, timeout
and an argv command in adapter.json. The command receives `{input}` and
`{output}` file paths; `{python}` and `{adapter}` are also available. No shell
interpolation is performed. Read request JSON with capability, prompt and
parameters. Write the requested file and emit one stdout JSON result:
`{"version": 1, "output": "/absolute/requested/path"}`. Send logs to stderr.

```bash
uv run tools/provider.py --adapter examples/providers/local-card/adapter.json --input examples/providers/local-card/request.json --project projects/demo --output public/title.svg
uv run tools/provider.py --adapter examples/providers/local-card/adapter.json --input examples/providers/local-card/request.json --project projects/demo --output public/title.svg --run
```

Preview returns readiness and missing environment variable names without
executing anything. `--run` executes the trusted local adapter, checks its
output contract and records the asset bytes/provenance in project history.
Existing outputs are protected; use new filenames for another take. The
included SVG provider runs offline with only Python. A hosted adapter should
validate its own parameters, report useful failures, and document costs/model
licenses. Keep it in your own repository according to CONTRIBUTING.md's hosted
integration policy.

Adapter processes run with your local environment and permissions. The contract
is an interoperability boundary, not a sandbox for untrusted extension code.
Add the provider command to a production task to obtain checkpointing, content
invalidation and budget planning. Include the adapter source and parameters in
that task's declared inputs.
