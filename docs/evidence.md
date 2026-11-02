# Check local evidence for script claims

`evidence.py` checks whether exact quotations and dataset values used in a script
still match the source files. It reports changed sources, uncovered script text,
missing quotes and wrong values before recording or rendering. `qc.py` checks
media and timing; it does not check these source references.

```bash
uv run tools/evidence.py lock --spec evidence.json --output locked-evidence.json
uv run tools/evidence.py check --spec locked-evidence.json --strict
```

A minimal `evidence.json` next to `script.md` and `source.md`:

```json
{
  "version": 1,
  "script": "script.md",
  "sources": {"release": {"path": "source.md"}},
  "claims": [{
    "id": "formats",
    "text": "The release includes three export formats.",
    "source": "release",
    "quote": "The release includes three export formats."
  }]
}
```

`lock` records hashes and keeps references valid when the output is in another
directory. Sources are local UTF-8 TXT, Markdown, CSV or JSON. Cell claims use
`selector: {"row": 0, "column": "value"}` for CSV or
`selector: {"pointer": "/metrics/users"}` for JSON, plus an `expected` value.
Numbers are compared as decimals and must appear in the claim text.

Instead of `script`, use `project` pointing to an existing JSON file with
`scenes`. This reads scene IDs and selected text fields, without migrating or
validating a new project schema. Default fields are `title`, `narration` and
`onScreenText`. Add `sceneId` to each claim. Scene times are included in findings.

A matching quote or value establishes its location, not whether the claim is
true. Paraphrases and numeric interpretations require human review. `--strict`
returns failure for warnings too. `--output report.json` saves a new report;
existing files are never overwritten.

This extracts the existing feature from #143 and removes its unnecessary
manifest dependency. The original comparison considered RAGChecker and
LlamaIndex; this command needs neither an LLM judge nor a retrieval service.
There are no package additions, hosted calls or model downloads. Code is MIT.

```bash
uv run python -m unittest discover -s tests/python -p test_evidence.py -v
```
