"""capture/*.txt -> src/captured.json: each run's command and output lines, colours kept as kinds.

The scenes pick lines from here with pick() (src/captured.ts), which throws at bundle time if a line
isn't in the capture, so nothing on screen can be typed in by hand. Rerun after any capture changes.
"""
import json, pathlib, re
here = pathlib.Path(__file__).resolve().parent.parent
SGR = re.compile(r"\x1b\[([0-9;]*)m")
KIND = {"92": "ok", "32": "ok", "91": "err", "31": "err", "90": "dim", "93": "accent", "33": "accent"}
out = {}
for f in sorted((here / "capture").glob("*.txt")):
    raw = f.read_text().splitlines()
    command = raw[0][2:] if raw and raw[0].startswith("$ ") else ""
    lines = []
    for line in raw[1:]:
        if line.startswith("# elapsed"):
            continue
        codes = SGR.findall(line)
        kind = next((KIND[c] for c in codes if c in KIND), "out")
        lines.append({"text": SGR.sub("", line).rstrip(), "kind": kind})
    out[f.stem] = {"command": command, "lines": lines}
(here / "src/captured.json").write_text(json.dumps(out, indent=1))
print({k: len(v["lines"]) for k, v in out.items()})
