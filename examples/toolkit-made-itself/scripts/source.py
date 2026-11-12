"""Copy the <shown> part of src/compose.tsx into src/source.json for scene 5's code panel."""
import json, pathlib
here = pathlib.Path(__file__).resolve().parent.parent
text = (here / "src/compose.tsx").read_text()
start = text.index("// <shown>\n") + len("// <shown>\n")
shown = text[start:text.index("// </shown>")].rstrip("\n").split("\n")
(here / "src/source.json").write_text(json.dumps(shown, indent=1))
print(len(shown), "lines")
