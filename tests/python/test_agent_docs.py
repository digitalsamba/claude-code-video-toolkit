"""The agent skill (skills/video-toolkit/SKILL.md) keeps up with the registry.

Claude Code reads CLAUDE.md and .claude/skills/; every other agent runtime (OpenClaw, Codex, Hermes, longarm,
MCP clients) reads skills/video-toolkit/SKILL.md. New tools, templates, components and skills reached CLAUDE.md
and missed the skill for a week (v0.25.0), so this fails when a registry entry is neither mentioned in the skill
nor marked as deliberately left out with a reason:

    "agentSkip": "why an agent outside Claude Code doesn't need this"

Stdlib only, no keys or network:

    python3 -m unittest tests/python/test_agent_docs.py -v
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "_internal" / "toolkit-registry.json"
SKILL = ROOT / "skills" / "video-toolkit" / "SKILL.md"

# Registry sections an agent outside Claude Code can use. Commands are Claude Code slash commands and brands
# and examples are listed from the registry at run time (the skill says how), so they are not checked here.
SECTIONS = ("tools", "templates", "components", "skills", "designSources")


def names_for(section: str, key: str, entry: dict) -> list[str]:
    """Any of these appearing in the skill counts as a mention."""
    names = [key]
    path = (entry.get("path") or "").rstrip("/")
    if path:
        names.append(Path(path).name)
    if section == "tools":
        names.append(f"{key}.py")
    return names


def mentioned(text: str, names: list[str]) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text) for n in names)


class AgentSkillCoversRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.skill = SKILL.read_text(encoding="utf-8")

    def entries(self):
        for section in SECTIONS:
            for key, entry in self.registry.get(section, {}).items():
                if key.startswith("_") or not isinstance(entry, dict):
                    continue
                if entry.get("status") in ("removed", "deprecated"):
                    continue
                yield section, key, entry

    def test_every_entry_is_in_the_skill_or_skipped_with_a_reason(self) -> None:
        missing = []
        for section, key, entry in self.entries():
            skip = entry.get("agentSkip")
            if skip is not None:
                self.assertTrue(isinstance(skip, str) and skip.strip(),
                                f"{section}.{key}: agentSkip needs a reason, not {skip!r}")
                continue
            if not mentioned(self.skill, names_for(section, key, entry)):
                missing.append(f"{section}.{key}")
        self.assertEqual(missing, [], "\n\nNot in skills/video-toolkit/SKILL.md: " + ", ".join(missing) +
                         "\nMention each where an agent would look for it, or add "
                         "\"agentSkip\": \"<reason>\" to its registry entry.")

    def test_skips_are_not_also_documented(self) -> None:
        # A skipped entry the skill now covers should drop its agentSkip, or the list of exceptions rots too.
        both = [f"{s}.{k}" for s, k, e in self.entries()
                if e.get("agentSkip") and mentioned(self.skill, [k, f"{k}.py"] if s == "tools" else [k])]
        self.assertEqual(both, [], "Covered by the skill but still marked agentSkip: " + ", ".join(both))

    def test_skill_is_dated_in_the_registry(self) -> None:
        entry = self.registry["skills"]["video-toolkit"]
        m = re.search(r"Defaults that are current \((\d{4}-\d{2}-\d{2})\)", self.skill)
        self.assertIsNotNone(m, "SKILL.md lost its 'Defaults that are current (YYYY-MM-DD)' heading")
        self.assertEqual(m.group(1), entry.get("updated"),
                         "Bump the date in the skill's 'Defaults that are current' heading and the registry's "
                         "skills.video-toolkit.updated together")


if __name__ == "__main__":
    unittest.main()
