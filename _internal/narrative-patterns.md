# Narrative Patterns for Sprint Reviews

Patterns for structuring sprint review videos beyond the default "here's what we shipped" format.

The rest of the toolkit docs cover mechanics — timing, transitions, components. This file
covers narrative: what order to tell things in, and how to frame them.

---

## Pattern: Discovery Story

**When to use:** The sprint involved solving problems that led to a strategic shift or new direction. The journey matters more than the deliverables.

**Structure:**

1. **Context** — Set the landscape. What existed before this sprint? What pressures were building?
2. **The First Fix** — A pragmatic solution that works. Celebrate the win, but acknowledge it's a bridge.
3. **The Workaround** — A hack that solves an immediate pain point but feels wrong. The "hack on a hack."
4. **The Turning Point** — The moment the team realises the root cause. Frame this as a context slide with clear bullet points on why the current approach is a dead end.
5. **The Answer** — The new direction. Show it working. Emphasise it's real, not a prototype (tests, builds, sessions).
6. **Decisions/Roadmap** — Frame decisions around the strategic shift. Acknowledge bridges as temporary. Show the phaseout plan.

**Key principles:**
- Reorder deliverables by narrative arc, not ticket order
- Use a "turning point" context slide to create dramatic tension between the workaround and the solution
- Frame bridge solutions honestly: "it works brilliantly, but it's not the long-term answer"
- Include user reactions/feedback as visual proof of bridge solution success
- The summary should capture the shift: "We went in doing X. We came out doing Y."

**Worked example (illustrative):**

A team maintaining two aging client apps against a newer shared component.

- Context: Two legacy clients, one newer shared component, one backend — three release cadences
- Fix: Ship the shared component inside the legacy clients — quick win, users happy, far cheaper than a rewrite
- Hack: A platform-specific workaround to keep the embedded component alive in the background — it works, but it's a hack on a hack
- Turning point: The embedded approach can never be a first-class experience here — the workarounds are the evidence, not the exception
- Answer: A properly built native client — with the test count, build pipeline and real sessions that prove it isn't a prototype
- Roadmap: Phase out the embedded stopgap over the following quarters

The arc is what transfers, not the specifics: **something works → something works badly →
you understand why → you change direction.** Most sprints that felt chaotic in the moment
have this shape hiding in them.

---

## Pattern: Version Clash Title

**When to use:** Multiple codebases/apps with divergent version numbers. Instantly communicates fragmentation or platform complexity.

**How:** Use the `version` field with `>` separators to show the evolution:

```
Legacy 1.38.0 > Embedded 4.0.5 > Native 1.0.0
```

**Requires a TitleSlide change — not yet implemented.** Both `sprint-review` and
`sprint-review-v2` currently build the string unconditionally:

```tsx
const versionString = info.build
  ? `Version ${info.version} (BUILD ${info.build})`
  : `Version ${info.version}`;
```

So the example above renders as `Version Legacy 1.38.0 > …`. To use this pattern, drop the
`Version ` prefix when the string doesn't start with a digit:

```tsx
const isRawVersion = !/^\d/.test(info.version);
const versionString = isRawVersion
  ? info.version
  : info.build
    ? `Version ${info.version} (BUILD ${info.build})`
    : `Version ${info.version}`;
```

---

## Pattern: Bridge Framing

**When to use:** A deliverable solves the problem today but isn't the long-term answer.

**In narration:** "Users love it — and it saved us a huge amount of work. It's a bridge solution, not the long-term answer, but right now it's working brilliantly."

**In decisions slide:** Frame as "a bridge, not a destination" with rationale explaining the phaseout path.

**In roadmap:** Include "Phase out [bridge solution]" as an upcoming node to show the team is eyes-open about technical debt.

**Visual proof:** Overlay positive user reactions / feedback on the demo to validate the bridge is genuinely working.

**Why it's worth the honesty:** a review that admits a bridge is temporary ages well. One that
presents a stopgap as the destination has to be quietly walked back next quarter.

---

## Future Patterns (to document as they emerge)

- **Before/After** — Side-by-side comparison (bug vs fix, old UX vs new UX)
- **Escalation** — Problem gets worse across scenes before the fix
- **Convergence** — Multiple workstreams coming together into one outcome
