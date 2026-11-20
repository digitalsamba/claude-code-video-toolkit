# Shared components

Reusable video components in `lib/components/`. See registry `components` section for the full list with descriptions. Import in templates via:

```tsx
import { AnimatedBackground, SlideTransition, Label } from '../../../../lib/components';
```

**Synthetic UI demos:** `Stage` (browser window, camera zoom/pan keyframes, cursor layer) and `Cursor` (eased
keyframe path, click ripples) drive product screens rebuilt as static React from the app's source, so nothing is
screen-recorded. Coordinates are canvas pixels (1440x810 by default) and every click is a frame number, looked up
from the narration with `cue('save')` (`lib/components/cues.ts`) so the scene follows the voice when it is
re-timed. For product walkthroughs and release-notes videos, prefer this to a Playwright recording when the app's
source is readable. The method and its gotchas are in `.claude/skills/synthetic-ui-demo/`. The reference builds
are `examples/ds-free-schedule-demo` and `examples/ds-release-649`, and their screen kits stay in the examples.

**Terminal demos:** `Terminal` (`lib/components/Terminal.tsx`) is the same idea for a CLI: commands type, output
prints (`kind: 'ok' | 'err' | 'dim'`) and menus select, each on a `cue()` frame, and the window scrolls once full.
Theme it with `terminalTheme(brand)`; `brands/video-toolkit` carries a terminal palette. Start a line's output after
`typedUntil(command)`, or the command is cut off mid-typing.

**Agent sessions:** `TerminalSession` (`lib/components/TerminalSession.tsx`) draws a session in an agent CLI rather than a
shell: your prompt typed into the input box and sent, the agent's prose streaming by word, tool calls with results,
a spinner, a question picker or permission prompt where the input box was, todos that tick, a diff. Pick the
harness with `skin` (`claude-code`, `codex`, `shell`, or an object for Kiro or another CLI: glyphs and strings
only, no logos). Colour with markup (`{ok|✓}`, `` `code` ``, `**bold**`) rather than whole lines; zoom onto a line
with `focus` and an `id`. Load the bundled JetBrains Mono with `loadMonoFont()` so the render is the same on every
OS, and put its family first in the theme's `font`. Reference: `examples/terminal-session`.

**Data charts:** `LineChart` and `BarChart` (`lib/components/DataChart.tsx`) draw charts as SVG in the brand
(`chartTheme(brand.colors, brand.fonts, 'dark' | 'light')`) instead of embedding a Matplotlib MP4. The line draws to
`keys` (`{ frame, x }` pairs, frames from `cue()`), so each stretch of data is drawn while the narration talks about
it; annotation bands and bar highlights take cue frames too, and `lineHead()` gives the same position for a counter.
`series` puts several lines on one chart with a legend; `Counter` counts a number up from a cue; `SplitBar` shows shares of a
whole (who did the work), and still labels a zero share so the narration can say it.
Prefer this inside Remotion videos; keep the `data-viz-chart` Matplotlib pattern for pure-Python builds. Reference:
`examples/data-chart-stars`.
