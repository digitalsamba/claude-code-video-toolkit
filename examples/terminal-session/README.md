# terminal-session

An agent CLI session drawn by `lib/components/TerminalSession`: someone asks for Modal to be set up, and the
agent checks the CLI, asks permission to log in, asks which apps to deploy, deploys them with a spinner running,
edits `.env` and sums up. Every line kind is used once, with a zoom onto two lines and a highlight on a third.

| Composition | Size | Skin |
|---|---|---|
| `ClaudeCode` | 1920x1080 | `claude-code` |
| `Codex` | 1920x1080 | `codex` |
| `Shell` | 1920x1080 | `shell` |
| `Vertical` | 1080x1920 | `claude-code`, larger type |

```bash
cp -r examples/terminal-session projects/my-session
cd projects/my-session && npm install
npm run studio
npm run render        # out/claude-code.mp4; render:all for the four
```

The frames are fixed numbers in `src/Session.tsx` (the `B` table). In a narrated video, take them from `cue()` so
the session follows the voice. The outputs shown (version numbers, URLs) are written for the example, not
captured.

`public/fonts/` holds JetBrains Mono (SIL Open Font License, `OFL.txt`), loaded with `loadMonoFont()` so the
render is the same on macOS, Linux and Windows. Symbols the font doesn't have (`⎿`, `☐`, the spinner) fall back
to a system font; each is drawn in a cell one character wide so the columns stay in line.
