/**
 * Terminal for Synthetic Demos
 *
 * A terminal window that types commands and prints output, each line on a narration cue,
 * the way Stage drives a rebuilt browser screen. Nothing is screen-recorded: the session is
 * data, so a re-timed voice moves every line with it (look frames up with `cue()`).
 *
 * Lines stack from the top; once they pass the window height the view scrolls so the newest
 * line stays in sight. Each line is one row (no wrapping) unless it says `rows`, so keep
 * output short or split it. A command still typing when the next line arrives is cut off there, so
 * place the output after `typedUntil(line)`. A last `{ at, command: '' }` leaves a blinking prompt. The window is marked `data-qc="product"`, so `qc.py --layout`
 * reports its text separately and doesn't treat a cropped line as an overflow.
 *
 * @example
 * ```tsx
 * import brand from '../../../brands/video-toolkit/brand.json';
 * import { Terminal, terminalTheme } from '../../../lib/components';
 *
 * <Terminal
 *   title="projects/my-video"
 *   theme={terminalTheme(brand)}
 *   lines={[
 *     { at: 6, command: '/design' },
 *     { at: cue('you pick'), text: 'Which design source should set the direction?' },
 *     { at: cue('you pick'), menu: ['Ours', 'taste-skill', 'frontend-design'],
 *       notes: ['nothing to install', 'recommended', 'installed'],
 *       select: [{ at: cue('design skill'), index: 1 }] },
 *     { at: cue('one page'), text: '✓ Wrote design.md', kind: 'ok' },
 *   ]}
 * />
 * ```
 */

import React from 'react';
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

export type TerminalTheme = {
  bg: string;
  /** Title bar */
  bar: string;
  ink: string;
  dim: string;
  /** Prompt, caret and the selected menu row */
  accent: string;
  ok: string;
  err: string;
  font: string;
};

export type TerminalKind = 'out' | 'dim' | 'ok' | 'err' | 'accent';

type LineBase = {
  /** Frame (in the scene) the line appears on: a `cue()` result */
  at: number;
  /** Rows the line takes, for scrolling (default 1; a menu counts its options) */
  rows?: number;
  /** Left indent in characters */
  indent?: number;
};

export type TerminalLine =
  | (LineBase & {
      /** Typed after the prompt, one character per `perChar` frames */
      command: string;
      perChar?: number;
      /** Overrides the terminal's prompt for this line */
      prompt?: string;
    })
  | (LineBase & { text: React.ReactNode; kind?: TerminalKind })
  | (LineBase & {
      menu: string[];
      /** A dimmer note after each option */
      notes?: string[];
      /** Selected row: an index from `at`, or moves at frames (before the first move, row 0) */
      select?: number | { at: number; index: number }[];
    });

export type TerminalProps = {
  lines: TerminalLine[];
  theme: TerminalTheme;
  /** Shown in the title bar; also the default first dim line when `cwd` is true */
  title?: string;
  /** Print the title as the first line, like a shell's working directory */
  cwd?: boolean;
  prompt?: string;
  width?: number;
  height?: number;
  fontSize?: number;
  lineHeight?: number;
  /** Title bar with window buttons (default true) */
  chrome?: boolean;
  radius?: number;
  style?: React.CSSProperties;
};

/** Theme from a brand.json: its `colors.terminal` block, else derived from the brand's dark colours. */
export const terminalTheme = (brand: {
  colors: Record<string, unknown> & { terminal?: Partial<TerminalTheme> };
  fonts?: { mono?: string };
}): TerminalTheme => {
  const c = brand.colors as Record<string, string>;
  const t = brand.colors.terminal ?? {};
  return {
    bg: t.bg ?? c.navyDark ?? c.bgDark ?? '#1b1b1f',
    bar: t.bar ?? c.navy ?? '#26262b',
    ink: t.ink ?? '#ffffff',
    dim: t.dim ?? '#9aa3bd',
    accent: t.accent ?? c.primaryLight ?? c.primary ?? '#93b4f2',
    ok: t.ok ?? '#7ccf8a',
    err: t.err ?? '#f08a7e',
    font: t.font ?? brand.fonts?.mono ?? MONO,
  };
};

const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

// a line is drawn from its own frame on (spring frame 0 is 0, so count that frame as the first step)
const rise = (frame: number, at: number, fps: number) =>
  frame < at ? 0 : Math.min(1, spring({ frame: frame - at + 1, fps, config: { damping: 200 }, durationInFrames: 10 }));

/** Frame a command line finishes typing: put the output after it (`{ at: typedUntil(cmd) + 6, ... }`). */
export const typedUntil = (line: { at: number; command: string; perChar?: number }) =>
  line.at + Math.max(0, line.command.length - 1) * (line.perChar ?? 3);

const rowsOf = (l: TerminalLine) => l.rows ?? ('menu' in l ? l.menu.length : 1);

const selectedAt = (select: number | { at: number; index: number }[] | undefined, frame: number) => {
  if (select === undefined) return 0;
  if (typeof select === 'number') return select;
  let index = 0;
  for (const s of select) if (frame >= s.at) index = s.index;
  return index;
};

export const Terminal: React.FC<TerminalProps> = ({
  lines,
  theme,
  title,
  cwd = false,
  prompt = '>',
  width = 1100,
  height = 620,
  fontSize = 28,
  lineHeight = 1.55,
  chrome = true,
  radius = 12,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const row = fontSize * lineHeight;
  const bar = chrome ? Math.round(fontSize * 1.9) : 0;
  const padY = Math.round(fontSize * 1.1);
  const padX = Math.round(fontSize * 1.4);
  const capacity = Math.max(1, Math.floor((height - bar - padY * 2) / row));

  const all: TerminalLine[] = cwd && title ? [{ at: -1, text: title, kind: 'dim' }, ...lines] : lines;
  const sorted = [...all].sort((a, b) => a.at - b.at);

  // the caret sits on the newest command until the next line arrives
  const shown = sorted.filter((l) => frame >= l.at);
  const last = shown[shown.length - 1];

  // scroll: rows that have arrived (eased as each one rises), less what fits
  const arrived = sorted.reduce((sum, l) => sum + rise(frame, l.at, fps) * rowsOf(l), 0);
  const scroll = Math.max(0, arrived - capacity) * row;

  const color = (kind: TerminalKind = 'out') =>
    ({ out: theme.ink, dim: theme.dim, ok: theme.ok, err: theme.err, accent: theme.accent })[kind];

  const caret = (typing: boolean) => {
    const on = typing || Math.floor(frame / (fps / 2)) % 2 === 0;
    return (
      <span
        style={{
          display: 'inline-block',
          width: '0.6em',
          height: '1.1em',
          verticalAlign: '-0.18em',
          marginLeft: '0.08em',
          background: theme.accent,
          opacity: on ? 1 : 0,
        }}
      />
    );
  };

  const render = (l: TerminalLine, i: number) => {
    const t = rise(frame, l.at, fps);
    if (t <= 0) return null;
    const pad = l.indent ? `${l.indent}ch` : undefined;
    const base: React.CSSProperties = {
      paddingLeft: pad,
      whiteSpace: 'pre',
      overflow: 'hidden',
      height: rowsOf(l) * row,
    };

    if ('command' in l) {
      const per = l.perChar ?? 3;
      const n = Math.max(0, Math.min(l.command.length, Math.floor((frame - l.at) / per) + 1));
      const typing = n < l.command.length;
      return (
        <div key={i} style={{ ...base, color: theme.ink }}>
          <span style={{ color: theme.accent }}>{l.prompt ?? prompt}</span> {l.command.slice(0, n)}
          {l === last && caret(typing)}
        </div>
      );
    }

    const enter: React.CSSProperties = { opacity: t, transform: `translateY(${(1 - t) * 8}px)` };

    if ('menu' in l) {
      const sel = selectedAt(l.select, frame);
      return (
        <div key={i} style={{ ...base, ...enter }}>
          {l.menu.map((option, k) => {
            const on = k === sel;
            return (
              <div key={k} style={{ height: row, color: on ? theme.accent : theme.ink, fontWeight: on ? 700 : 400 }}>
                {on ? '›' : ' '} {k + 1}. {option}
                {l.notes?.[k] && <span style={{ color: on ? theme.accent : theme.dim }}>  {l.notes[k]}</span>}
              </div>
            );
          })}
        </div>
      );
    }

    return (
      <div key={i} style={{ ...base, ...enter, color: color(l.kind) }}>
        {l.text}
      </div>
    );
  };

  return (
    <div
      data-qc="product"
      style={{
        width,
        height,
        background: theme.bg,
        borderRadius: radius,
        overflow: 'hidden',
        fontFamily: theme.font,
        fontSize,
        lineHeight,
        boxSizing: 'border-box',
        ...style,
      }}
    >
      {chrome && (
        <div
          style={{
            height: bar,
            background: theme.bar,
            display: 'flex',
            alignItems: 'center',
            padding: `0 ${Math.round(fontSize * 0.7)}px`,
            gap: Math.round(fontSize * 0.3),
            position: 'relative',
          }}
        >
          {[0, 1, 2].map((k) => (
            <div
              key={k}
              style={{ width: fontSize * 0.42, height: fontSize * 0.42, borderRadius: '50%', background: theme.dim, opacity: 0.45 }}
            />
          ))}
          {title && (
            <div
              style={{
                position: 'absolute',
                left: 0,
                right: 0,
                textAlign: 'center',
                color: theme.dim,
                fontSize: fontSize * 0.72,
                lineHeight: `${bar}px`,
              }}
            >
              {title}
            </div>
          )}
        </div>
      )}
      <div style={{ padding: `${padY}px ${padX}px` }}>
        {/* clip to whole rows, so the padding never shows a line that has scrolled off */}
        <div style={{ height: capacity * row, overflow: 'hidden' }}>
          <div style={{ transform: `translateY(${-scroll}px)` }}>{sorted.map(render)}</div>
        </div>
      </div>
    </div>
  );
};
