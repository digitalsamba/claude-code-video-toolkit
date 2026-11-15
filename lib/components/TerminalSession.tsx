/**
 * Terminal Session: an agent CLI conversation for synthetic demos
 *
 * `Terminal` draws a plain shell. `TerminalSession` draws a session in an agent harness the way viewers know it:
 * your prompt typed into the input box and then sent up into the transcript, the agent's prose streaming in by
 * word, tool calls with their results under them, a spinner while it works, a question picker or a permission
 * prompt in place of the input box, a todo list that ticks, a diff. Each thing happens on a frame from `cue()`, so
 * a re-voiced narration moves the session with it. Nothing is screen-recorded.
 *
 * The harness is a `skin`: `'claude-code'`, `'codex'` or `'shell'` are built in, and a skin is a plain object, so
 * another harness (Kiro, Gemini CLI, your own) is a few glyphs and strings. The skins draw each harness's shapes
 * and glyphs in the brand's colours; they carry no logos or wordmarks.
 *
 * Text is wrapped to the window (hanging indents included), and inline markup colours only what should stand out:
 * `**bold**`, `` `code` `` and `{ok|✓}` / `{err|✗}` / `{dim|...}` / `{accent|...}`. A `focus` list of camera
 * keyframes zooms onto a line by its `id`, and `highlight` lays a band behind one. The window is marked
 * `data-qc="product"` like `Terminal`.
 *
 * @example
 * ```tsx
 * import brand from '../../../brands/video-toolkit/brand.json';
 * import { TerminalSession, terminalTheme } from '../../../lib/components';
 *
 * <TerminalSession
 *   skin="claude-code"
 *   theme={terminalTheme(brand)}
 *   status={['{accent|Opus} · {code|my-video} · {dim|main}']}
 *   lines={[
 *     { at: 10, user: 'set up modal for me' },
 *     { at: cue('check'), spinner: 'Checking', until: cue('installed') },
 *     { at: cue('installed'), say: "Modal's CLI is installed. I'll log you in next." },
 *     { at: cue('log in'), tool: 'Bash', arg: 'uv run modal setup', result: ['{ok|✓} Token written to `~/.modal.toml`'],
 *       resultAt: cue('done'), id: 'token' },
 *   ]}
 *   focus={[{ at: cue('done'), id: 'token', zoom: 1.6 }, { at: cue('next'), zoom: 1 }]}
 * />
 * ```
 */

import React from 'react';
import { continueRender, delayRender, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import type { TerminalTheme } from './Terminal';

// ---------------------------------------------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------------------------------------------

export type SessionTheme = TerminalTheme & {
  /** Band behind your prompt in the transcript (default: ink at 9% over the background) */
  band?: string;
  /** Inline `code` (default: accent) */
  code?: string;
  /** Rules around the input box (default: dim at 45%) */
  rule?: string;
  /** Diff row backgrounds */
  addBg?: string;
  delBg?: string;
  /** Highlight band (default: accent at 20%) */
  mark?: string;
};

const mix = (color: string, pct: number) => `color-mix(in srgb, ${color} ${pct}%, transparent)`;

const fill = (t: SessionTheme) => ({
  ...t,
  band: t.band ?? mix(t.ink, 9),
  code: t.code ?? t.accent,
  rule: t.rule ?? mix(t.dim, 45),
  addBg: t.addBg ?? mix(t.ok, 18),
  delBg: t.delBg ?? mix(t.err, 18),
  mark: t.mark ?? mix(t.accent, 22),
});
type Theme = ReturnType<typeof fill>;

// ---------------------------------------------------------------------------------------------------------------
// Skins
// ---------------------------------------------------------------------------------------------------------------

export type SessionSkin = {
  name: string;
  /** Before your prompt, in the input box and in the transcript */
  prompt: string;
  /** Before the agent's prose and each tool call ('' for none) */
  bullet: string;
  /** Before a tool's result */
  elbow: string;
  /** A band behind your prompt once it is sent */
  userBand: boolean;
  /** Blank rows between items in the transcript */
  gap: number;
  /** Input: ruled lines above and below, a filled box, or typed straight into the transcript (a shell) */
  input: 'rules' | 'box' | 'inline';
  spinnerFrames: string[];
  /** Markup for the spinner line, from the verb and the seconds so far */
  spinnerLine: (verb: string, seconds: number) => string;
  /** Markup for a tool call's head line */
  toolHead: (name: string, arg?: string) => string;
  /** Markup for the transcript line left by an answered question */
  answered: (question: string, answer: string) => string[];
  /** The options of a permission prompt; the first is "yes" */
  permissionOptions: (command: string) => string[];
  permissionQuestion: string;
  /** Hint line under a picker ('' for none) */
  pickerHint: string;
};

export const sessionSkins: Record<'claude-code' | 'codex' | 'shell', SessionSkin> = {
  'claude-code': {
    name: 'claude-code',
    prompt: '❯',
    bullet: '●',
    elbow: '⎿',
    userBand: true,
    gap: 1,
    input: 'rules',
    spinnerFrames: ['·', '✢', '✳', '✶', '✻', '✽', '✻', '✶', '✳', '✢'],
    spinnerLine: (verb, s) => `{accent|${verb}…} {dim|(${s}s · esc to interrupt)}`,
    toolHead: (name, arg) => (arg ? `**${name}**(${arg})` : `**${name}**`),
    answered: (q, a) => ['User answered the question:', `{dim|·} ${q} {dim|→} **${a}**`],
    permissionOptions: (cmd) => [
      'Yes',
      `Yes, and don't ask again for \`${cmd.split(' ').slice(0, 2).join(' ')}\` commands`,
      'No, and tell Claude what to do differently',
    ],
    permissionQuestion: 'Do you want to proceed?',
    pickerHint: 'Enter to select · ↑/↓ to navigate · Esc to cancel',
  },
  codex: {
    name: 'codex',
    prompt: '›',
    bullet: '•',
    elbow: '└',
    userBand: true,
    gap: 1,
    input: 'box',
    spinnerFrames: ['•', '◦'],
    spinnerLine: (verb, s) => `{accent|${verb}} {dim|(${s}s • esc to interrupt)}`,
    toolHead: (name, arg) => (name === 'Bash' ? `**Ran** ${arg ?? ''}` : arg ? `**${name}** ${arg}` : `**${name}**`),
    answered: (q, a) => [`${q} {dim|→} **${a}**`],
    permissionOptions: () => ['Yes, proceed', "Yes, and don't ask again this session", 'No, and tell Codex what to do differently'],
    permissionQuestion: 'Would you like to run the following command?',
    pickerHint: 'Press enter to confirm or esc to cancel',
  },
  shell: {
    name: 'shell',
    prompt: '$',
    bullet: '',
    elbow: '',
    userBand: false,
    gap: 0,
    input: 'inline',
    spinnerFrames: ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
    spinnerLine: (verb) => `{accent|${verb}…}`,
    toolHead: (name, arg) => `{accent|$} ${arg ?? name}`,
    answered: (q, a) => [`${q} **${a}**`],
    permissionOptions: () => ['Yes', 'No'],
    permissionQuestion: 'Continue?',
    pickerHint: '',
  },
};

// ---------------------------------------------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------------------------------------------

type Base = {
  /** Frame (in the scene) the item starts: a `cue()` result */
  at: number;
  /** For `focus` and `highlight` */
  id?: string;
  /** Blank rows before this item (default: the skin's gap) */
  gap?: number;
};

export type SessionLine =
  /** Your prompt: types into the input box from `at`, then is sent to the transcript at `submit` */
  | (Base & { user: string; perChar?: number; submit?: number })
  /** The agent's prose, streamed in by word (`wps` words a second, default 9; `instant` for all at once) */
  | (Base & { say: string; wps?: number; instant?: boolean })
  /** A tool call; the result shows at `resultAt` (default `at` + 0.6 s), with `more` hidden lines noted */
  | (Base & { tool: string; arg?: string; result?: string | string[]; resultAt?: number; more?: number; failed?: boolean })
  /** The spinner over the input box from `at` until `until` */
  | (Base & { spinner: string; until: number })
  /** A todo list; each item ticks on its `done` frame */
  | (Base & { todo: { text: string; done?: number }[]; title?: string })
  /** A file edit: rows marked '+', '-' or ' ' */
  | (Base & { diff: string; rows: { n?: number; sign: '+' | '-' | ' '; text: string }[]; summary?: string })
  /** A question picker in place of the input box until `answer`; leaves a line in the transcript */
  | (Base & { ask: string; header?: string; options: { label: string; note?: string }[]; select?: number | { at: number; index: number }[]; answer: number })
  /** A permission prompt for a command in place of the input box until `answer` */
  | (Base & { permission: string; title?: string; description?: string; select?: number | { at: number; index: number }[]; answer: number })
  /** A command typed in the transcript (shell skin, or `!` in an agent) */
  | (Base & { command: string; perChar?: number; prompt?: string })
  /** Any other text, with inline markup; `indent` in characters */
  | (Base & { text: string; indent?: number; tone?: Tone });

export type SessionFocus = { at: number; zoom: number; id?: string; x?: number; y?: number; over?: number };
export type SessionHighlight = { id: string; from: number; to?: number };

export type TerminalSessionProps = {
  lines: SessionLine[];
  theme: SessionTheme;
  skin?: keyof typeof sessionSkins | SessionSkin;
  /** Status lines under the input box, in markup (agent skins) */
  status?: string[];
  /** Window title; omit with `chrome={false}` for a full-frame terminal */
  title?: string;
  width?: number;
  height?: number;
  fontSize?: number;
  lineHeight?: number;
  /** A character's width over the font size: 0.6 suits JetBrains Mono, SF Mono, Menlo and Consolas */
  charWidth?: number;
  chrome?: boolean;
  radius?: number;
  focus?: SessionFocus[];
  highlight?: SessionHighlight[];
  style?: React.CSSProperties;
};

/** Frame your prompt finishes typing. */
export const userTypedUntil = (l: { at: number; user: string; perChar?: number }) =>
  l.at + Math.max(0, l.user.length - 1) * (l.perChar ?? 2);

/** Frame your prompt is sent (put the agent's first line after it). */
export const submittedAt = (l: { at: number; user: string; perChar?: number; submit?: number }) =>
  l.submit ?? userTypedUntil(l) + 10;

// ---------------------------------------------------------------------------------------------------------------
// Markup and wrapping
// ---------------------------------------------------------------------------------------------------------------

export type Tone = 'out' | 'dim' | 'ok' | 'err' | 'accent' | 'code';
type Seg = { text: string; tone?: Tone; bold?: boolean; strike?: boolean; bg?: string };
type Row = { segs: Seg[]; indent: number; bg?: string; item: number; enter: number };

/** `**bold**`, `` `code` `` and `{tone|text}` to segments. */
export const parseMarkup = (s: string, base: Omit<Seg, 'text'> = {}): Seg[] => {
  const re = /\*\*(.+?)\*\*|`([^`]+)`|\{(out|dim|ok|err|accent|code)\|([^}]*)\}/g;
  const out: Seg[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ ...base, text: s.slice(last, m.index) });
    if (m[1] !== undefined) out.push({ ...base, text: m[1], bold: true });
    else if (m[2] !== undefined) out.push({ ...base, text: m[2], tone: 'code' });
    else out.push({ ...base, text: m[4], tone: m[3] as Tone });
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push({ ...base, text: s.slice(last) });
  return out;
};

const sameStyle = (a: Seg, b: Seg) => a.tone === b.tone && a.bold === b.bold && a.strike === b.strike && a.bg === b.bg;

/** The first `n` words of a run of segments (spaces between them kept). */
const firstWords = (segs: Seg[], n: number): Seg[] => {
  const out: Seg[] = [];
  let words = 0;
  for (const seg of segs) {
    let text = '';
    for (const tok of seg.text.split(/(\s+)/)) {
      if (!tok) continue;
      if (!/^\s+$/.test(tok)) {
        if (words >= n) break;
        words++;
      }
      text += tok;
    }
    if (text) out.push({ ...seg, text });
    if (words >= n) break;
  }
  return out;
};

const countWords = (segs: Seg[]) => segs.reduce((n, s) => n + s.text.split(/\s+/).filter(Boolean).length, 0);

/** Word-wrap segments to `cols`, the first row starting at `first` characters and the rest at `hang`. */
const wrap = (segs: Seg[], cols: number, first: number, hang: number) => {
  const rows: { segs: Seg[]; indent: number }[] = [];
  let cur: Seg[] = [];
  let width = 0;
  let indent = first;
  let wrapped = false; // spaces are dropped only at the start of a row that a wrap began
  const avail = () => Math.max(4, cols - indent);
  const add = (text: string, style: Seg) => {
    const prev = cur[cur.length - 1];
    if (prev && sameStyle(prev, style)) prev.text += text;
    else cur.push({ ...style, text });
    width += [...text].length;
  };
  const push = (atWrap = true) => {
    const lastSeg = cur[cur.length - 1];
    if (lastSeg && atWrap) lastSeg.text = lastSeg.text.replace(/\s+$/, '');
    rows.push({ segs: cur, indent });
    wrapped = atWrap;
    cur = [];
    width = 0;
    indent = hang;
  };
  for (const seg of segs) {
    for (const tok of seg.text.split(/(\n|[^\S\n]+)/)) {
      if (!tok) continue;
      if (tok === '\n') {
        push(false);
        continue;
      }
      const len = [...tok].length;
      if (/^\s+$/.test(tok)) {
        if (width === 0 && wrapped) continue;
        if (width + len > avail()) push();
        else add(tok, seg);
        continue;
      }
      let t = [...tok];
      while (width + t.length > avail()) {
        if (width > 0) {
          push();
          continue;
        }
        add(t.slice(0, avail()).join(''), seg);
        t = t.slice(avail());
        push();
      }
      if (t.length) add(t.join(''), seg);
    }
  }
  if (cur.length || rows.length === 0) push(false);
  return rows;
};

const selectedAt = (select: number | { at: number; index: number }[] | undefined, frame: number) => {
  if (select === undefined) return 0;
  if (typeof select === 'number') return select;
  let index = 0;
  for (const s of select) if (frame >= s.at) index = s.index;
  return index;
};

// ---------------------------------------------------------------------------------------------------------------
// Layout: everything is rows of characters, so wrapping, scrolling and the camera all count the same thing
// ---------------------------------------------------------------------------------------------------------------

type Ctx = { skin: SessionSkin; cols: number; fps: number; theme: Theme };

/** Frame an item first shows in the transcript (or Infinity if it never does). */
const shownAt = (l: SessionLine, skin: SessionSkin) => {
  if ('user' in l) return skin.input === 'inline' ? l.at : submittedAt(l);
  if ('spinner' in l || 'permission' in l) return Infinity;
  if ('ask' in l) return l.answer;
  return l.at;
};

const pre = (glyph: string, tone: Tone = 'out'): Seg[] => (glyph ? [{ text: `${glyph} `, tone }] : []);

/** Rows a transcript item has at frame `f`. */
const itemRows = (l: SessionLine, i: number, f: number, c: Ctx): Row[] => {
  const { skin, cols, fps, theme } = c;
  const lead = skin.bullet ? 2 : 0;
  const rows = (segs: Seg[], first: number, hang: number, extra: Partial<Row> = {}): Row[] =>
    wrap(segs, cols, first, hang).map((r) => ({ ...r, item: i, enter: shownAt(l, skin), ...extra }));
  const tool = (head: string, toneOfBullet: Tone, body: Seg[][], bodyTone: Tone = 'dim') => {
    const out = rows([...pre(skin.bullet, toneOfBullet), ...parseMarkup(head)], 0, lead);
    body.forEach((segs, k) => {
      const elbow = k === 0 && skin.elbow ? [{ text: `${skin.elbow}  `, tone: 'dim' as Tone }] : [];
      const ind = skin.elbow ? lead : lead;
      const hang = skin.elbow ? lead + 3 : lead;
      out.push(...rows([...elbow, ...segs.map((s) => ({ ...s, tone: s.tone ?? bodyTone }))], k === 0 ? ind : hang, hang));
    });
    return out;
  };

  if ('user' in l) {
    if (skin.input === 'inline') {
      const n = Math.max(0, Math.min(l.user.length, Math.floor((f - l.at) / (l.perChar ?? 2)) + 1));
      return rows([...pre(skin.prompt, 'accent'), { text: l.user.slice(0, n) }], 0, 2);
    }
    return rows([...pre(skin.prompt, 'dim'), { text: l.user }], 0, 2, skin.userBand ? { bg: theme.band } : {});
  }
  if ('command' in l) {
    const n = Math.max(0, Math.min(l.command.length, Math.floor((f - l.at) / (l.perChar ?? 2)) + 1));
    return rows([...pre(l.prompt ?? (skin.input === 'inline' ? skin.prompt : '!'), 'accent'), { text: l.command.slice(0, n) }], 0, 2);
  }
  if ('say' in l) {
    const all = parseMarkup(l.say);
    const n = l.instant ? Infinity : Math.floor(((f - l.at) / fps) * (l.wps ?? 9)) + 1;
    return rows([...pre(skin.bullet), ...firstWords(all, n)], 0, lead);
  }
  if ('tool' in l) {
    const resultAt = l.resultAt ?? l.at + Math.round(fps * 0.6);
    const done = f >= resultAt;
    const blink = Math.floor((f - l.at) / (fps / 3)) % 2 === 0;
    const bullet: Tone = done ? (l.failed ? 'err' : 'ok') : blink ? 'out' : 'dim';
    const result = l.result === undefined ? [] : Array.isArray(l.result) ? l.result : [l.result];
    const body = done ? result.map((r) => parseMarkup(r)) : [];
    if (done && l.more) body.push([{ text: `… +${l.more} lines`, tone: 'dim' }]);
    return tool(skin.toolHead(l.tool, l.arg), bullet, body);
  }
  if ('todo' in l) {
    const firstOpen = l.todo.findIndex((t) => t.done === undefined || f < t.done);
    const body = l.todo.map((t, k): Seg[] => {
      const ticked = t.done !== undefined && f >= t.done;
      return ticked
        ? [{ text: '☒ ', tone: 'ok' }, { text: t.text, tone: 'dim', strike: true }]
        : [{ text: '☐ ', tone: 'out' }, { text: t.text, tone: 'out', bold: k === firstOpen }];
    });
    return tool(`**${l.title ?? 'Update Todos'}**`, 'ok', body, 'out');
  }
  if ('diff' in l) {
    const adds = l.rows.filter((r) => r.sign === '+').length;
    const dels = l.rows.filter((r) => r.sign === '-').length;
    const summary =
      l.summary ??
      `Updated \`${l.diff}\` with ${adds} addition${adds === 1 ? '' : 's'}${dels ? ` and ${dels} removal${dels === 1 ? '' : 's'}` : ''}`;
    const out = tool(skin.toolHead('Update', l.diff), 'ok', [parseMarkup(summary)]);
    const pad = String(Math.max(...l.rows.map((r) => r.n ?? 0), 1)).length;
    for (const r of l.rows) {
      const bg = r.sign === '+' ? theme.addBg : r.sign === '-' ? theme.delBg : undefined;
      const num = r.n === undefined ? ' '.repeat(pad) : String(r.n).padStart(pad);
      const segs: Seg[] = [
        { text: `${num} `, tone: 'dim' },
        { text: `${r.sign} `, tone: r.sign === '+' ? 'ok' : r.sign === '-' ? 'err' : 'dim' },
        { text: r.text, tone: 'out' },
      ];
      // a diff row never wraps: it is cut at the window edge like a real one
      out.push({ segs, indent: lead + 3, bg, item: i, enter: l.at });
    }
    return out;
  }
  if ('ask' in l) {
    const opt = l.options[selectedAt(l.select, l.answer)];
    const [head, ...rest] = skin.answered(l.ask, opt?.label ?? '');
    return [
      ...rows([...pre(skin.bullet), ...parseMarkup(head)], 0, lead),
      ...rest.flatMap((r, k) =>
        rows([...(k === 0 && skin.elbow ? [{ text: `${skin.elbow}  `, tone: 'dim' as Tone }] : []), ...parseMarkup(r)], lead, lead + 3),
      ),
    ];
  }
  if ('text' in l) return rows(parseMarkup(l.text, l.tone ? { tone: l.tone } : {}), l.indent ?? 0, l.indent ?? 0);
  return [];
};

/** The transcript at frame `f`: rows of every item shown so far, with the skin's gaps between them. */
const transcript = (lines: SessionLine[], f: number, c: Ctx): Row[] => {
  const out: Row[] = [];
  let first = true;
  lines.forEach((l, i) => {
    if (f < shownAt(l, c.skin)) return;
    const gap = l.gap ?? c.skin.gap;
    if (!first) for (let g = 0; g < gap; g++) out.push({ segs: [], indent: 0, item: -1, enter: -Infinity });
    first = false;
    out.push(...itemRows(l, i, f, c));
  });
  return out;
};

/** The panel under the transcript at frame `f`: spinner, input box or an open picker, and the status lines. */
const bottomRows = (lines: SessionLine[], f: number, c: Ctx, status: string[]) => {
  const { skin, cols, fps } = c;
  if (skin.input === 'inline') {
    const sp = lines.find((l) => 'spinner' in l && f >= l.at && f < l.until) as Extract<SessionLine, { spinner: string }> | undefined;
    if (!sp) return { kind: 'none' as const, rows: [] as Row[], spinner: null as Row | null, status: [] as Row[] };
    const glyph = skin.spinnerFrames[Math.floor((f - sp.at) / 3) % skin.spinnerFrames.length];
    return {
      kind: 'none' as const,
      rows: [],
      spinner: { segs: [{ text: `${glyph} `, tone: 'accent' as Tone }, ...parseMarkup(skin.spinnerLine(sp.spinner, 0))], indent: 0, item: -1, enter: sp.at },
      status: [],
    };
  }

  const statusRows = status.flatMap((s) => wrap(parseMarkup(s), cols, 2, 4).map((r) => ({ ...r, item: -1, enter: -Infinity })));

  const sp = lines.find((l) => 'spinner' in l && f >= l.at && f < l.until) as Extract<SessionLine, { spinner: string }> | undefined;
  const spinner: Row | null = sp
    ? {
        segs: [
          { text: `${skin.spinnerFrames[Math.floor((f - sp.at) / 3) % skin.spinnerFrames.length]} `, tone: 'accent' },
          ...parseMarkup(skin.spinnerLine(sp.spinner, Math.floor((f - sp.at) / fps))),
        ],
        indent: 0,
        item: -1,
        enter: sp.at,
      }
    : null;

  const open = lines.find((l) => ('ask' in l || 'permission' in l) && f >= l.at && f < l.answer) as
    | Extract<SessionLine, { ask: string }>
    | Extract<SessionLine, { permission: string }>
    | undefined;
  if (open) {
    const sel = selectedAt(open.select, f);
    const rows: Row[] = [];
    const add = (segs: Seg[], first = 0, hang = first) =>
      rows.push(...wrap(segs, cols - 4, first, hang).map((r) => ({ ...r, item: -1, enter: open.at })));
    const options =
      'ask' in open ? open.options : skin.permissionOptions(open.permission).map((label) => ({ label, note: undefined as string | undefined }));
    if ('ask' in open) {
      if (open.header) add([{ text: ` ${open.header} `, tone: 'accent', bold: true, bg: c.theme.band }]);
      add([{ text: open.ask, bold: true }]);
    } else {
      add([{ text: open.title ?? 'Bash command', bold: true }]);
      add([{ text: open.permission }], 2);
      if (open.description) add([{ text: open.description, tone: 'dim' }], 2);
      add([{ text: skin.permissionQuestion }]);
    }
    options.forEach((o, k) => {
      const on = k === sel;
      add(
        [
          { text: on ? `${skin.prompt} ` : '  ', tone: 'accent' },
          ...parseMarkup(`${k + 1}. ${o.label}`, on ? { tone: 'accent', bold: true } : {}),
        ],
        0,
        5,
      );
      if (o.note) add([{ text: o.note, tone: 'dim' }], 5);
    });
    if (skin.pickerHint) add([{ text: skin.pickerHint, tone: 'dim' }]);
    return { kind: 'panel' as const, rows, spinner: null, status: statusRows };
  }

  const typing = lines.find((l) => 'user' in l && f >= l.at && f < submittedAt(l)) as
    | Extract<SessionLine, { user: string }>
    | undefined;
  const n = typing ? Math.max(0, Math.min(typing.user.length, Math.floor((f - typing.at) / (typing.perChar ?? 2)) + 1)) : 0;
  const rows = wrap([...pre(skin.prompt, 'out'), { text: typing ? typing.user.slice(0, n) : '' }], cols, 0, 2).map((r) => ({
    ...r,
    item: -1,
    enter: -Infinity,
  }));
  return { kind: 'input' as const, rows, spinner, status: statusRows, typing: !!typing && n < (typing?.user.length ?? 0) };
};

// ---------------------------------------------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------------------------------------------

/**
 * Load a monospace font from the project's `public/` so the session looks the same on every machine
 * (`ui-monospace` is a different font on macOS, Linux and Windows). Call once at module level:
 * `loadMonoFont('JetBrains Mono', staticFile('fonts/JetBrainsMono-Regular.woff2'), staticFile('fonts/JetBrainsMono-Bold.woff2'))`,
 * then put the family first in the theme's `font`.
 */
export const loadMonoFont = (family: string, regular: string, bold?: string) => {
  if (typeof document === 'undefined' || typeof FontFace === 'undefined') return;
  const handle = delayRender(`Loading ${family}`);
  const faces = [new FontFace(family, `url(${regular})`, { weight: '400' })];
  if (bold) faces.push(new FontFace(family, `url(${bold})`, { weight: '700' }));
  Promise.all(faces.map((face) => face.load().then((loaded) => document.fonts.add(loaded))))
    .catch((err) => console.warn(`[TerminalSession] ${family} did not load, using the fallback`, err))
    .finally(() => continueRender(handle));
};

// ---------------------------------------------------------------------------------------------------------------
// The component
// ---------------------------------------------------------------------------------------------------------------

const SMOOTH = [1, 2, 3, 4, 4, 3, 2, 1]; // scroll eases over 8 frames (a triangle, so it starts and ends gently)

export const TerminalSession: React.FC<TerminalSessionProps> = ({
  lines,
  theme: themeIn,
  skin: skinIn = 'claude-code',
  status = [],
  title,
  width = 1600,
  height = 900,
  fontSize = 28,
  lineHeight = 1.5,
  charWidth = 0.6,
  chrome = true,
  radius = 12,
  focus = [],
  highlight = [],
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = fill(themeIn);
  const skin = typeof skinIn === 'string' ? sessionSkins[skinIn] : skinIn;

  const row = fontSize * lineHeight;
  const bar = chrome ? Math.round(fontSize * 1.9) : 0;
  const padY = Math.round(fontSize * 0.9);
  const padX = Math.round(fontSize * 1.2);
  const cols = Math.max(20, Math.floor((width - padX * 2) / (fontSize * charWidth)));
  const ctx: Ctx = { skin, cols, fps, theme };
  const sorted = lines.map((l, i) => ({ l, i })).sort((a, b) => a.l.at - b.l.at || a.i - b.i).map((x) => x.l);

  // the panel's height in rows: spinner (+1 blank), input or picker (+1 for its rules or box), status
  const panelHeight = (b: ReturnType<typeof bottomRows>) =>
    b.kind === 'none'
      ? (b.spinner ? 1 : 0)
      : (b.spinner ? 2 : 0) + b.rows.length + (b.kind === 'panel' ? 1.4 : 1) + (b.status.length ? b.status.length + 0.4 : 0);

  const at = (f: number) => {
    const rows = transcript(sorted, f, ctx);
    const bottom = bottomRows(sorted, f, ctx, status);
    const capacity = Math.max(1, Math.floor((height - bar - padY * 2) / row - panelHeight(bottom)));
    return { rows, bottom, capacity, scroll: Math.max(0, rows.length - capacity) };
  };

  const now = at(frame);
  const scroll =
    SMOOTH.reduce((sum, w, k) => sum + w * (k === 0 ? now.scroll : at(frame - k).scroll), 0) /
    SMOOTH.reduce((a, b) => a + b, 0);

  const toneColor = (tone: Tone = 'out') =>
    ({ out: theme.ink, dim: theme.dim, ok: theme.ok, err: theme.err, accent: theme.accent, code: theme.code })[tone];

  const caret = (typing: boolean) => (
    <span
      style={{
        display: 'inline-block',
        width: `${charWidth}em`,
        height: '1.15em',
        verticalAlign: '-0.2em',
        background: theme.ink,
        opacity: typing || Math.floor(frame / (fps / 2)) % 2 === 0 ? 1 : 0,
      }}
    />
  );

  // symbols (bullets, elbows, boxes, arrows, spinners) often come from a fallback font with another width, so each
  // gets a cell exactly one character wide and the columns after it stay in line
  const cells = (text: string) =>
    text.split(/([\u2190-\u2BFF\u2022\u00B7])/u).map((part, k) =>
      k % 2 === 1 ? (
        <span key={k} style={{ display: 'inline-block', width: '1ch', textAlign: 'center' }}>
          {part}
        </span>
      ) : (
        part
      ),
    );
  const drawSegs = (segs: Seg[]) =>
    segs.map((s, k) => (
      <span
        key={k}
        style={{
          color: toneColor(s.tone),
          fontWeight: s.bold ? 700 : 400,
          textDecoration: s.strike ? 'line-through' : undefined,
          background: s.bg,
          borderRadius: s.bg ? 4 : undefined,
        }}
      >
        {cells(s.text)}
      </span>
    ));

  const marked = (item: number) => {
    const id = sorted[item]?.id;
    if (!id) return 0;
    let o = 0;
    for (const h of highlight) {
      if (h.id !== id) continue;
      const on = interpolate(frame, [h.from, h.from + 8], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
      const off = h.to === undefined ? 1 : interpolate(frame, [h.to, h.to + 8], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
      o = Math.max(o, Math.min(on, off));
    }
    return o;
  };

  const drawRow = (r: Row, k: number, extra?: React.ReactNode) => {
    const t = r.enter === -Infinity ? 1 : Math.min(1, spring({ frame: frame - r.enter + 1, fps, config: { damping: 200 }, durationInFrames: 8 }));
    const m = r.item >= 0 ? marked(r.item) : 0;
    return (
      <div
        key={k}
        style={{
          position: 'relative',
          height: row,
          whiteSpace: 'pre',
          overflow: 'hidden',
          paddingLeft: `${r.indent}ch`,
          background: r.bg,
          marginLeft: r.bg ? -padX / 2 : 0,
          marginRight: r.bg ? -padX / 2 : 0,
          paddingRight: r.bg ? padX / 2 : 0,
          boxSizing: 'border-box',
          opacity: frame < r.enter ? 0 : t,
        }}
      >
        {m > 0 && (
          <div style={{ position: 'absolute', inset: `0 ${-padX / 2}px`, background: theme.mark, opacity: m, borderRadius: 4 }} />
        )}
        <span style={{ position: 'relative', marginLeft: r.bg ? padX / 2 : 0 }}>
          {drawSegs(r.segs)}
          {extra}
        </span>
      </div>
    );
  };

  // ---- camera: zoom onto a line by id, or a point (x, y as 0-1 of the window)
  const yOf = (id: string | undefined) => {
    if (!id) return height / 2;
    const idx = now.rows.findIndex((r) => r.item >= 0 && sorted[r.item]?.id === id);
    if (idx < 0) return height / 2;
    const span = now.rows.filter((r) => r.item >= 0 && sorted[r.item]?.id === id).length;
    return bar + padY + (idx - scroll + span / 2) * row;
  };
  const view = (k: SessionFocus | undefined) =>
    k ? { z: k.zoom, x: (k.x ?? 0.5) * width, y: k.id ? yOf(k.id) : (k.y ?? 0.5) * height } : { z: 1, x: width / 2, y: height / 2 };
  const keys = [...focus].sort((a, b) => a.at - b.at);
  let cam = view(undefined);
  for (let k = 0; k < keys.length; k++) {
    if (frame < keys[k].at) break;
    const from = k === 0 ? view(undefined) : view(keys[k - 1]);
    const to = view(keys[k]);
    const p = interpolate(frame, [keys[k].at, keys[k].at + (keys[k].over ?? 20)], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.cubic),
    });
    cam = { z: from.z + (to.z - from.z) * p, x: from.x + (to.x - from.x) * p, y: from.y + (to.y - from.y) * p };
  }
  const tx = Math.min(0, Math.max(width - width * cam.z, width / 2 - cam.x * cam.z));
  const ty = Math.min(0, Math.max(height - height * cam.z, height / 2 - cam.y * cam.z));

  const b = now.bottom;
  const lastRow = now.rows[now.rows.length - 1];
  const lastItem = sorted[lastRow?.item ?? -1];
  const inlineCaret =
    skin.input === 'inline' && lastItem && ('command' in lastItem || 'user' in lastItem) && lastRow.item === now.rows[now.rows.length - 1].item;

  return (
    <div data-qc="product" style={{ width, height, borderRadius: radius, overflow: 'hidden', position: 'relative', ...style }}>
      <div
        style={{
          width,
          height,
          background: theme.bg,
          fontFamily: theme.font,
          fontSize,
          lineHeight: `${row}px`,
          color: theme.ink,
          transformOrigin: '0 0',
          transform: `translate(${tx}px, ${ty}px) scale(${cam.z})`,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {chrome && (
          <div
            style={{
              height: bar,
              flex: 'none',
              background: theme.bar,
              display: 'flex',
              alignItems: 'center',
              padding: `0 ${Math.round(fontSize * 0.7)}px`,
              gap: Math.round(fontSize * 0.3),
              position: 'relative',
            }}
          >
            {[0, 1, 2].map((k) => (
              <div key={k} style={{ width: fontSize * 0.42, height: fontSize * 0.42, borderRadius: '50%', background: theme.dim, opacity: 0.45 }} />
            ))}
            {title && (
              <div style={{ position: 'absolute', left: 0, right: 0, textAlign: 'center', color: theme.dim, fontSize: fontSize * 0.72, lineHeight: `${bar}px` }}>
                {title}
              </div>
            )}
          </div>
        )}

        <div style={{ flex: 1, padding: `${padY}px ${padX}px`, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {/* transcript: clipped to whole rows */}
          <div style={{ height: now.capacity * row, overflow: 'hidden', flex: 'none' }}>
            <div style={{ transform: `translateY(${-scroll * row}px)` }}>
              {now.rows.map((r, k) =>
                drawRow(r, k, inlineCaret && k === now.rows.length - 1 ? caret(false) : undefined),
              )}
            </div>
          </div>

          {/* the panel: spinner, then the input box or an open picker, then the status lines */}
          <div style={{ flex: 'none', marginTop: 'auto' }}>
            {b.spinner && (
              <>
                {b.kind !== 'none' && <div style={{ height: row }} />}
                {drawRow(b.spinner, -1)}
              </>
            )}
            {b.kind === 'input' && (
              <div
                style={{
                  borderTop: skin.input === 'rules' ? `2px solid ${theme.rule}` : undefined,
                  borderBottom: skin.input === 'rules' ? `2px solid ${theme.rule}` : undefined,
                  background: skin.input === 'box' ? theme.band : undefined,
                  borderRadius: skin.input === 'box' ? 6 : 0,
                  padding: `${row * 0.5 - 2}px ${skin.input === 'box' ? padX / 2 : 0}px`,
                  margin: `0 ${skin.input === 'rules' ? -padX : 0}px`,
                  paddingLeft: skin.input === 'rules' ? padX : undefined,
                }}
              >
                {b.rows.map((r, k) => drawRow(r, k, k === b.rows.length - 1 ? caret(!!b.typing) : undefined))}
              </div>
            )}
            {b.kind === 'panel' && (
              <div style={{ border: `2px solid ${theme.rule}`, borderRadius: 8, padding: `${row * 0.2}px ${padX / 2}px` }}>
                {b.rows.map((r, k) => drawRow(r, k))}
              </div>
            )}
            {b.status.length > 0 && (
              <div style={{ paddingTop: row * 0.4 }}>{b.status.map((r, k) => drawRow(r, k))}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
