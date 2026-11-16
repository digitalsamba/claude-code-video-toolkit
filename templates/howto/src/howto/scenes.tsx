/**
 * The scene kinds. Each takes its props (already resolved against the narration), the frame's layout and its
 * act's field. Sizes and colours come from theme.ts only.
 */
import React from 'react';
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { Counter, terminalTheme, TerminalSession } from '../../../../lib/components';
import { BRAND } from '../brand';
import { arrive, Ground, Pen, Punch, Rail, ramp, Rise, stopAt, UnderTrack, useEnter, railGeometry, Words } from './parts';
import { display, type Field, figure, label, type Layout, MONO, note, SANS, title } from './theme';
import type { BarsProps, CommandProps, FigureProps, LedgerProps, Lines, Said, SessionProps, StatementProps } from './types';

type Base = { L: Layout; f: Field };

const broken = (lines: Lines) =>
  lines.map((l, i) => (
    <React.Fragment key={i}>
      {i > 0 && <br />}
      {l}
    </React.Fragment>
  ));

/** The current one of a list of cued things: the last whose frame has come. */
const current = <T extends { at: number }>(items: T[], frame: number) => {
  let k = -1;
  items.forEach((it, i) => {
    if (frame >= it.at) k = i;
  });
  return k;
};

/** Where a figure scene sets its number: the next scene's `hero` starts from here. */
const figureBox = (L: Layout) => ({ left: L.x - 8, top: L.vertical ? 640 : 300, size: L.vertical ? 260 : 300 });

const Foot: React.FC<Base & { text?: string }> = ({ L, f, text }) =>
  text ? <div style={{ ...note(L, f), position: 'absolute', left: L.x, top: L.foot - L.size.note, maxWidth: L.vertical ? L.right - L.x : '62%' }}>{text}</div> : null;

// ---------------------------------------------------------------- opening, chapter card, close

/**
 * The opening: a prompt and a caret; the chevron fills into play; the track draws out with a stop for each
 * chapter; the title rises over it. The narration starts on the title (layout.json `open.lead`).
 */
export const Open: React.FC<Base & { text: Lines; chapters: number; titleAt: number; lineCues?: number[]; stopCues?: number[] }> = ({
  L,
  f,
  text,
  chapters,
  titleAt,
  lineCues = [],
  stopCues = [],
}) => {
  const frame = useCurrentFrame();
  const blink = frame < 22 ? Math.floor(frame / 8) % 2 === 0 : true;
  const lit = Array.from({ length: chapters }, (_, i) => stopCues[i] !== undefined && frame >= stopCues[i]);
  const state = { fill: ramp(frame, 20, 14), drawn: ramp(frame, 30, 30), head: 0, caretOn: blink, lit };
  const g = railGeometry(L);
  return (
    <>
      <Ground L={L} f={f} />
      <Rail L={L} f={f} n={chapters} state={state} />
      <div style={{ ...display(L, f), position: 'absolute', left: L.x, bottom: L.H - (L.trackY - g.s - (L.vertical ? 90 : 72)) }}>
        {text.map((line, i) => (
          <Words key={i} text={line} at={lineCues[i] ?? titleAt} blur={10} gap={3} />
        ))}
      </div>
    </>
  );
};

export const ASK_START = 8; // a chapter's question starts typing here, a character a frame
/** How long a chapter card with a question runs: it types, is sent, and the answer opens. */
export const askFrames = (text: string) => ASK_START + text.length + 22;

/** Where the question sits: the answer opens out of this box. */
export const askBox = (L: Layout) => {
  const g = railGeometry(L);
  const h = L.vertical ? 250 : 190;
  const bottom = L.trackY - g.s - 70;
  return { left: L.x - 36, right: L.right + 36, top: bottom - h, bottom, h };
};

/**
 * A chapter's question, typed at a prompt the way you would ask an agent: the largest size that fits, a block
 * caret, a hairline box. When it is sent the caret goes and the box takes the accent.
 */
const Ask: React.FC<Base & { text: string; frame: number }> = ({ L, f, text, frame }) => {
  const b = askBox(L);
  const typed = Math.max(0, Math.min(text.length, frame - ASK_START));
  const sent = frame >= ASK_START + text.length + 8;
  const room = b.right - b.left - 72;
  const perLine = L.vertical ? Math.ceil((text.length + 2) / 2) + 2 : text.length + 2;
  const size = Math.min(L.size.figure, Math.floor(room / (perLine * 0.6)));
  const blink = typed < text.length || Math.floor(frame / 8) % 2 === 0;
  return (
    <div
      style={{
        position: 'absolute',
        left: b.left,
        width: b.right - b.left,
        top: b.top,
        height: b.h,
        boxSizing: 'border-box',
        border: `3px solid ${sent ? f.accent : f.hair.replace(/[\d.]+\)$/, '0.35)')}`,
        borderRadius: 18,
        display: 'flex',
        alignItems: 'center',
        padding: '0 36px',
        fontFamily: MONO,
        fontWeight: 700,
        fontSize: size,
        lineHeight: 1.15,
        letterSpacing: '-0.02em',
        color: f.ink,
      }}
    >
      <div style={{ whiteSpace: L.vertical ? 'pre-wrap' : 'pre' }}>
        <span style={{ color: f.accent }}>❯ </span>
        {text.slice(0, typed)}
        {!sent && <span style={{ display: 'inline-block', width: '0.55em', height: '0.9em', verticalAlign: '-0.12em', background: f.accent, marginLeft: 4, opacity: blink ? 1 : 0 }} />}
      </div>
    </div>
  );
};

/**
 * A chapter card: the playhead moves to the chapter's stop and the chapter's title rises. With `instant` (a
 * Short's first frame, its cover) everything is already in place on frame 0.
 */
export const ChapterCard: React.FC<Base & { index: number; of: number; text: Lines; ask?: string; instant?: boolean }> = ({ L, f, index, of, text, ask, instant }) => {
  const frame = useCurrentFrame();
  const from = index === 0 ? 0 : stopAt(index - 1, of);
  const to = stopAt(index, of);
  const head = instant ? to : from + (to - from) * ramp(frame, 4, 18);
  const g = railGeometry(L);
  const landed = 20; // the playhead reaches the stop, which lands large and settles
  const pop = instant || frame < landed ? undefined : { stop: index, scale: 1 + 0.9 * (1 - arrive(frame, landed, 12)) };
  const big = L.vertical ? figure(L, f) : display(L, f);
  return (
    <>
      <Ground L={L} f={f} />
      <Rail L={L} f={f} n={of} state={{ fill: 1, drawn: 1, head, caretOn: true, pop }} />
      {ask ? (
        <Ask L={L} f={f} text={ask} frame={instant ? 999 : frame} />
      ) : (
        <div style={{ ...big, position: 'absolute', left: L.x, bottom: L.H - (L.trackY - g.s - 64) }}>
          {text.map((line, i) => (
            <Words key={i} text={line} at={instant ? -20 : 6 + i * 5} blur={10} gap={3} />
          ))}
        </div>
      )}
      <UnderTrack L={L} f={f} at={instant ? -10 : 12}>
        {index + 1} of {of}
      </UnderTrack>
    </>
  );
};

/**
 * The close: the playhead runs to the end, the track rolls back into the caret, the play triangle opens into a
 * prompt again and the caret blinks: your turn. "Next" rises on the narration; the link under the track.
 */
export const Close: React.FC<Base & { chapters: number; next: Lines; link: string; nextAt: number; linkAt: number; endAt: number; runUntil?: number; hold?: boolean }> = ({
  L,
  f,
  chapters,
  next,
  link,
  nextAt,
  linkAt,
  endAt,
  runUntil,
  hold,
}) => {
  const frame = useCurrentFrame();
  const last = stopAt(chapters - 1, chapters);
  const run = ramp(frame, 6, Math.max(22, (runUntil ?? 28) - 6)); // to the end of the track, while "next" is said
  const back = hold ? 0 : ramp(frame, endAt, 24); // and back into the caret (with `hold` the brand's ident closes instead)
  const head = back > 0 ? 1 - back : last + (1 - last) * run;
  const fill = hold ? 1 : 1 - ramp(frame, endAt + 22, 14);
  const done = !hold && frame > endAt + 36;
  const blink = done ? Math.floor((frame - endAt - 36) / 15) % 2 === 0 : true;
  const g = railGeometry(L);
  return (
    <>
      <Rail L={L} f={f} n={chapters} state={{ fill, drawn: back > 0 ? 1 - back : 1, head, caretOn: blink }} />
      <Rise at={nextAt} style={{ ...figure(L, f), position: 'absolute', left: L.x, bottom: L.H - (L.trackY - g.s - 64) }}>
        {broken(next)}
      </Rise>
      <UnderTrack L={L} f={f} at={linkAt}>
        <span style={{ color: f.ink }}>{link}</span>
      </UnderTrack>
    </>
  );
};

// ---------------------------------------------------------------- ledger: a receipt, rows landing on their words

/** A line that can grow: `then` lands on later words, in the secondary ink */
const SaidLine: React.FC<{ said: Said; style: React.CSSProperties; thenColor: string }> = ({ said, style, thenColor }) => (
  <div style={style}>
    <Words inline text={said.text} at={said.at} />
    {said.then && !said.then.newLine && ' '}
    {said.then && <Words inline={!said.then.newLine} text={said.then.text} at={said.then.at} style={{ color: thenColor }} />}
  </div>
);

export const Ledger: React.FC<Base & LedgerProps> = ({ L, f, title: head, rows, allAt, note: foot, after, hero }) => {
  const frame = useCurrentFrame();
  const active = after && frame >= after.at ? -1 : current(rows, frame);
  const all = allAt !== undefined && frame >= allAt && current(rows, frame) === current(rows, allAt);
  const big = rows.length <= 3; // a few rows fill the frame: the detail under the label, the value at figure size
  const rowGap = L.vertical ? (big ? 96 : 64) : big ? 60 : 44;
  const titleTop = L.top + (hero && L.vertical ? 190 : 0);
  const rowsTop = titleTop + (head ? (L.vertical ? 220 : 170) : 20);
  // the hero: the number this ledger is about. It starts where the figure scene before left it and moves into place
  const from = figureBox(L);
  const to = L.vertical ? { left: L.x - 4, top: L.top - 20, size: L.size.display } : { left: L.x - 6, top: rowsTop + 150, size: L.size.display };
  const k = hero?.carry === false ? 1 : ramp(frame, 3, 20);
  const mix = (p: number, q: number) => p + (q - p) * k;
  const rowsLeft = hero && !L.vertical ? L.x + 560 : L.x;
  return (
    <>
      {head && <SaidLine said={head} thenColor={f.ink2} style={{ ...title(L, f), position: 'absolute', left: L.x, top: titleTop }} />}
      {hero && (
        <Punch
          at={hero.carry === false ? (hero.at ?? 0) : -30}
          style={{ ...display(L, f), fontSize: to.size, color: f.accent, position: 'absolute', left: mix(from.left, to.left), top: mix(from.top, to.top), whiteSpace: 'nowrap' }}
        >
          <div style={{ transform: `scale(${mix(from.size / to.size, 1)})`, transformOrigin: 'left top' }}>{hero.text}</div>
        </Punch>
      )}
      <div style={{ position: 'absolute', left: rowsLeft, right: L.W - L.right, top: rowsTop, display: 'flex', flexDirection: 'column', gap: rowGap }}>
        {rows.map((r, i) => {
          const on = i === active || (all && i <= current(rows, frame));
          const vAt = r.valueAt ?? r.at;
          const under = big || L.vertical; // the detail goes under the label
          const value = frame < vAt ? null : typeof r.value === 'number' ? <Counter value={r.value} start={vAt} frames={22} format={r.format ?? ((n) => Math.round(n).toLocaleString('en-GB'))} /> : r.value;
          const detail = r.detail && (
            <Rise at={r.detailAt ?? r.at} style={{ ...label(L, f), marginLeft: under ? 0 : 28, marginTop: under ? 8 : 0, whiteSpace: under && L.vertical ? undefined : 'nowrap' }}>
              {r.detail}
            </Rise>
          );
          return (
            <div key={i} style={{ display: 'flex', alignItems: under ? 'center' : 'baseline' }}>
              <div style={under ? undefined : { display: 'flex', alignItems: 'baseline' }}>
                <Words text={r.label} at={r.at} style={{ ...title(L, f), whiteSpace: 'nowrap' }} />
                {detail}
              </div>
              <div
                style={{
                  flex: 1,
                  margin: '0 24px',
                  borderBottom: `4px dotted ${f.ink}`,
                  opacity: 0.22,
                  transform: `translateY(${under ? 0 : -10}px) scaleX(${arrive(frame, r.at, 18)})`,
                  transformOrigin: 'left center',
                }}
              />
              <div style={{ position: 'relative' }}>
                <Punch at={vAt} origin="right center" style={{ ...(big && typeof r.value === 'number' ? figure(L, f) : title(L, f)), color: on ? f.accent : f.ink, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
                  {value}
                </Punch>
                {i === active && !all && <Pen at={vAt + (typeof r.value === 'number' ? 22 : 8)} color={f.accent} padY={big ? 14 : 12} padX={big ? 34 : 26} />}
              </div>
            </div>
          );
        })}
      </div>
      {after && (
        <SaidLine
          said={after}
          thenColor={f.accent}
          style={{ ...title(L, f), color: f.accent, position: 'absolute', left: L.x, right: L.W - L.right, top: L.foot - L.size.note - (L.vertical ? 260 : 150) }}
        />
      )}
      <Foot L={L} f={f} text={foot} />
    </>
  );
};

// ---------------------------------------------------------------- figure: one number, the lines that explain it

export const Figure: React.FC<Base & FigureProps> = ({ L, f, title: head, value, prefix = '', suffix = '', format, at, lines, note: foot }) => {
  const frame = useCurrentFrame();
  const k = current(lines, frame);
  let first = k;
  while (first > 0 && lines[first].add) first--;
  const shown = k < 0 ? [] : lines.slice(first, k + 1);
  const box = figureBox(L);
  const figTop = box.top;
  return (
    <>
      <Ground L={L} f={f} />
      {head && <SaidLine said={head} thenColor={f.ink2} style={{ ...title(L, f), position: 'absolute', left: L.x, top: L.top }} />}
      <Punch at={at} from={1.22} origin="left center" style={{ ...display(L, f), fontSize: box.size, position: 'absolute', left: box.left, top: figTop, color: f.accent, fontVariantNumeric: 'tabular-nums' }}>
        {prefix}
        <Counter value={value} start={at} frames={26} format={format ?? ((n) => String(Math.round(n)))} />
        {suffix}
      </Punch>
      <div style={{ position: 'absolute', left: L.x, right: L.W - L.right, top: figTop + (L.vertical ? 300 : 330), display: 'flex', flexDirection: 'column', gap: 18 }}>
        {shown.map((l, i) => (
          <Rise key={`${first}-${i}`} at={l.at} style={{ ...title(L, f), color: i === shown.length - 1 ? f.ink : f.ink2, position: 'relative', alignSelf: 'flex-start' }}>
            {broken(l.text)}
            {i === shown.length - 1 && <Pen kind="under" at={l.at + 12} color={f.accent} />}
          </Rise>
        ))}
      </div>
      <Foot L={L} f={f} text={foot} />
    </>
  );
};

// ---------------------------------------------------------------- statement: lines that arrive with the words

export const Statement: React.FC<Base & StatementProps> = ({ L, f, lines, note: foot }) => {
  const frame = useCurrentFrame();
  const k = current(lines, frame);
  const style = { display, figure, title };
  return (
    <>
      <Ground L={L} f={f} />
      <div style={{ position: 'absolute', left: L.x, right: L.W - L.right, top: L.top + 60, display: 'flex', flexDirection: 'column', gap: 40 }}>
        {lines.map((l, i) => (
          <Rise key={i} at={l.at} style={{ ...style[l.size ?? 'title'](L, f), color: i === k ? f.ink : f.ink2 }}>
            {broken(l.text)}
          </Rise>
        ))}
      </div>
      <Foot L={L} f={f} text={foot} />
    </>
  );
};

// ---------------------------------------------------------------- bars: a few values against a limit

export const Bars: React.FC<Base & BarsProps> = ({ L, f, title: head, bars, limit, format, note: foot, after }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const summing = after && frame >= after.at;
  const k = summing ? -1 : current(bars, frame);
  const top = L.top + (head ? (L.vertical ? 300 : 210) : 60);
  const bottom = L.foot - (L.vertical ? 520 : 190);
  const h = bottom - top;
  const max = Math.max(...bars.map((b) => b.value), limit?.value ?? 0) * 1.12;
  const y = (v: number) => bottom - (v / max) * h;
  const span = L.right - L.x;
  const slot = span / bars.length;
  const bw = L.vertical ? slot * 0.76 : Math.min(slot * 0.56, 220);
  const limitOn = useEnter(limit?.at ?? Number.MAX_SAFE_INTEGER);
  return (
    <>
      {head && !summing && (
        <div style={{ ...title(L, f), position: 'absolute', left: L.x, top: L.top }}>
          {head.text.map((line, i) => (
            <Words key={i} text={line} at={head.at + i * 4} />
          ))}
        </div>
      )}
      {after && summing && (
        <Words text={after.text} at={after.at} style={{ ...title(L, f), position: 'absolute', left: L.x, top: L.top, color: f.accent }} />
      )}
      <svg width={L.W} height={L.H} style={{ position: 'absolute', inset: 0 }}>
        <line x1={L.x} x2={L.right} y1={bottom} y2={bottom} stroke={f.ink} strokeOpacity={0.35} strokeWidth={2} />
        {bars.map((b, i) => {
          const grow = arrive(frame, b.at, 20);
          const x = L.x + i * slot + (slot - bw) / 2;
          const yy = y(b.value * grow);
          return <rect key={i} x={x} y={yy} width={bw} height={bottom - yy} fill={i === k ? f.accent : f.other} />;
        })}
        {limit && (
          <line
            x1={L.x}
            x2={L.x + span * limitOn}
            y1={y(limit.value)}
            y2={y(limit.value)}
            stroke={summing ? f.accent : f.ink}
            strokeWidth={summing ? 5 : 3}
            strokeDasharray="14 10"
          />
        )}
      </svg>
      {limit && (
        <div style={{ ...label(L, f), position: 'absolute', right: L.W - L.right, top: y(limit.value) - L.size.label * 1.6, color: summing ? f.accent : f.ink, fontWeight: 700, opacity: limitOn }}>
          {limit.label}
        </div>
      )}
      {bars.map((b, i) => {
        const x = L.x + i * slot;
        const vAt = b.valueAt ?? b.at + 8;
        const shown = frame >= vAt;
        return (
          <React.Fragment key={i}>
            <div style={{ ...label(L, f), position: 'absolute', left: x, width: slot, top: bottom + 16, textAlign: 'center', color: f.ink }}>{b.label}</div>
            {b.sub && (
              <Rise at={b.sub.at} style={{ ...label(L, f), position: 'absolute', left: x, width: slot, top: bottom + 16 + L.size.label * 1.35, textAlign: 'center' }}>
                {b.sub.text}
              </Rise>
            )}
            {shown && (
              <Punch at={vAt} origin="center top" from={1.3} style={{ ...label(L, f), fontWeight: 700, position: 'absolute', left: x, width: slot, top: y(b.value) + 14, textAlign: 'center', color: i === k ? f.ground : f.ink, fontVariantNumeric: 'tabular-nums' }}>
                <span style={{ position: 'relative', display: 'inline-block' }}>
                  {format(b.value)}
                  {i === k && <Pen at={vAt + 10} color={f.ground} padX={16} padY={8} width={4} />}
                </span>
              </Punch>
            )}
          </React.Fragment>
        );
      })}
      <Foot L={L} f={f} text={foot} />
    </>
  );
};

// ---------------------------------------------------------------- session: the field is the terminal

export const Session: React.FC<Base & SessionProps> = ({ L, f, lines, skin = 'claude-code', status, focus, highlight }) => {
  const fontSize = L.vertical ? 34 : 36;
  const padX = Math.round(fontSize * 1.2);
  const theme = { ...terminalTheme(BRAND), bg: f.ground, bar: f.ground, font: MONO };
  return (
    <TerminalSession
      skin={skin}
      theme={theme}
      lines={lines}
      status={status}
      chrome={false}
      radius={0}
      width={L.W - 2 * (L.x - padX)}
      height={L.H - (L.vertical ? 720 : 60)}
      fontSize={fontSize}
      focus={focus}
      highlight={highlight}
      style={{ position: 'absolute', left: L.x - padX, top: L.vertical ? 140 : 30 }}
    />
  );
};

// ---------------------------------------------------------------- command: the one thing to type

export const Command: React.FC<Base & CommandProps> = ({ L, f, command, at, lines, note: foot }) => {
  const frame = useCurrentFrame();
  const typed = Math.max(0, Math.min(command.length, Math.floor((frame - at) / 3) + 1));
  const blinkOn = frame < at + command.length * 3 || Math.floor(frame / 15) % 2 === 0;
  const size = L.size.figure;
  return (
    <>
      <Ground L={L} f={f} />
      <div style={{ position: 'absolute', left: L.x, top: L.vertical ? 520 : 300, fontFamily: MONO, fontWeight: 700, fontSize: size, color: f.ink, whiteSpace: 'pre', letterSpacing: '-0.02em' }}>
        <span style={{ color: f.accent }}>❯ </span>
        {frame >= at ? command.slice(0, typed) : ''}
        <span style={{ display: 'inline-block', width: '0.55em', height: size * 0.12, background: f.accent, marginLeft: 8, opacity: blinkOn ? 1 : 0 }} />
      </div>
      <div style={{ position: 'absolute', left: L.x, right: L.W - L.right, top: (L.vertical ? 520 : 300) + size * 1.6, display: 'flex', flexDirection: 'column', gap: 18 }}>
        {lines.map((l, i) => (
          <Rise key={i} at={l.at} style={{ ...title(L, f), fontFamily: SANS }}>
            {l.text}
          </Rise>
        ))}
      </div>
      <Foot L={L} f={f} text={foot} />
    </>
  );
};

export const SCENES = { ledger: Ledger, figure: Figure, statement: Statement, bars: Bars, session: Session, command: Command };

