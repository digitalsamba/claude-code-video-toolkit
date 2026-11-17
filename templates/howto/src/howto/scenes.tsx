/**
 * The scene kinds. Each takes its props (already resolved against the narration), the frame's layout and its
 * act's field. Sizes and colours come from theme.ts only.
 */
import React from 'react';
import { Loop, OffthreadVideo, Img, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { Counter, terminalTheme, TerminalSession } from '../../../../lib/components';
import { BRAND } from '../brand';
import { arrive, Ground, Pen, Punch, Rail, ramp, Rise, stopAt, UnderTrack, useEnter, railGeometry, Words } from './parts';
import { display, type Field, FIELDS, figure, label, type Layout, MONO, note, SANS, title } from './theme';
import { ASK_SENT, ASK_START } from './timeline';
import type { BarsProps, CommandProps, FigureProps, LedgerProps, LedgerRow, Lines, Said, SessionProps, SplitProps, SplitSide, StatementProps } from './types';

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

/** In a Short nothing a scene draws goes below this: the burned captions sit under it (Episode.tsx, bottom 330). */
export const CAPTION_TOP = 1380;

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
  const sent = frame >= ASK_START + text.length + ASK_SENT;
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

/** A question on its own, before the opening (an episode's hook). */
export const Question: React.FC<Base & { text: string; instant?: boolean }> = ({ L, f, text, instant }) => {
  const frame = useCurrentFrame();
  return (
    <>
      <Ground L={L} f={f} />
      <Ask L={L} f={f} text={text} frame={instant ? 999 : frame} />
    </>
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

/** What a row is about, in a window: a clip playing, images flicked through, or a waveform for something heard. */
const Shown: React.FC<{ show: NonNullable<LedgerRow['show']>; at: number; box: { left: number; top: number; w: number; h: number } }> = ({ show, at, box }) => {
  const frame = useCurrentFrame();
  const t = arrive(frame, at, 12);
  const k = frame - at;
  const dark = FIELDS.work;
  return (
    <div style={{ position: 'absolute', left: box.left, top: box.top, width: box.w, height: box.h, borderRadius: 12, overflow: 'hidden', background: dark.ground, opacity: Math.min(1, t * 3), transform: `scale(${0.94 + 0.06 * t})`, transformOrigin: 'left top' }}>
      {show.video && (
        <Sequence from={at} layout="none">
          <Loop durationInFrames={140}>
            <OffthreadVideo src={staticFile(show.video)} muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </Loop>
        </Sequence>
      )}
      {show.images && (
        <Img
          src={staticFile(show.images[Math.floor(Math.max(0, k) / 10) % show.images.length])}
          // one picture is pushed in slowly, so the window is never a frozen frame; several are flicked through
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: show.images.length === 1 ? `scale(${1 + Math.min(0.32, Math.max(0, k) * 0.0011)})` : undefined }}
        />
      )}
      {show.wave && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: box.w / 90 }}>
          {Array.from({ length: 36 }, (_, i) => {
            const h = 0.12 + 0.8 * Math.abs(Math.sin(i * 0.7 + k * 0.27) * Math.sin(i * 0.31 - k * 0.12));
            return <div key={i} style={{ width: box.w / 72, height: `${h * 78}%`, borderRadius: box.w / 144, background: dark.accent }} />;
          })}
        </div>
      )}
    </div>
  );
};

export const Ledger: React.FC<Base & LedgerProps> = ({ L, f, title: head, rows, allAt, note: foot, after, hero, big: large }) => {
  const frame = useCurrentFrame();
  const active = after && frame >= after.at ? -1 : current(rows, frame);
  const all = allAt !== undefined && frame >= allAt && current(rows, frame) === current(rows, allAt);
  const big = large ?? rows.length <= 3; // a few rows fill the frame: the detail under the label, the value at figure size
  const rowGap = L.vertical ? (big ? 96 : 64) : big ? 60 : 44;
  const shows = rows.some((r) => r.show); // a window beside the rows
  const aside = !!hero || shows; // the left of the frame is taken, by the hero, the window or both
  const titleTop = L.top + (aside && L.vertical ? (hero ? 190 : 250) : 0);
  const rowsTop = titleTop + (head ? (L.vertical ? 220 : 170) : 20);
  // the hero: the number this ledger is about. It starts where the figure scene before left it and moves into place
  const from = figureBox(L);
  const to = L.vertical ? { left: L.x - 4, top: L.top - 20, size: L.size.display } : { left: L.x - 6, top: rowsTop + (shows ? -10 : 150), size: L.size.display };
  // with a hero the window sits under it (beside it in a Short); alone it takes the hero's place
  const win = L.vertical
    ? { left: hero ? L.right - 480 : L.x, top: L.top - 50, w: 480, h: 270 }
    : { left: L.x, top: rowsTop + (hero ? 215 : 12), w: 480, h: 270 };
  let showing = -1;
  rows.forEach((r, i) => {
    if (r.show && frame >= r.at) showing = i;
  });
  const k = hero?.carry === false ? 1 : ramp(frame, 3, 20);
  const mix = (p: number, q: number) => p + (q - p) * k;
  const rowsLeft = aside && !L.vertical ? L.x + 560 : L.x;
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
      {showing >= 0 && <Shown key={showing} show={rows[showing].show!} at={rows[showing].at} box={win} />}
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
          // in a Short the burned captions own the band above the note, so the line ends above them
          style={{ ...title(L, f), color: f.accent, position: 'absolute', left: L.x, right: L.W - L.right, ...(L.vertical ? { bottom: L.H - CAPTION_TOP + 40 } : { top: L.foot - L.size.note - 150 }) }}
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

// ---------------------------------------------------------------- split: who does what, counted

const landed = (side: SplitSide, frame: number) => side.rows.filter((r) => frame >= r.at);
const lastAt = (side: SplitSide, frame: number) => Math.max(-1, ...landed(side, frame).map((r) => r.at));

/**
 * Two columns ("You", "Claude Code"), each row landing on its words, and under each column a count of what has
 * landed so far. The accent is on the row being said; when a column's count is said (`countAt`) the count takes
 * it and the pen circles it. In a Short the columns stack and each count sits at the end of its head's line.
 */
export const Split: React.FC<Base & SplitProps> = ({ L, f, title: head, left, right, note: foot }) => {
  const frame = useCurrentFrame();
  const sides = [left, right];
  // what holds the accent: the newest of every row and every count that has been said
  const rowAt = Math.max(lastAt(left, frame), lastAt(right, frame));
  const countSaid = sides.map((s) => (s.countAt !== undefined && frame >= s.countAt ? s.countAt : -1));
  const newest = Math.max(rowAt, ...countSaid);
  const top = L.top + (head ? (L.vertical ? 170 : 150) : 0);
  const gutter = L.vertical ? 0 : 96;
  const span = L.right - L.x;
  const leftW = L.vertical ? span : Math.round((span - gutter) * 0.42);
  const boxes = L.vertical
    ? null
    : [
        { left: L.x, width: leftW },
        { left: L.x + leftW + gutter, width: span - leftW - gutter },
      ];
  const rowStyle: React.CSSProperties = { ...label(L, f), color: f.ink, lineHeight: 1.25 };
  const column = (side: SplitSide, i: number) => {
    // the count climbs to each row's share over a few frames, so a row worth ten is seen being counted
    const n = Math.round(side.rows.reduce((sum, r) => sum + (r.count ?? 1) * ramp(frame, r.at, Math.min(48, 6 + 4 * (r.count ?? 1))), 0));
    const counted = countSaid[i] >= 0 && countSaid[i] === newest;
    const bump = counted ? side.countAt! : lastAt(side, frame);
    const count = (
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Punch key={bump} at={bump} origin={L.vertical ? 'right center' : 'left bottom'} style={{ ...(L.vertical ? title(L, f) : figure(L, f)), color: counted ? f.accent : f.ink, fontVariantNumeric: 'tabular-nums' }}>
          {n}
        </Punch>
        {counted && <Pen at={side.countAt! + 8} color={f.accent} padX={L.vertical ? 22 : 30} padY={L.vertical ? 10 : 14} />}
      </div>
    );
    return (
      <div key={i} style={L.vertical ? { marginBottom: 64 } : { position: 'absolute', top, bottom: L.H - L.foot + L.size.note + 44, ...boxes![i], display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
          <Words text={side.head} at={4 + i * 6} style={{ ...title(L, f), whiteSpace: 'nowrap' }} />
          {L.vertical && n > 0 && (
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 44 }}>
              {side.unit && <div style={label(L, f)}>{side.unit}</div>}
              {count}
            </div>
          )}
        </div>
        <div style={{ height: 3, background: f.ink, opacity: 0.35, margin: L.vertical ? '14px 0 22px' : '16px 0 26px', transform: `scaleX(${arrive(frame, 4 + i * 6, 20)})`, transformOrigin: 'left center' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: L.vertical ? 12 : 24, flex: 1 }}>
          {side.rows.map((r, k) => (
            <div key={k}>
              <Words text={r.text} at={r.at} style={{ ...rowStyle, color: r.at === newest && frame >= r.at ? f.accent : f.ink, fontWeight: 700 }} />
              {r.detail && (
                <Rise at={r.detailAt ?? r.at} style={{ ...note(L, f), marginTop: 4 }}>
                  {r.detail}
                </Rise>
              )}
            </div>
          ))}
        </div>
        {!L.vertical && n > 0 && (
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 52 }}>
            {count}
            {side.unit && <div style={label(L, f)}>{side.unit}</div>}
          </div>
        )}
      </div>
    );
  };
  return (
    <>
      {head && <SaidLine said={head} thenColor={f.ink2} style={{ ...title(L, f), position: 'absolute', left: L.x, right: L.W - L.right, top: L.top }} />}
      {L.vertical ? <div style={{ position: 'absolute', left: L.x, right: L.W - L.right, top }}>{sides.map(column)}</div> : sides.map(column)}
      {!L.vertical && <div style={{ position: 'absolute', left: boxes![1].left - gutter / 2, top, bottom: L.H - L.foot + L.size.note + 44, width: 2, background: f.ink, opacity: 0.12 }} />}
      <Foot L={L} f={f} text={foot} />
    </>
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

export const SCENES = { ledger: Ledger, figure: Figure, statement: Statement, bars: Bars, session: Session, command: Command, split: Split };

