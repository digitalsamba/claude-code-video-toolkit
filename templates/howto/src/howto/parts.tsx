/**
 * The pieces every scene shares: the mark and its track (the episode's one recurring object), the ground, the
 * entrance, the episode badge.
 *
 * The mark is the brand's own (brands/<name>/assets/mark.svg, through scripts/ident.py) when it has one. Without
 * one it is a prompt chevron with a caret. `fill` closes the chevron into a play triangle; the caret is the
 * playhead, `head` (0-1) along a track with a stop per chapter; `drawn` (0-1) is how much of the track exists.
 */
import React from 'react';
import { Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { MARK } from './ident';
import { type Field, type Layout, note, SANS } from './theme';

const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

/** 0-1 over [from, from + over] frames, eased in and out. */
export const ramp = (frame: number, from: number, over: number) =>
  interpolate(frame, [from, from + over], [0, 1], { ...clamp, easing: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2) });

/** Spring entrance: 0 before `at`, settles to 1 in about 10 frames. */
export const useEnter = (at: number) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < at) return 0;
  return Math.min(1, spring({ frame: frame - at + 1, fps, config: { damping: 200 }, durationInFrames: 10 }));
};

/** Rises 14 px into place on `at` (a cue). Nothing exits: the scene cuts. */
export const Rise: React.FC<{ at: number; style?: React.CSSProperties; children: React.ReactNode }> = ({ at, style, children }) => {
  const t = useEnter(at);
  return <div style={{ ...style, opacity: t, transform: `translateY(${(1 - t) * 14}px)` }}>{children}</div>;
};

/** 0-1 from `at`: an arrival, fast off the mark with a long tail (motion-craft, "Ease by role"). */
export const arrive = (frame: number, at: number, over = 14) => interpolate(frame, [at, at + over], [0, 1], { ...clamp, easing: Easing.out(Easing.exp) });

/**
 * A title's entrance: its words come in one after another, in place. With `blur` they sharpen as they come (the
 * opening, a chapter card); without, they just appear (a scene's title, a row's label). Nothing moves, so the
 * line can be read while it lands.
 */
export const Words: React.FC<{ text: string; at: number; gap?: number; blur?: number; inline?: boolean; style?: React.CSSProperties }> = ({
  text,
  at,
  gap = 2,
  blur = 0,
  inline,
  style,
}) => {
  const frame = useCurrentFrame();
  const words = text.split(' ');
  const Tag = inline ? 'span' : 'div';
  return (
    <Tag style={style}>
      {words.map((w, i) => {
        const t = interpolate(frame, [at + i * gap, at + i * gap + (blur ? 6 : 4)], [0, 1], clamp);
        return (
          <span key={i} style={{ display: 'inline-block', whiteSpace: 'pre', opacity: t, filter: blur && t < 1 ? `blur(${(1 - t) * blur}px)` : undefined }}>
            {w}
            {i < words.length - 1 ? ' ' : ''}
          </span>
        );
      })}
    </Tag>
  );
};

/** A value's entrance: it lands a little large and settles, from the edge it is set against. */
export const Punch: React.FC<{ at: number; origin?: string; from?: number; style?: React.CSSProperties; children: React.ReactNode }> = ({
  at,
  origin = 'left bottom',
  from = 1.16,
  style,
  children,
}) => {
  const frame = useCurrentFrame();
  const t = arrive(frame, at);
  const o = interpolate(frame, [at, at + 3], [0, 1], clamp);
  return <div style={{ ...style, opacity: o, transform: `scale(${from - (from - 1) * t})`, transformOrigin: origin }}>{children}</div>;
};

/**
 * A pen mark on whatever it is put inside (the parent needs `position: relative`): a loop round a value or a
 * stroke under a line, drawn in 9 frames from `at`. It marks the thing being said.
 */
export const Pen: React.FC<{ at: number; color: string; kind?: 'circle' | 'under'; padX?: number; padY?: number; width?: number }> = ({
  at,
  color,
  kind = 'circle',
  padX = 26,
  padY = 10,
  width = 6,
}) => {
  const frame = useCurrentFrame();
  const ref = React.useRef<HTMLDivElement>(null);
  const [size, setSize] = React.useState<[number, number] | null>(null);
  // the mark is drawn in pixels at the size of what it marks, so the stroke keeps one width
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (el && (!size || size[0] !== el.offsetWidth || size[1] !== el.offsetHeight)) setSize([el.offsetWidth, el.offsetHeight]);
  });
  const t = interpolate(frame, [at, at + 9], [0, 1], { ...clamp, easing: Easing.inOut(Easing.cubic) });
  const box: React.CSSProperties = kind === 'circle' ? { left: -padX, right: -padX, top: -padY, bottom: -padY } : { left: 0, right: 0, bottom: -padY - 10, height: 16 };
  const shape = kind === 'circle' ? PEN_CIRCLE : PEN_UNDER;
  const d = size ? shape.replace(/(-?[\d.]+) (-?[\d.]+)/g, (_, x, y) => `${((+x * size[0]) / 100).toFixed(1)} ${((+y * size[1]) / 100).toFixed(1)}`) : '';
  return (
    <div ref={ref} style={{ position: 'absolute', ...box, pointerEvents: 'none' }}>
      {size && frame >= at && (
        <svg width={size[0]} height={size[1]} style={{ overflow: 'visible', display: 'block' }}>
          <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - t} />
        </svg>
      )}
    </div>
  );
};
const PEN_CIRCLE = 'M 14 22 C 34 -6, 104 -4, 99 46 C 95 104, 6 106, 2 54 C -1 18, 44 -4, 90 12';
const PEN_UNDER = 'M 0 60 C 28 40, 66 78, 100 48';

/** The terminal's rows as hairlines: the template's ground, for type scenes only. */
export const Ground: React.FC<{ L: Layout; f: Field }> = ({ L, f }) => (
  <svg width={L.W} height={L.H} style={{ position: 'absolute', inset: 0 }}>
    {Array.from({ length: Math.ceil(L.H / L.row) }, (_, i) => (
      <line key={i} x1={0} x2={L.W} y1={(i + 0.5) * L.row} y2={(i + 0.5) * L.row} stroke={f.hair} strokeWidth={2} />
    ))}
  </svg>
);

/** Series, episode and toolkit version, bottom right: an out-of-date episode is easy to spot. */
export const Badge: React.FC<{ L: Layout; f: Field; text: string }> = ({ L, f, text }) => (
  <div style={{ ...note(L, f), position: 'absolute', right: L.W - L.right, top: L.foot - L.size.note, whiteSpace: 'nowrap' }}>{text}</div>
);

// ---------------------------------------------------------------- the mark and the track

export type RailState = {
  /** Chevron (0) to play triangle (1) */
  fill: number;
  /** How much of the track is drawn, 0-1 */
  drawn: number;
  /** Playhead position along the track, 0-1 */
  head: number;
  /** Caret visible (for blinking); the playhead never blinks once the track exists */
  caretOn: boolean;
  /** Stops lit ahead of the playhead (the opening names the chapters) */
  lit?: boolean[];
  /** A stop the playhead has just reached, and how large it is drawn (it lands large and settles) */
  pop?: { stop: number; scale: number };
};

export const railGeometry = (L: Layout) => {
  const s = L.vertical ? 84 : 96; // chevron height
  const w = MARK ? (s * MARK.box[2]) / MARK.box[3] : s * 0.62; // the brand's mark at the chevron's height, or the chevron
  const gap = s * 0.24;
  const cw = s * 0.5; // caret width
  const ch = Math.max(10, s * 0.14); // caret height
  const x0 = L.x + w + gap;
  const x1 = L.right;
  return { s, w, gap, cw, ch, x0, x1, len: x1 - x0 - cw };
};

/** Where chapter `i` (0-based) of `n` sits on the track, 0-1. */
export const stopAt = (i: number, n: number) => (i + 1) / (n + 1);

export const Rail: React.FC<{ L: Layout; f: Field; n: number; state: RailState }> = ({ L, f, n, state }) => {
  const g = railGeometry(L);
  const { s, w, cw, ch, x0, len } = g;
  const base = L.trackY; // bottom of the chevron and of the caret
  const top = base - s;
  const t = s * 0.2; // arm thickness where it meets the left edge
  const innerTip = (w - t * 1.55) * (1 - state.fill);
  const chevron = [
    [0, 0],
    [w, s / 2],
    [0, s],
    [0, s - t],
    [innerTip, s / 2],
    [0, t],
  ]
    .map(([px, py]) => `${L.x + px},${top + py}`)
    .join(' ');

  const lineY = base - ch / 2;
  const headX = x0 + state.head * len;
  const drawnTo = x0 + cw / 2 + state.drawn * (len);

  return (
    <svg width={L.W} height={L.H} style={{ position: 'absolute', inset: 0, overflow: 'visible' }}>
      {MARK ? (
        <g transform={`translate(${L.x} ${top}) scale(${s / MARK.box[3]}) translate(${-MARK.box[0]} ${-MARK.box[1]})`}>
          {MARK.parts.map((p) => (
            <path key={p.id} d={p.d} fill={p[f.mark] ?? f.ink} fillRule="evenodd" />
          ))}
        </g>
      ) : (
        <polygon points={chevron} fill={f.ink} />
      )}
      {state.drawn > 0 && (
        <line x1={x0 + cw / 2} x2={drawnTo} y1={lineY} y2={lineY} stroke={f.ink} strokeOpacity={0.35} strokeWidth={3} strokeLinecap="round" />
      )}
      {Array.from({ length: n }, (_, i) => {
        const p = stopAt(i, n);
        const cx = x0 + p * len + cw / 2;
        if (cx > drawnTo + 1) return null;
        const passed = state.head >= p - 0.001 || !!state.lit?.[i];
        const pop = Math.min(1, (drawnTo - cx) / 40 + 0.2);
        return (
          <circle
            key={i}
            cx={cx}
            cy={lineY}
            r={(L.vertical ? 9 : 10) * pop * (state.pop?.stop === i ? state.pop.scale : 1)}
            fill={passed ? f.accent : f.ground}
            stroke={passed ? f.accent : f.ink}
            strokeOpacity={passed ? 1 : 0.5}
            strokeWidth={3}
          />
        );
      })}
      {state.caretOn && <rect x={headX} y={base - ch} width={cw} height={ch} fill={state.drawn > 0 ? f.accent : f.ink} />}
    </svg>
  );
};

/** A word or two beside the playhead, under the track (a chapter's number, the link). */
export const UnderTrack: React.FC<{ L: Layout; f: Field; at: number; children: React.ReactNode }> = ({ L, f, at, children }) => (
  <Rise at={at} style={{ position: 'absolute', left: L.x, top: L.trackY + (L.vertical ? 56 : 52), fontFamily: SANS, fontSize: L.size.label, color: f.ink2 }}>
    {children}
  </Rise>
);
