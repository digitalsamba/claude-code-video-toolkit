// Chalk on the board: lettering that writes itself left to right, strokes that draw on, and the shapes a
// lecturer sketches. Every time is a composition frame (the board is one long scene), usually from q().
// The sounds (public/audio/sfx/chalk-write2.mp3, chalk-tap.mp3, wipe.mp3) ship with templates/lecture, all CC0.
import React from 'react';
import { Audio, Sequence, interpolate, staticFile, useCurrentFrame, Easing } from 'remotion';
import { loadFont } from '@remotion/google-fonts/Kalam';

export const { fontFamily: HAND } = loadFont('normal', { weights: ['400', '700'], subsets: ['latin'] });

export const WHITE = '#F3F0E6';
export const YELLOW = '#F4D26A';
/** The five sizes on the board. */
export const SIZE = { xl: 96, l: 64, m: 46, s: 34, xs: 28 } as const;

/** One SVG filter gives everything on the board a chalk grain and a slightly unsteady edge. */
export const ChalkDefs: React.FC = () => (
  <svg width="0" height="0" style={{ position: 'absolute' }}>
    <defs>
      <filter id="chalkFx" x="-2%" y="-2%" width="104%" height="104%" colorInterpolationFilters="sRGB">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3" result="n" />
        <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -1.35 1.55" result="holes" />
        <feComposite in="SourceGraphic" in2="holes" operator="in" result="g" />
        <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="8" result="w" />
        <feDisplacementMap in="g" in2="w" scale="2.6" xChannelSelector="R" yChannelSelector="G" />
      </filter>
    </defs>
  </svg>
);

const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

/** The scratch of chalk for as long as something is being written. */
const Scratch: React.FC<{ at: number; dur: number; seed: number }> = ({ at, dur, seed }) => {
  if (dur < 9) return null;
  const takes = [9, 110, 40];
  return (
    <Sequence from={at} durationInFrames={Math.min(dur, 84)} layout="none">
      <Audio src={staticFile('audio/sfx/chalk-write2.mp3')} trimBefore={takes[seed % 3]} volume={0.16} />
    </Sequence>
  );
};

export type ChalkProps = {
  x: number;
  y: number;
  /** Frame the writing starts. */
  at: number;
  /** Frames it takes (default: by its length). */
  dur?: number;
  size?: number;
  color?: string;
  weight?: 400 | 700;
  /** Degrees of tilt, for an aside. */
  tilt?: number;
  width?: number;
  quiet?: boolean;
  children: React.ReactNode;
};

const textOf = (n: React.ReactNode): string =>
  typeof n === 'string' || typeof n === 'number' ? String(n)
    : Array.isArray(n) ? n.map(textOf).join('')
      : React.isValidElement(n) ? textOf((n.props as { children?: React.ReactNode }).children) : '';

/** A line of chalk lettering, written from its left edge. */
export const Chalk: React.FC<ChalkProps> = ({ x, y, at, dur, size = SIZE.m, color = WHITE, weight = 700, tilt = 0, width, quiet, children }) => {
  const frame = useCurrentFrame();
  const len = textOf(children).length;
  const d = dur ?? Math.max(8, Math.round(len * 1.5));
  if (frame < at) return null;
  const p = interpolate(frame, [at, at + d], [0, 1], clamp);
  return (
    <>
      <div
        style={{
          position: 'absolute', left: x, top: y, width, fontFamily: HAND, fontSize: size, fontWeight: weight, color,
          lineHeight: 1.12, whiteSpace: width ? 'normal' : 'nowrap', opacity: 0.95,
          transform: tilt ? `rotate(${tilt}deg)` : undefined, transformOrigin: '0 50%',
          clipPath: p >= 1 ? undefined : `inset(-12% ${(1 - p) * 100}% -12% -3%)`,
        }}
      >
        {children}
      </div>
      {quiet ? null : <Scratch at={at} dur={d} seed={Math.round(x + y)} />}
    </>
  );
};

/** The accent chalk, for the figure being said. */
export const Y: React.FC<{ children: React.ReactNode }> = ({ children }) => <span style={{ color: YELLOW }}>{children}</span>;

export type StrokeProps = {
  d: string;
  at: number;
  dur?: number;
  width?: number;
  color?: string;
  /** A dotted line: revealed left to right between these x positions instead of drawn along its length. */
  dotted?: [number, number];
  fill?: string;
  quiet?: boolean;
};

/** A chalk stroke that draws on. */
export const Stroke: React.FC<StrokeProps> = ({ d, at, dur = 12, width = 5, color = WHITE, dotted, fill, quiet = true }) => {
  const frame = useCurrentFrame();
  if (frame < at) return null;
  const p = interpolate(frame, [at, at + dur], [0, 1], { ...clamp, easing: Easing.inOut(Easing.quad) });
  const id = `c${Math.round(at)}_${Math.round(dotted ? dotted[0] : 0)}_${d.length}`;
  return (
    <>
      <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible' }}>
        {dotted ? (
          <>
            <clipPath id={id}>
              <rect x={dotted[0] - 10} y={0} width={(dotted[1] - dotted[0] + 20) * p} height={1080} />
            </clipPath>
            <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={`1 ${width * 3}`}
              clipPath={`url(#${id})`} opacity={0.95} />
          </>
        ) : (
          <path d={d} fill={fill && p >= 1 ? fill : 'none'} stroke={color} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round"
            pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - p} opacity={0.95} />
        )}
      </svg>
      {quiet ? null : <Scratch at={at} dur={dur} seed={d.length} />}
    </>
  );
};

/** A tap of the chalk: a tick, a dot, a full stop. */
export const Tap: React.FC<{ at: number; volume?: number }> = ({ at, volume = 0.22 }) => (
  <Sequence from={at} durationInFrames={15} layout="none">
    <Audio src={staticFile('audio/sfx/chalk-tap.mp3')} volume={volume} />
  </Sequence>
);

// ---------------------------------------------------------------- shapes (paths with a hand's wobble)

const rnd = (seed: number) => {
  let s = (seed * 9301 + 49297) % 233280;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280 - 0.5;
  };
};

export const line = (x1: number, y1: number, x2: number, y2: number, seed = 1, wob = 4) => {
  const r = rnd(seed);
  const len = Math.hypot(x2 - x1, y2 - y1);
  const k = Math.min(1, len / 300) * wob;
  const mx = (x1 + x2) / 2 + r() * 2 * k;
  const my = (y1 + y2) / 2 + r() * 2 * k;
  return `M${x1} ${y1} Q${mx.toFixed(1)} ${my.toFixed(1)} ${x2} ${y2}`;
};

export const box = (x: number, y: number, w: number, h: number, seed = 1) => {
  const r = rnd(seed);
  const j = () => r() * 7;
  const p = [[x + j(), y + j()], [x + w + j(), y + j()], [x + w + j(), y + h + j()], [x + j(), y + h + j()]];
  return `M${p[0][0].toFixed(1)} ${p[0][1].toFixed(1)} L${p[1][0].toFixed(1)} ${p[1][1].toFixed(1)} L${p[2][0].toFixed(1)} ${p[2][1].toFixed(1)} `
    + `L${p[3][0].toFixed(1)} ${p[3][1].toFixed(1)} L${(p[0][0] - 3).toFixed(1)} ${(p[0][1] - 5).toFixed(1)}`;
};

export const arrow = (x1: number, y1: number, x2: number, y2: number, seed = 1, head = 16) => {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const h = (da: number) => `L${(x2 - head * Math.cos(a + da)).toFixed(1)} ${(y2 - head * Math.sin(a + da)).toFixed(1)}`;
  return `${line(x1, y1, x2, y2, seed, 3)} ${h(0.5)} M${x2} ${y2} ${h(-0.5)}`;
};

export const tick = (x: number, y: number, s = 34) =>
  `M${x} ${y + s * 0.55} L${x + s * 0.34} ${y + s * 0.92} L${x + s} ${y}`;

export const cross = (x: number, y: number, s = 30) => `M${x} ${y} L${x + s} ${y + s} M${x + s} ${y} L${x} ${y + s}`;

export const ring = (cx: number, cy: number, rx: number, ry: number) =>
  `M${cx - rx} ${cy + 3} C${cx - rx} ${cy - ry * 1.3} ${cx + rx} ${cy - ry * 1.3} ${cx + rx} ${cy} `
  + `C${cx + rx} ${cy + ry * 1.3} ${cx - rx} ${cy + ry * 1.35} ${cx - rx - 4} ${cy - 6}`;

/** An underline under lettering at (x, y) of a given size and width. */
export const under = (x: number, y: number, size: number, w: number, seed = 1) => line(x - 4, y + size * 1.16, x + w, y + size * 1.13, seed, 5);

// ---------------------------------------------------------------- a segment's stretch on the board

/** What one segment puts on the board: there from `from`, wiped left to right from `to` (the eraser takes 14 frames). */
export const Seg: React.FC<{ from: number; to: number; children: React.ReactNode }> = ({ from, to, children }) => {
  const frame = useCurrentFrame();
  if (frame < from || frame > to + 15) return null;
  const w = interpolate(frame, [to, to + 14], [0, 100], { ...clamp, easing: Easing.inOut(Easing.quad) });
  return (
    <>
      <div style={{ position: 'absolute', inset: 0, clipPath: w > 0 ? `inset(0 0 0 ${w}%)` : undefined }}>{children}</div>
      {w > 0 && w < 100 ? (
        <div style={{ position: 'absolute', left: `${w - 9}%`, top: 150, width: '10%', height: 860, borderRadius: 60,
          background: 'linear-gradient(90deg, rgba(243,240,230,0) 0%, rgba(243,240,230,0.2) 70%, rgba(243,240,230,0.05) 100%)' }} />
      ) : null}
      <Sequence from={to} durationInFrames={22} layout="none">
        <Audio src={staticFile('audio/sfx/wipe.mp3')} volume={0.2} />
      </Sequence>
    </>
  );
};
