import { useId } from 'react';

/** Pip positions for a standard die face (normalised 0-1). */
const PIPS: Record<number, [number, number][]> = {
  1: [[0.5, 0.5]],
  2: [[0.25, 0.25], [0.75, 0.75]],
  3: [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]],
  4: [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
  5: [[0.25, 0.25], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.75, 0.75]],
  6: [[0.25, 0.2], [0.75, 0.2], [0.25, 0.5], [0.75, 0.5], [0.25, 0.8], [0.75, 0.8]],
};
export const pips = (value: number): [number, number][] => PIPS[value] ?? [];

export type DieTone = 'ivory' | 'crimson' | 'navy';

const TONES: Record<DieTone, { a: string; b: string; edge: string; pip: string; pipHi: string; stroke: string; glow: string }> = {
  ivory: { a: '#fffaf0', b: '#f1dcb6', edge: '#b8955f', pip: '#16161a', pipHi: '#5a5a63', stroke: 'rgba(60,30,0,0.18)', glow: 'rgba(253,240,213,0.55)' },
  crimson: { a: '#ea4651', b: '#8e0b16', edge: '#3f0006', pip: '#fdf0d5', pipHi: '#ffffff', stroke: 'rgba(0,0,0,0.28)', glow: 'rgba(230,57,70,0.7)' },
  navy: { a: '#17648f', b: '#083551', edge: '#01192a', pip: '#a9cbe0', pipHi: '#e2f1fa', stroke: 'rgba(102,155,188,0.4)', glow: 'rgba(102,155,188,0.6)' },
};

export interface DieProps {
  value: number;
  size: number;
  tone?: DieTone;
  /** Soft glow radius in px (0 = none) */
  glow?: number;
  /** Fade the pips (for a "?" overlay) */
  pipOpacity?: number;
  style?: React.CSSProperties;
}

/**
 * The episode's subject as a physical object (see CLAUDE.md "Objects, not icons"):
 * copy this pattern for cards, coins, balls, chips.
 *
 * A chunky, physical die face: gradient body, a thick bottom edge, a gloss
 * line and domed pips. Three tones: ivory (the "real" die), crimson (the
 * fixed six / the answer), navy (unknown or neutral, matches the panels).
 */
export const Die: React.FC<DieProps> = ({ value, size, tone = 'ivory', glow = 0, pipOpacity = 1, style }) => {
  const id = useId().replace(/[^a-zA-Z0-9]/g, '');
  const t = TONES[tone];
  const r = 9.5;
  return (
    <svg
      width={size}
      height={size * 1.06}
      viewBox="0 0 100 106"
      style={{ overflow: 'visible', display: 'block', filter: glow > 0 ? `drop-shadow(0 0 ${glow}px ${t.glow})` : undefined, ...style }}
    >
      <defs>
        <linearGradient id={`${id}f`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={t.a} />
          <stop offset="100%" stopColor={t.b} />
        </linearGradient>
        <radialGradient id={`${id}p`} cx="0.35" cy="0.3" r="0.75">
          <stop offset="0%" stopColor={t.pipHi} />
          <stop offset="45%" stopColor={t.pip} />
          <stop offset="100%" stopColor={t.pip} />
        </radialGradient>
      </defs>
      <rect x={3} y={9} width={94} height={94} rx={21} fill={t.edge} />
      <rect x={3} y={3} width={94} height={94} rx={21} fill={`url(#${id}f)`} stroke={t.stroke} strokeWidth={1.5} />
      <path d="M16,9 Q50,3 84,9" stroke="#ffffff" strokeOpacity={0.55} strokeWidth={3} strokeLinecap="round" fill="none" />
      <g opacity={pipOpacity}>
        {pips(value).map(([px, py], i) => (
          <g key={i}>
            <circle cx={12 + px * 76} cy={13.5 + py * 76} r={r} fill="rgba(0,0,0,0.18)" />
            <circle cx={12 + px * 76} cy={12 + py * 76} r={r} fill={`url(#${id}p)`} />
          </g>
        ))}
      </g>
    </svg>
  );
};

/** Inline die glyph for use inside a sentence (vertical-align to the text). */
export const InlineDie: React.FC<{ value: number; size: number; tone?: DieTone }> = ({ value, size, tone = 'crimson' }) => (
  <span style={{ display: 'inline-block', verticalAlign: '-0.18em', margin: '0 0.12em' }}>
    <Die value={value} size={size} tone={tone} />
  </span>
);

