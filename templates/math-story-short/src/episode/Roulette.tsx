import { theme } from '../config/theme';
import type { Outcome } from './data';

// European wheel order, clockwise from the zero
const ORDER = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26];
const REDS = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);
export const pocketColor = (n: number): Outcome => (n === 0 ? 'green' : REDS.has(n) ? 'red' : 'black');
const FILL: Record<Outcome, string> = { red: '#c1121f', black: '#16181d', green: '#2a9d8f' };

export interface RouletteProps {
  size: number;
  /** Wheel rotation in degrees */
  rotation: number;
  /** Ball position in degrees (screen space, 0 = top) */
  ballAngle: number;
  /** 0 = on the outer track, 1 = settled in the pockets */
  ballDrop: number;
  /** Dim every pocket except this colour; null shows all */
  highlight?: Outcome | null;
  /** 0-1: how strongly the highlight dims the rest */
  highlightAmount?: number;
}

const polar = (r: number, deg: number): [number, number] => {
  const a = ((deg - 90) * Math.PI) / 180;
  return [r * Math.cos(a), r * Math.sin(a)];
};

/** Brand-coloured European roulette wheel with depth, numbers and a ball. */
export const Roulette: React.FC<RouletteProps> = ({ size, rotation, ballAngle, ballDrop, highlight = null, highlightAmount = 1 }) => {
  const R = 500;
  const step = 360 / 37;
  const pockets = ORDER.map((n, i) => {
    const a0 = i * step - step / 2;
    const a1 = a0 + step;
    const [x0, y0] = polar(340, a0);
    const [x1, y1] = polar(340, a1);
    const [x2, y2] = polar(430, a1);
    const [x3, y3] = polar(430, a0);
    const [tx, ty] = polar(392, i * step);
    const col = pocketColor(n);
    const dim = highlight && col !== highlight ? 1 - 0.75 * highlightAmount : 1;
    const glow = highlight && col === highlight ? highlightAmount : 0;
    return { n, i, col, dim, glow, d: `M${x0},${y0} A340,340 0 0 1 ${x1},${y1} L${x2},${y2} A430,430 0 0 0 ${x3},${y3} Z`, tx, ty };
  });
  const ballR = 466 - ballDrop * 80;
  const [bx, by] = polar(ballR, ballAngle);

  return (
    <svg width={size} height={size} viewBox={`${-R} ${-R} ${R * 2} ${R * 2}`} style={{ overflow: 'visible' }}>
      <defs>
        <radialGradient id="rim" cx="0.5" cy="0.4" r="0.6">
          <stop offset="80%" stopColor="#0f4a6e" />
          <stop offset="100%" stopColor="#022a42" />
        </radialGradient>
        <radialGradient id="bowl" cx="0.45" cy="0.4" r="0.6">
          <stop offset="0%" stopColor="#0d4a70" />
          <stop offset="100%" stopColor="#012238" />
        </radialGradient>
        <radialGradient id="cone" cx="0.4" cy="0.35" r="0.7">
          <stop offset="0%" stopColor="#1b5d86" />
          <stop offset="100%" stopColor="#03263d" />
        </radialGradient>
        <filter id="wheel-shadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="30" stdDeviation="30" floodColor="#000a14" floodOpacity="0.6" />
        </filter>
        <filter id="pocket-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="10" />
        </filter>
      </defs>

      <circle r={490} fill="url(#rim)" filter="url(#wheel-shadow)" />
      <circle r={490} fill="none" stroke="rgba(253,240,213,0.18)" strokeWidth={3} />
      <circle r={480} fill="none" stroke="rgba(102,155,188,0.25)" strokeWidth={1.5} />
      {/* ball track */}
      <circle r={456} fill="none" stroke="rgba(0,20,34,0.55)" strokeWidth={34} />

      <g transform={`rotate(${rotation})`}>
        {pockets.filter((p) => p.glow > 0).map((p) => (
          <path key={`g${p.i}`} d={p.d} fill={FILL[p.col]} opacity={0.9 * p.glow} filter="url(#pocket-glow)" />
        ))}
        {pockets.map((p) => (
          <path key={p.i} d={p.d} fill={FILL[p.col]} opacity={p.dim} stroke="rgba(253,240,213,0.35)" strokeWidth={2} />
        ))}
        {pockets.map((p) => (
          <text
            key={`t${p.i}`}
            x={p.tx}
            y={p.ty}
            transform={`rotate(${p.i * step}, ${p.tx}, ${p.ty})`}
            textAnchor="middle"
            dominantBaseline="central"
            fontFamily={theme.fonts.mono}
            fontWeight={700}
            fontSize={30}
            fill={theme.cream}
            opacity={0.9 * p.dim}
          >
            {p.n}
          </text>
        ))}
        <circle r={340} fill="url(#bowl)" stroke="rgba(253,240,213,0.25)" strokeWidth={3} />
        {Array.from({ length: 37 }, (_, i) => {
          const [x0, y0] = polar(250, i * step - step / 2);
          const [x1, y1] = polar(338, i * step - step / 2);
          return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke="rgba(102,155,188,0.18)" strokeWidth={2} />;
        })}
        <circle r={230} fill="url(#cone)" stroke="rgba(102,155,188,0.35)" strokeWidth={2} />
        {[0, 90, 180, 270].map((a) => {
          const [x, y] = polar(190, a);
          return (
            <g key={a}>
              <line x1={0} y1={0} x2={x} y2={y} stroke={theme.cream} strokeWidth={14} strokeLinecap="round" opacity={0.9} />
              <circle cx={x} cy={y} r={18} fill={theme.cream} />
            </g>
          );
        })}
        <circle r={56} fill={theme.cream} />
        <circle r={30} fill={theme.primary} />
      </g>

      <circle cx={bx + 6} cy={by + 10} r={22} fill="rgba(0,10,20,0.45)" />
      <circle cx={bx} cy={by} r={22} fill="#fffaf0" />
      <circle cx={bx - 7} cy={by - 7} r={7} fill="#ffffff" />
    </svg>
  );
};
