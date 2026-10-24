import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';

const W = 960;
const H = 560;
const PIVOT = { x: 480, y: 120 };
const ARM = 340;
const HANG = 330;
const CHIP_H = 12;

type ChipColor = 'red' | 'black' | 'green';
const FACE: Record<ChipColor, [string, string]> = {
  red: ['#d61f2c', '#8f0a14'],
  black: ['#2a2d35', '#0d0e11'],
  green: ['#35b5a5', '#1d7a6e'],
};

export interface BalanceScaleProps {
  /** Frame the left (win) chips start dropping */
  winsAt: number;
  /** Frame the right (loss) chips start dropping */
  lossesAt: number;
  /** Frame the net result pops above the pivot */
  netAt: number;
}

/**
 * 18 winning chips against 19 losing ones on a balance: the beam dips toward
 * the wins, then the losses land and it tips the other way. The odd green chip
 * on top of the loss stack is the zero.
 */
export const BalanceScale: React.FC<BalanceScaleProps> = ({ winsAt, lossesAt, netAt }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = spring({ frame, fps, config: { damping: 18, stiffness: 100 } });

  const left: ChipColor[] = Array(18).fill('red');
  const right: ChipColor[] = [...Array(18).fill('black'), 'green'];
  const dropped = (start: number, n: number) => Array.from({ length: n }, (_, i) => spring({ frame: frame - start - i * 2, fps, config: { damping: 15, stiffness: 220 } }));
  const leftDrop = dropped(winsAt, left.length);
  const rightDrop = dropped(lossesAt, right.length);
  const weightL = leftDrop.reduce((a, b) => a + b, 0);
  const weightR = rightDrop.reduce((a, b) => a + b, 0);
  // Angle follows the weight difference; one extra chip still tips it visibly
  const diff = weightR - weightL;
  const target = Math.max(-11, Math.min(11, diff * 0.9 + Math.tanh(diff * 2) * 5));
  const angle = target * spring({ frame: frame - winsAt, fps, config: { damping: 8, stiffness: 60 } });
  const rad = (angle * Math.PI) / 180;
  const end = (side: -1 | 1) => ({ x: PIVOT.x + side * ARM * Math.cos(rad), y: PIVOT.y + side * ARM * Math.sin(rad) });
  const net = spring({ frame: frame - netAt, fps, config: { damping: 10, stiffness: 180 } });
  const zeroGlow = interpolate(frame, [lossesAt + 40, lossesAt + 52], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

  const pan = (side: -1 | 1, chips: ChipColor[], drops: number[], label: string, value: string, valueColor: string) => {
    const e = end(side);
    const panY = e.y + HANG;
    return (
      <g>
        <line x1={e.x} y1={e.y} x2={e.x - 128} y2={panY} stroke="rgba(253,240,213,0.45)" strokeWidth={3} />
        <line x1={e.x} y1={e.y} x2={e.x + 128} y2={panY} stroke="rgba(253,240,213,0.45)" strokeWidth={3} />
        {chips.map((c, i) => {
          const d = drops[i];
          if (d <= 0.001) return null;
          const cy = panY - 10 - i * CHIP_H - (1 - Math.min(1, d)) * 420;
          const [top, side2] = FACE[c];
          const glow = c === 'green' ? zeroGlow : 0;
          return (
            <g key={i} opacity={Math.min(1, d * 3)}>
              {glow > 0 && <ellipse cx={e.x} cy={cy} rx={70} ry={26} fill="#35b5a5" opacity={0.5 * glow} filter="url(#chip-glow)" />}
              <rect x={e.x - 56} y={cy} width={112} height={CHIP_H} fill={side2} />
              <ellipse cx={e.x} cy={cy + CHIP_H} rx={56} ry={16} fill={side2} />
              <ellipse cx={e.x} cy={cy} rx={56} ry={16} fill={top} stroke="rgba(253,240,213,0.35)" strokeWidth={1.5} />
              <ellipse cx={e.x} cy={cy} rx={36} ry={10} fill="none" stroke="rgba(253,240,213,0.5)" strokeWidth={2} strokeDasharray="6 6" />
            </g>
          );
        })}
        <path d={`M${e.x - 140},${panY} Q${e.x},${panY + 46} ${e.x + 140},${panY} Z`} fill="#0f4a6e" stroke="rgba(253,240,213,0.4)" strokeWidth={2.5} />
        <text x={e.x} y={panY + 92} textAnchor="middle" fontFamily={theme.fonts.mono} fontSize={30} fill={theme.muted}>
          {label}
        </text>
        <text x={e.x} y={panY + 150} textAnchor="middle" fontFamily={theme.fonts.sans} fontWeight={800} fontSize={58} fill={valueColor}>
          {value}
        </text>
      </g>
    );
  };

  const l = end(-1);
  const r = end(1);
  return (
    <svg width={W} height={H + 120} viewBox={`0 0 ${W} ${H + 120}`} style={{ overflow: 'visible', opacity: enter }}>
      <defs>
        <filter id="chip-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="12" />
        </filter>
        <linearGradient id="post" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#0b3c5c" />
          <stop offset="50%" stopColor="#1b5d86" />
          <stop offset="100%" stopColor="#0b3c5c" />
        </linearGradient>
      </defs>
      <path d={`M${PIVOT.x - 90},${H} L${PIVOT.x - 30},${PIVOT.y + 30} L${PIVOT.x + 30},${PIVOT.y + 30} L${PIVOT.x + 90},${H} Z`} fill="url(#post)" stroke="rgba(102,155,188,0.4)" strokeWidth={2} />
      <rect x={PIVOT.x - 150} y={H} width={300} height={22} rx={11} fill="#0b3c5c" stroke="rgba(102,155,188,0.4)" strokeWidth={2} />
      {pan(-1, left, leftDrop, '18 wins', '+$18', theme.cream)}
      {pan(1, right, rightDrop, '19 losses', '−$19', '#ff5a63')}
      <line x1={l.x} y1={l.y} x2={r.x} y2={r.y} stroke={theme.cream} strokeWidth={14} strokeLinecap="round" />
      <circle cx={PIVOT.x} cy={PIVOT.y} r={24} fill={theme.primary} stroke={theme.cream} strokeWidth={5} />
      {net > 0.01 && (
        <g transform={`translate(${PIVOT.x}, ${PIVOT.y - 70}) scale(${net})`}>
          <rect x={-110} y={-52} width={220} height={92} rx={22} fill={theme.primary} />
          <text x={0} y={18} textAnchor="middle" fontFamily={theme.fonts.sans} fontWeight={900} fontSize={60} fill={theme.cream}>
            −$1
          </text>
        </g>
      )}
    </svg>
  );
};
