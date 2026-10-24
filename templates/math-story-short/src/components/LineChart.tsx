import { useCurrentFrame } from 'remotion';
import { theme } from '../config/theme';

export interface Axis {
  min: number;
  max: number;
  scale?: 'linear' | 'log';
  ticks: number[];
  format: (v: number) => string;
  label?: string;
}

export interface Series {
  id: string;
  /** Value at x. Called ~`samples` times per frame, so keep it cheap (array lookups). */
  y: (x: number) => number;
  color: string;
  width?: number;
  opacity?: number;
  /** Soft neon glow under the stroke */
  glow?: boolean;
  /** Gradient fill between the line and y=0 */
  area?: boolean;
  /** Draw from axis min up to this x (animate it to "draw" the line) */
  revealTo: number;
  /** Pulsing end dot with a name tag */
  marker?: string;
  samples?: number;
}

export interface LineChartProps {
  width: number;
  height: number;
  x: Axis;
  y: Axis;
  series: Series[];
  /** Dashed reference line, e.g. y=0 or the 2.7c expectation */
  refLines?: { y: number; label?: string; color?: string }[];
}

const PAD = { left: 190, right: 60, top: 40, bottom: 84 };

/**
 * Animated SVG line chart for the editor panel. Domains are plain numbers, so
 * the caller animates zoom-outs by interpolating min/max per frame.
 */
export const LineChart: React.FC<LineChartProps> = ({ width, height, x, y, series, refLines = [] }) => {
  const frame = useCurrentFrame();
  const iw = width - PAD.left - PAD.right;
  const ih = height - PAD.top - PAD.bottom;
  const tx = (v: number) => {
    const t =
      x.scale === 'log'
        ? (Math.log10(v) - Math.log10(x.min)) / (Math.log10(x.max) - Math.log10(x.min))
        : (v - x.min) / (x.max - x.min);
    return PAD.left + t * iw;
  };
  const ty = (v: number) => PAD.top + (1 - (v - y.min) / (y.max - y.min)) * ih;
  const clampY = (v: number) => Math.max(PAD.top - 20, Math.min(PAD.top + ih + 20, v));
  // Ticks inside the plot are solid; ones pushed out by a zoom fade over 30px
  const tickOpacity = (px: number, lo: number, hi: number) =>
    Math.max(0, Math.min(1, (px - lo + 30) / 30, (hi - px + 30) / 30));

  const sample = (s: Series) => {
    const n = s.samples ?? 360;
    const end = Math.min(s.revealTo, x.max);
    if (end <= x.min) return [] as [number, number][];
    const pts: [number, number][] = [];
    for (let i = 0; i <= n; i++) {
      const xv =
        x.scale === 'log'
          ? Math.pow(10, Math.log10(x.min) + (i / n) * (Math.log10(end) - Math.log10(x.min)))
          : x.min + (i / n) * (end - x.min);
      pts.push([tx(xv), clampY(ty(s.y(xv)))]);
    }
    return pts;
  };

  const zeroPx = ty(Math.max(y.min, Math.min(y.max, 0)));

  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <defs>
        <clipPath id="plot-clip">
          <rect x={PAD.left - 4} y={PAD.top - 24} width={iw + 8} height={ih + 48} />
        </clipPath>
        {series.map((s) => (
          <linearGradient key={s.id} id={`area-${s.id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={s.color} stopOpacity={0.32} />
            <stop offset="100%" stopColor={s.color} stopOpacity={0} />
          </linearGradient>
        ))}
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="7" />
        </filter>
      </defs>

      {/* grid + y ticks */}
      {y.ticks.map((v) => {
        const py = ty(v);
        const o = tickOpacity(py, PAD.top, PAD.top + ih);
        if (o <= 0) return null;
        return (
          <g key={`y${v}`} opacity={o}>
            <line x1={PAD.left} x2={PAD.left + iw} y1={py} y2={py} stroke={theme.hairline} strokeWidth={1.5} />
            <text x={PAD.left - 24} y={py + 14} textAnchor="end" fontFamily={theme.fonts.sans} fontWeight={700} fontSize={40} fill={theme.muted}>
              {y.format(v)}
            </text>
          </g>
        );
      })}
      {y.label && (
        <text x={24} y={PAD.top - 4} fontFamily={theme.fonts.mono} fontSize={24} fill={theme.faint}>
          {y.label}
        </text>
      )}

      {/* x axis */}
      <line x1={PAD.left} x2={PAD.left + iw} y1={PAD.top + ih} y2={PAD.top + ih} stroke={theme.faint} strokeWidth={2} />
      {x.ticks.map((v) => {
        const px = tx(v);
        const o = tickOpacity(px, PAD.left, PAD.left + iw);
        if (o <= 0) return null;
        return (
          <text key={`x${v}`} x={px} y={PAD.top + ih + 58} opacity={o} textAnchor="middle" fontFamily={theme.fonts.sans} fontWeight={700} fontSize={40} fill={theme.muted}>
            {x.format(v)}
          </text>
        );
      })}
      {x.label && (
        <text x={PAD.left + iw / 2} y={PAD.top + ih + 88} textAnchor="middle" fontFamily={theme.fonts.mono} fontSize={24} fill={theme.faint}>
          {x.label}
        </text>
      )}

      {refLines.map((r, i) => {
        const py = ty(r.y);
        return (
          <g key={i}>
            <line x1={PAD.left} x2={PAD.left + iw} y1={py} y2={py} stroke={r.color ?? theme.cream} strokeOpacity={0.5} strokeWidth={2} strokeDasharray="10 10" strokeDashoffset={-frame * 0.6} />
            {r.label && (
              <text x={PAD.left + 14} y={py - 16} fontFamily={theme.fonts.sans} fontWeight={800} fontSize={36} fill={r.color ?? theme.cream} opacity={0.85}>
                {r.label}
              </text>
            )}
          </g>
        );
      })}

      <g clipPath="url(#plot-clip)">
        {series.map((s) => {
          const pts = sample(s);
          if (pts.length < 2) return null;
          const d = 'M' + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('L');
          const last = pts[pts.length - 1];
          return (
            <g key={s.id} opacity={s.opacity ?? 1}>
              {s.area && <path d={`${d}L${last[0]},${zeroPx}L${pts[0][0]},${zeroPx}Z`} fill={`url(#area-${s.id})`} />}
              {s.glow && <path d={d} fill="none" stroke={s.color} strokeWidth={(s.width ?? 4) * 2.5} opacity={0.55} filter="url(#glow)" strokeLinejoin="round" />}
              <path d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 4} strokeLinejoin="round" strokeLinecap="round" />
            </g>
          );
        })}
      </g>

      {series
        .filter((s) => s.marker)
        .map((s) => {
          const pts = sample(s);
          if (!pts.length) return null;
          const [mx, my] = pts[pts.length - 1];
          const ring = (frame % 36) / 36;
          const tagW = s.marker!.length * 24 + 32;
          // Flip the tag to the left of the dot when it would leave the panel
          const tagX = mx + 22 + tagW > width - 8 ? mx - 22 - tagW : mx + 22;
          return (
            <g key={`m-${s.id}`}>
              <circle cx={mx} cy={my} r={10 + ring * 22} fill="none" stroke={s.color} strokeWidth={3} opacity={1 - ring} />
              <circle cx={mx} cy={my} r={11} fill={theme.cream} stroke={s.color} strokeWidth={5} />
              <g transform={`translate(${tagX}, ${my - 26})`}>
                <rect width={tagW} height={52} rx={14} fill={s.color} />
                <text x={16} y={37} fontFamily={theme.fonts.sans} fontWeight={800} fontSize={34} fill={theme.cream}>
                  {s.marker}
                </text>
              </g>
            </g>
          );
        })}
    </svg>
  );
};
