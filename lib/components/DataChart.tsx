/**
 * Data charts drawn by Remotion
 *
 * Line and bar charts rendered as SVG, frame by frame, instead of a pre-rendered chart
 * video. The chart takes the brand's colours and font, its labels are real text (so
 * captions and `qc.py --layout` see them), and every move is a frame number, so a chart
 * can follow the narration: draw each stretch of data while it's being talked about,
 * light a bar on the word that names it. Take those frames from `cue()` (./cues.ts).
 *
 * @example
 * ```tsx
 * import { LineChart, BarChart, chartTheme, lineHead, voice } from '../../../lib/components';
 *
 * const theme = chartTheme(brand.colors, brand.fonts);          // dark by default
 * const keys = [                                               // where the line has drawn to
 *   { frame: cue('history'), x: day('2026-01-20') },
 *   { frame: cue('weeks', { edge: 'end' }), x: day('2026-02-28') },
 *   { frame: cue('today', { edge: 'end' }), x: day('2026-11-04') },
 * ];
 * const head = lineHead(points, frame, keys);                  // { x, y } for a counter
 *
 * <LineChart points={points} keys={keys} width={1760} height={680} theme={theme}
 *   xTicks={months} bands={[{ from, to, label: '+402 in 3 days', showAt: cue('four hundred') }]} />
 *
 * <BarChart bars={[{ label: 'Mar', value: 512, highlightAt: cue('march') }, ...]}
 *   width={1680} height={680} theme={theme} start={10} />
 * ```
 *
 * Worked example: examples/data-chart-stars (narrated, cued, data fetched live).
 */

import React, { useId, useMemo } from 'react';
import { Easing, interpolate, interpolateColors, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { hexToRgba } from './utils';

// ---------------------------------------------------------------- theme

export type ChartTheme = {
  /** The data colour: the line, a highlighted bar. */
  primary: string;
  /** Annotation bands. */
  accent: string;
  /** Titles and highlighted values. */
  text: string;
  /** Axis labels, unhighlighted bars. */
  muted: string;
  grid: string;
  /** The background the chart sits on (the head dot's ring). */
  bg: string;
  font: string;
  /**
   * Unhighlighted data, as one solid neutral step from the ground (e.g. #d9d4c7 on bone). When set, bars
   * use it at full strength instead of `muted` at partial opacity, which over a coloured ground mixes
   * into a colour nobody chose (45% black over lime is olive).
   */
  other?: string;
};

type ColorsLike = {
  primary: string;
  accent?: string;
  textDark: string;
  textLight: string;
  bgLight: string;
  bgDark: string;
};

/**
 * A chart theme from a brand's colours (brand.json `colors`, or a template theme's).
 * `dark` draws on `bgDark`, `light` on `bgLight`.
 */
export const chartTheme = (
  colors: ColorsLike,
  fonts: { primary: string },
  mode: 'dark' | 'light' = 'dark',
): ChartTheme => {
  const dark = mode === 'dark';
  return {
    primary: colors.primary,
    accent: colors.accent ?? colors.primary,
    text: dark ? colors.bgLight : colors.textDark,
    muted: dark ? '#94a3b8' : colors.textLight,
    grid: dark ? 'rgba(148, 163, 184, 0.16)' : hexToRgba(colors.textDark, 0.1),
    bg: dark ? colors.bgDark : colors.bgLight,
    font: fonts.primary,
  };
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const tabular: React.CSSProperties = { fontVariantNumeric: 'tabular-nums' };
const defaultFormat = (v: number) => Math.round(v).toLocaleString('en-GB');

// ---------------------------------------------------------------- line chart

export type LinePoint = { x: number; y: number };

/**
 * Where the line has drawn to: `keys` pins data x positions to frames, and the head
 * eases between them, so the narration decides how fast each stretch is drawn. Two keys
 * (first point at frame A, last at frame B) give a plain draw-on. Frames must increase.
 */
export type HeadKey = { frame: number; x: number };

/** The drawing head at a frame: data x, the y interpolated there, and 0-1 progress. */
export const lineHead = (points: LinePoint[], frame: number, keys: HeadKey[]) => {
  const x = interpolate(frame, keys.map((k) => k.frame), keys.map((k) => k.x), {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.cubic),
  });
  const x0 = points[0].x;
  const x1 = points[points.length - 1].x;
  const progress = frame < keys[0].frame ? 0 : (x - x0) / (x1 - x0 || 1);
  const i = points.findIndex((pt) => pt.x >= x);
  if (i <= 0) return { x, y: points[0].y, progress };
  const a = points[i - 1];
  const b = points[i];
  return { x, y: a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x || 1), progress };
};

/** First frame the head reaches data x `x` (for timing your own annotations to the line). */
export const headFrameAt = (keys: HeadKey[], x: number) => {
  for (let k = 1; k < keys.length; k++) {
    const a = keys[k - 1];
    const b = keys[k];
    if (b.x < x) continue;
    if (a.x >= x) return a.frame;
    for (let f = a.frame; f <= b.frame; f++) {
      const t = Easing.inOut(Easing.cubic)((f - a.frame) / (b.frame - a.frame || 1));
      if (a.x + (b.x - a.x) * t >= x) return f;
    }
    return b.frame;
  }
  return keys[keys.length - 1].frame;
};

/** A highlighted x span. It appears when the head passes `to`, or at `showAt` (a cue). */
export type ChartBand = { from: number; to: number; label: string; showAt?: number };

/** One line of a multi-series chart. The first series' x range sets the axis. */
export type LineSeries = {
  name: string;
  points: LinePoint[];
  /** Defaults: the theme's primary for the first series, accent for the second, muted after. */
  color?: string;
  /** Shade under the line (default: first series only). */
  fill?: boolean;
};

export type LineChartProps = {
  /** A single line. Use `series` instead for several (with a legend). */
  points?: LinePoint[];
  series?: LineSeries[];
  keys: HeadKey[];
  width: number;
  height: number;
  theme: ChartTheme;
  /** x-axis labels at data x positions. Labels that would overlap the previous one are dropped. */
  xTicks?: { x: number; label: string }[];
  bands?: ChartBand[];
  /** The y axis grows with the drawn data (default) instead of starting at the final max. */
  followData?: boolean;
  /** Headroom above the data, as a multiple of the max shown (default 1.25). */
  headroom?: number;
  /** y-axis label text (default: grouped integers). */
  formatY?: (v: number) => string;
  fontSize?: number;
  /** Show the series legend (default: when there is more than one series). */
  legend?: boolean;
  /** Shading under filled lines: a fading gradient (default), a flat tint (`areaOpacity`), or none. */
  area?: 'gradient' | 'flat' | 'none';
  /** Opacity of a flat area (default 0.12). */
  areaOpacity?: number;
};

// 1-2-5 steps, from 0.01 up to 5e7.
const STEPS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7].flatMap((k) => [1, 2, 5].map((m) => m * 10 ** k));

export const LineChart: React.FC<LineChartProps> = ({
  points, series, keys, width, height, theme, xTicks = [], bands = [], followData = true, headroom = 1.25,
  formatY = defaultFormat, fontSize = 28, legend, area = 'gradient', areaOpacity = 0.12,
}) => {
  const palette = [theme.primary, theme.accent, theme.muted];
  const lines = (series ?? [{ name: '', points: points ?? [] }]).map((l, i) => ({
    ...l, color: l.color ?? palette[Math.min(i, palette.length - 1)], fill: l.fill ?? i === 0,
  }));
  if (!lines.length || !lines[0].points.length) throw new Error('LineChart: no points (pass `points` or `series`)');
  const base = lines[0].points;
  const showLegend = legend ?? lines.length > 1;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const id = useId().replace(/[^a-zA-Z0-9]/g, '');
  // The legend sits in its own band above the plot, clear of the data and of band labels.
  const m = { top: showLegend ? fontSize * 2.4 : 30, right: 40, bottom: fontSize * 2.2, left: fontSize * 3.6 };
  const w = width - m.left - m.right;
  const h = height - m.top - m.bottom;

  const head = lineHead(base, frame, keys);
  const heads = lines.map((l) => lineHead(l.points, frame, keys));
  const all = lines.flatMap((l) => l.points);
  const finalMax = Math.max(...all.map((p) => p.y));
  const shownMax = followData
    ? Math.max(...all.filter((p) => p.x <= head.x).map((p) => p.y), ...heads.map((hd) => hd.y))
    : finalMax;
  const yMax = Math.max(shownMax * headroom, finalMax * 0.02, 1e-9);

  const x0 = base[0].x;
  const x1 = base[base.length - 1].x;
  const sx = (x: number) => ((x - x0) / (x1 - x0 || 1)) * w;
  const sy = (y: number) => h - (y / yMax) * h;

  // About five gridlines at a 1-2-5 step; as the axis rescales, one step's lines
  // crossfade into the next over a short window instead of popping.
  const target = yMax / 4.5;
  const si = Math.max(0, STEPS.findIndex((s, j) => s <= target && target < STEPS[j + 1]));
  const f = Math.log(target / STEPS[si]) / Math.log(STEPS[si + 1] / STEPS[si]);
  const k = clamp01((f - 0.45) / 0.1); // narrow, so an axis at rest is rarely caught mid-fade
  const fade = k * k * (3 - 2 * k);
  const ticks = new Map<number, number>();
  for (const [s, o] of [[STEPS[si], 1 - fade], [STEPS[si + 1], fade]] as const) {
    if (o <= 0.01) continue;
    for (let n = 0; n * s <= yMax; n++) {
      const v = Number((n * s).toPrecision(12));
      ticks.set(v, Math.max(ticks.get(v) ?? 0, o));
    }
  }

  // Drop x labels that would collide with the one before (rough width: 0.6em a character).
  const shownXTicks = useMemo(() => {
    const out: { x: number; label: string }[] = [];
    let lastRight = -Infinity;
    for (const t of xTicks) {
      const half = (t.label.length * fontSize * 0.6) / 2;
      const px = sx(t.x);
      if (px - half < lastRight + fontSize * 0.5) continue;
      out.push(t);
      lastRight = px + half;
    }
    return out;
  }, [xTicks, w, fontSize]);

  const bandFrames = useMemo(() => bands.map((b) => b.showAt ?? headFrameAt(keys, b.to)), [bands, keys]);

  const paths = lines.map((l) => {
    const d = l.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
    const last = l.points[l.points.length - 1].x;
    return { d, area: `${d} L${sx(last).toFixed(1)},${h} L${sx(l.points[0].x).toFixed(1)},${h} Z` };
  });
  const clipW = sx(head.x);
  const axisIn = spring({ frame: frame - (keys[0].frame - 20), fps, config: { damping: 200 } });

  return (
    <svg width={width} height={height} style={{ fontFamily: theme.font, overflow: 'visible' }}>
      <defs>
        <clipPath id={`reveal-${id}`}>
          <rect x={-2} y={-m.top} width={clipW + 2} height={h + m.top + 2} />
        </clipPath>
        {lines.map((l, i) => (
          <linearGradient key={i} id={`fill-${id}-${i}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={l.color} stopOpacity={0.35} />
            <stop offset="100%" stopColor={l.color} stopOpacity={0} />
          </linearGradient>
        ))}
      </defs>
      <g transform={`translate(${m.left},${m.top})`}>
        {[...ticks].filter(([, o]) => o * axisIn > 0.02).map(([v, o]) => (
          <g key={v} opacity={o * axisIn} transform={`translate(0,${sy(v)})`}>
            <line x1={0} x2={w} stroke={theme.grid} strokeWidth={v === 0 ? 2 : 1} />
            <text x={-fontSize * 0.7} dy="0.35em" textAnchor="end" fill={theme.muted} fontSize={fontSize} style={tabular}>
              {formatY(v)}
            </text>
          </g>
        ))}
        {axisIn > 0.02 && shownXTicks.map((t) => (
          <text key={t.x} x={sx(t.x)} y={h + fontSize * 1.6} textAnchor="middle" fill={theme.muted}
            fontSize={fontSize} opacity={axisIn}>
            {t.label}
          </text>
        ))}

        {bands.map((b, i) => {
          const at = bandFrames[i];
          const t = frame >= at ? spring({ frame: frame - at, fps, config: { damping: 18 } }) : 0;
          const bx = sx(b.from);
          const bw = Math.max(4, sx(b.to) - bx);
          return (
            t > 0.02 && <g key={i} opacity={clamp01(t)}>
              <rect x={bx} y={0} width={bw} height={h} fill={theme.accent} opacity={0.14} />
              <text x={bx + bw / 2} y={fontSize - (1 - t) * 20} textAnchor="middle" fill={theme.text}
                fontSize={fontSize * 1.3} fontWeight={700}>
                {b.label}
              </text>
            </g>
          );
        })}

        <g clipPath={`url(#reveal-${id})`}>
          {area !== 'none' && lines.map((l, i) => l.fill && (
            <path key={`a${i}`} d={paths[i].area}
              fill={area === 'flat' ? l.color : `url(#fill-${id}-${i})`} fillOpacity={area === 'flat' ? areaOpacity : 1} />
          ))}
          {[...lines.keys()].reverse().map((i) => (
            <path key={`l${i}`} d={paths[i].d} fill="none" stroke={lines[i].color} strokeWidth={i === 0 ? 5 : 4}
              strokeLinejoin="round" strokeLinecap="round" />
          ))}
        </g>
        {head.progress > 0 && heads.map((hd, i) => hd.x >= lines[i].points[0].x && (
          <g key={`h${i}`}
            transform={`translate(${sx(Math.min(hd.x, lines[i].points[lines[i].points.length - 1].x))},${sy(hd.y)})`}>
            <circle r={18} fill={lines[i].color} opacity={0.25} />
            <circle r={9} fill={lines[i].color} stroke={theme.bg} strokeWidth={3} />
          </g>
        ))}
        {showLegend && axisIn > 0.02 && (
          <g opacity={axisIn} transform={`translate(${w},${-m.top + fontSize * 0.6})`}>
            {lines.map((l, i) => {
              const after = lines.slice(i + 1).reduce((acc, n) => acc + n.name.length * fontSize * 0.6 + fontSize * 2.4, 0);
              return (
                <g key={i} transform={`translate(${-after - l.name.length * fontSize * 0.6},0)`}>
                  <rect x={-fontSize * 1.5} y={-4} width={fontSize} height={8} rx={4} fill={l.color} />
                  <text x={0} dy="0.35em" fill={theme.text} fontSize={fontSize}>{l.name}</text>
                </g>
              );
            })}
          </g>
        )}
      </g>
    </svg>
  );
};

// ---------------------------------------------------------------- bar chart

export type ChartBar = {
  label: string;
  value: number;
  /** Frame the bar turns to the theme's primary colour (a narration cue). */
  highlightAt?: number;
  /** The colour it turns to instead (e.g. theme.accent, to tell two groups apart). */
  highlightColor?: string;
  /** Small second line under the label ("to 4th"). */
  note?: string;
  /** Frame this bar starts to grow (a cue). Default: `start` plus its place in the stagger. */
  growAt?: number;
};

export type BarChartProps = {
  bars: ChartBar[];
  width: number;
  height: number;
  theme: ChartTheme;
  /** Frame the first bar starts to grow. */
  start: number;
  /** Frames between one bar and the next (default 6). */
  stagger?: number;
  /** Value label text (default: grouped integers). Called with the counting-up value. */
  formatValue?: (v: number) => string;
  fontSize?: number;
  /** Corner radius of the bars (default: rounded, up to 10px; 0 for square columns). */
  radius?: number;
};

export const BarChart: React.FC<BarChartProps> = ({
  bars, width, height, theme, start, stagger = 6, formatValue = defaultFormat, fontSize = 34, radius,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m = { top: fontSize * 2.4, bottom: fontSize * 2.2 };
  const h = height - m.top - m.bottom;
  const max = Math.max(...bars.map((b) => b.value), 1e-9) * 1.05;
  const slot = width / bars.length;
  const bw = slot * 0.56;

  return (
    <svg width={width} height={height} style={{ fontFamily: theme.font, overflow: 'visible' }}>
      <g transform={`translate(0,${m.top})`}>
        <line x1={0} x2={width} y1={h} y2={h} stroke={theme.grid} strokeWidth={2} />
        {bars.map((b, i) => {
          const t = spring({ frame: frame - (b.growAt ?? start + i * stagger), fps, config: { damping: 14, mass: 0.9 } });
          const hi = b.highlightAt === undefined ? 0 : clamp01(spring({ frame: frame - b.highlightAt, fps, config: { damping: 12 } }));
          const pop = 1 + 0.06 * Math.sin(Math.PI * hi);
          const bh = (b.value / max) * h * t * pop;
          const x = slot * i + (slot - bw * pop) / 2;
          const cx = slot * i + slot / 2;
          const vy = h - bh - fontSize * 0.6;
          const grow = (1.1 + 0.45 * hi) / 1.1;
          return (
            <g key={i}>
              <rect x={x} y={h - bh} width={bw * pop} height={Math.max(0, bh)} rx={radius ?? Math.min(10, bw / 6)}
                fill={interpolateColors(hi, [0, 1], [theme.other ?? theme.muted, b.highlightColor ?? theme.primary])}
                opacity={theme.other ? 1 : 0.45 + 0.55 * hi} />
              {/* the highlight grows the value by scaling it about its baseline, not by changing fontSize every
                  frame: a font size that changes each frame crashed Chrome after a few dozen frames (Remotion
                  4.0.539's headless shell, "Page crashed!") */}
              {t > 0.01 && <text x={cx} y={vy} textAnchor="middle"
                transform={hi > 0 ? `translate(${cx} ${vy}) scale(${grow}) translate(${-cx} ${-vy})` : undefined}
                fill={interpolateColors(hi, [0, 1], [theme.other ? theme.text : theme.muted, theme.text])}
                fontSize={fontSize * 1.1} fontWeight={700} opacity={clamp01(t * 3)} style={tabular}>
                {formatValue(b.value * clamp01(t))}
              </text>}
              <text x={slot * i + slot / 2} y={h + fontSize * 1.5} textAnchor="middle" fill={theme.muted} fontSize={fontSize}>
                {b.label}
              </text>
              {b.note && (
                <text x={slot * i + slot / 2} y={h + fontSize * 2.7} textAnchor="middle" fill={theme.muted}
                  fontSize={Math.max(28, fontSize * 0.8)}>
                  {b.note}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
};

// ---------------------------------------------------------------- counter

export type CounterProps = {
  /** The number it lands on. */
  value: number;
  /** Frame it starts counting (a cue), and how long it takes (default 40). */
  start: number;
  frames?: number;
  from?: number;
  format?: (v: number) => string;
  style?: React.CSSProperties;
};

/** A number that counts up to `value` with an ease-out, in tabular figures. */
export const Counter: React.FC<CounterProps> = ({ value, start, frames = 40, from = 0, format = defaultFormat, style }) => {
  const frame = useCurrentFrame();
  const v = interpolate(frame, [start, start + frames], [from, value], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  return <span style={{ ...tabular, ...style }}>{format(v)}</span>;
};

// ---------------------------------------------------------------- split bar

export type SplitSegment = {
  label: string;
  value: number;
  color?: string;
  /** Frame this segment's label appears (a cue). Default: just after the bar, one by one. */
  showAt?: number;
};

export type SplitBarProps = {
  /** Shares of one whole, left to right. A zero segment still gets its label, so "0" can be said. */
  segments: SplitSegment[];
  width: number;
  theme: ChartTheme;
  /** Frame the bar grows from the left (a cue), and when each label appears after it. */
  start: number;
  /** Bar thickness (default 64). */
  height?: number;
  formatValue?: (v: number) => string;
  fontSize?: number;
};

/** A horizontal bar split into labelled shares: who did the work, where the time went. */
export const SplitBar: React.FC<SplitBarProps> = ({
  segments, width, theme, start, height = 64, formatValue = defaultFormat, fontSize = 34,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  const palette = [theme.primary, theme.accent, theme.other ?? theme.muted];
  const grow = spring({ frame: frame - start, fps, config: { damping: 200 }, durationInFrames: 30 });
  let x = 0;
  const parts = segments.map((s, i) => {
    const w = (s.value / total) * width;
    const part = { ...s, x, w, color: s.color ?? palette[Math.min(i, palette.length - 1)] };
    x += w;
    return part;
  });
  // Zero shares are labelled one under another at the right end, so two of them never overlap.
  const zeros = parts.filter((p) => p.value === 0).length;
  const labelH = fontSize * (2.6 + 1.2 * Math.max(0, zeros - 1));
  const id = useId().replace(/[^a-zA-Z0-9]/g, '');

  return (
    <svg width={width} height={height + labelH} style={{ fontFamily: theme.font, overflow: 'visible' }}>
      <rect x={0} y={0} width={width} height={height} rx={height / 2} fill={theme.grid} />
      <clipPath id={`split-${id}`}>
        <rect x={0} y={0} width={width * grow} height={height} rx={height / 2} />
      </clipPath>
      <g clipPath={`url(#split-${id})`}>
        {parts.map((p, i) => <rect key={i} x={p.x} y={0} width={p.w} height={height} fill={p.color} />)}
      </g>
      {parts.map((p, i) => {
        const t = clamp01(spring({ frame: frame - (p.showAt ?? start + 20 + i * 12), fps, config: { damping: 200 } }));
        if (t < 0.02) return null;
        const zero = p.value === 0;
        // A zero share sits at the right end; others are centred under their segment.
        const cx = zero ? width : p.x + p.w / 2;
        const row = zero ? parts.slice(0, i).filter((q) => q.value === 0).length : 0;
        const anchor = zero ? 'end' : 'middle';
        return (
          <g key={i} opacity={t} transform={`translate(${cx},${height + fontSize * (1.3 + 1.2 * row) + (1 - t) * 12})`}>
            <text textAnchor={anchor} fill={zero ? theme.muted : theme.text} fontSize={fontSize} fontWeight={700} style={tabular}>
              {p.label} {formatValue(p.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
