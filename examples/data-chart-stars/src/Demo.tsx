/**
 * data-chart-stars: the toolkit's GitHub star history, narrated by Lugh, drawn by
 * lib/components DataChart. Every visual change is cued off a spoken word:
 *
 *   01a "This is the video toolkit's star history on GitHub."     axes and line start on "history"
 *   01b "It started slowly, ... in the first six weeks."            line reaches 28 Feb on "weeks"
 *   01c "Then, at the end of March, four hundred stars ..."         spike drawn, band on "four hundred"
 *   01d "Since then it has kept on climbing, ... two thousand four hundred."  line runs on to today
 *   02a "Month by month, March is still the big one."               March bar lights on "March"
 *   02b "But October was our strongest month since June."           October bar lights on "October"
 *
 * Phrase starts and scene lengths come from src/timing.json (scripts/timing.py), so a
 * regenerated line re-times the whole scene.
 */
import React from 'react';
import { AbsoluteFill, Audio, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import brand from '../../../brands/digital-samba/brand.json';
import { BarChart, chartTheme, LineChart, lineHead, LinePoint, voice } from '../../../lib/components';
import stars from '../data/star_series.json';
import timing from './timing.json';
import c01a from '../public/audio/scenes/01a_intro.captions.json';
import c01b from '../public/audio/scenes/01b_slow.captions.json';
import c01c from '../public/audio/scenes/01c_spike.captions.json';
import c01d from '../public/audio/scenes/01d_today.captions.json';
import c02a from '../public/audio/scenes/02a_march.captions.json';
import c02b from '../public/audio/scenes/02b_october.captions.json';

const theme = chartTheme(brand.colors, brand.fonts);
const day = (s: string) => Date.parse(`${s}T00:00:00Z`);
const series = stars as [string, number][];
const points: LinePoint[] = series.map(([d, v]) => ({ x: day(d), y: v }));
const TODAY = series[series.length - 1][0];
const TODAY_LABEL = new Date(day(TODAY)).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });

// Monthly gains and the 28-30 March spike, derived from the same series.
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'];
const ym = (i: number) => `2026-${String(i + 1).padStart(2, '0')}`;
const totalBefore = (d: string) => Math.max(0, ...series.filter(([s]) => s < d).map(([, v]) => v));
const gains = MONTHS.map((_, i) => totalBefore(ym(i + 1)) - totalBefore(ym(i)));
const spike = totalBefore('2026-03-31') - totalBefore('2026-03-28');

const T = timing as { phrases: Record<string, number>; scenes: { stars: number; months: number } };
const S1 = voice([
  { file: '01a_intro', at: T.phrases['01a_intro'], captions: c01a },
  { file: '01b_slow', at: T.phrases['01b_slow'], captions: c01b },
  { file: '01c_spike', at: T.phrases['01c_spike'], captions: c01c },
  { file: '01d_today', at: T.phrases['01d_today'], captions: c01d },
]);
const S2 = voice([
  { file: '02a_march', at: T.phrases['02a_march'], captions: c02a },
  { file: '02b_october', at: T.phrases['02b_october'], captions: c02b },
]);
export const TOTAL = T.scenes.stars + T.scenes.months;

// Where the line has drawn to, word by word.
const cue = S1.cue;
const HEAD = [
  { frame: cue('history') - 10, x: day('2026-01-20') },
  { frame: cue('weeks', { edge: 'end' }), x: day('2026-02-28') },
  { frame: cue('end of march'), x: day('2026-03-27') },
  { frame: cue('three days', { edge: 'end' }), x: day('2026-03-31') },
  { frame: cue('climbing'), x: day('2026-04-01') },
  { frame: cue('two thousand four hundred', { edge: 'end' }), x: day(TODAY) },
];

const Heading: React.FC<{ title: string; sub: string }> = ({ title, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = spring({ frame, fps, config: { damping: 200 } });
  return (
    <div style={{ position: 'absolute', left: 120, top: 90, opacity: t, transform: `translateY(${(1 - t) * 24}px)` }}>
      <div style={{ fontSize: 64, fontWeight: 800, color: theme.text, letterSpacing: -1 }}>{title}</div>
      <div style={{ fontSize: 32, color: theme.muted, marginTop: 8 }}>{sub}</div>
    </div>
  );
};

const Fade: React.FC<{ len: number; children: React.ReactNode }> = ({ len, children }) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 15, len - 15, len], [0, 1, 1, 0], { extrapolateRight: 'clamp' });
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

const Voice: React.FC<{ lines: [string, number][] }> = ({ lines }) => (
  <>
    {lines.map(([file, at]) => (
      <Sequence key={file} from={at}>
        <Audio src={staticFile(`audio/scenes/${file}.mp3`)} />
      </Sequence>
    ))}
  </>
);

const Footer: React.FC = () => (
  <div style={{ position: 'absolute', left: 120, bottom: 40, fontSize: 28, color: theme.muted }}>
    Source: GitHub Stars API, 20 Jan to {TODAY_LABEL}
  </div>
);

const StarsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const head = lineHead(points, frame, HEAD);
  return (
    <Fade len={T.scenes.stars}>
      <Voice lines={S1.lines} />
      <Heading title="GitHub stars" sub="digitalsamba/claude-code-video-toolkit" />
      <div style={{ position: 'absolute', right: 120, top: 70, textAlign: 'right' }}>
        <div style={{ fontSize: 132, fontWeight: 800, color: theme.primary, fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>
          {Math.round(head.y).toLocaleString('en-GB')}
        </div>
        <div style={{ fontSize: 30, color: theme.muted }}>stars</div>
      </div>
      <div style={{ position: 'absolute', left: 80, top: 300 }}>
        <LineChart
          points={points}
          keys={HEAD}
          width={1760}
          height={680}
          theme={theme}
          xTicks={MONTHS.slice(1).map((label, i) => ({ x: day(`${ym(i + 1)}-01`), label }))}
          bands={[{ from: day('2026-03-28'), to: day('2026-03-31'), label: `+${spike} in 3 days`, showAt: cue('four hundred') }]}
        />
      </div>
      <Footer />
    </Fade>
  );
};

const MonthsScene: React.FC = () => (
  <Fade len={T.scenes.months}>
    <Voice lines={S2.lines} />
    <Heading title="New stars per month" sub="The same data, as monthly gains" />
    <div style={{ position: 'absolute', left: 120, top: 290 }}>
      <BarChart
        width={1680}
        height={680}
        theme={theme}
        start={10}
        stagger={4}
        formatValue={(v: number) => `+${Math.round(v)}`}
        bars={MONTHS.map((label, i) => ({
          label,
          value: gains[i],
          highlightAt: label === 'Mar' ? S2.cue('march') : label === 'Oct' ? S2.cue('october') : undefined,
          note: label === 'Jan' ? 'from 20th' : label === 'Nov' ? `to ${Number(TODAY.slice(8))}th` : undefined,
        }))}
      />
    </div>
    <Footer />
  </Fade>
);

export const DataChartDemo: React.FC = () => (
  <AbsoluteFill style={{ background: theme.bg, fontFamily: theme.font }}>
    <Sequence durationInFrames={T.scenes.stars}>
      <StarsScene />
    </Sequence>
    <Sequence from={T.scenes.stars} durationInFrames={T.scenes.months}>
      <MonthsScene />
    </Sequence>
  </AbsoluteFill>
);
