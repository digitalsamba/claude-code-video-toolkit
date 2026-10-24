/**
 * Episode visuals: one component per visual group key used in story-config.ts.
 * Each receives the group's beats (one per scene) and decides what to show on
 * each beat. Frames are relative to the group's start.
 *
 * Two zones: <Stage> (y 300-1090, the main picture) and <Aside> (right of Max,
 * for live numbers). Moments inside a beat are placed by fraction of the beat
 * (`at(beats, 1, 0.4)`), so they stay roughly in sync when sync_timing.py
 * stretches a scene. Nudge the fractions after the real voiceover is in.
 */
import { Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';
import { EditorPanel } from '../components/EditorPanel';
import { EquationCard, Frac, Hl } from '../components/Equation';
import { Camera, Sfx } from '../components/Fx';
import { LineChart } from '../components/LineChart';
import { StatRow, fmtInt, fmtMoney } from '../components/StatRow';
import { useBeat, type VisualGroup } from '../timeline';
import { BalanceScale } from './BalanceScale';
import { aheadAt, at as pathAt, casinoWins, CASINO_MAX, CASINO_MIN, keptCents, maxOutcomes, maxPath, PLAYERS, players } from './data';
import { Roulette } from './Roulette';

type Beats = VisualGroup['beats'];
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;
const ease = Easing.bezier(0.45, 0, 0.2, 1);
/** Group-relative frame at a fraction of a beat */
const at = (beats: Beats, beat: number, frac: number) => beats[beat].from + Math.round(beats[beat].frames * frac);

const STAGE_W = 1080 - theme.layout.gutter * 2;
const STAGE_H = theme.layout.stageBottom - theme.layout.stageTop;
const ASIDE_W = 1080 - theme.layout.gutter - theme.layout.asideLeft;

const Stage: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ position: 'absolute', left: theme.layout.gutter, top: theme.layout.stageTop, width: STAGE_W, height: STAGE_H, ...style }}>
    {children}
  </div>
);

/** The zone right of Max in side layout: live readouts. */
const Aside: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ position: 'absolute', left: theme.layout.asideLeft, top: theme.layout.asideTop, width: ASIDE_W, ...style }}>{children}</div>
);

const Chip: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="-50 -50 100 100">
    <circle r={48} fill={theme.primary} />
    {Array.from({ length: 8 }, (_, i) => (
      <rect key={i} x={-6} y={-48} width={12} height={14} rx={2} fill={theme.cream} transform={`rotate(${i * 45})`} />
    ))}
    <circle r={32} fill={theme.primaryDark} stroke={theme.cream} strokeWidth={2.5} strokeDasharray="5 5" />
    <text textAnchor="middle" dominantBaseline="central" fontFamily={theme.fonts.sans} fontWeight={800} fontSize={30} fill={theme.cream}>
      $1
    </text>
  </svg>
);

/** Wheel spin: ball runs counter to the wheel and settles near the end of each cycle. */
const spinState = (frame: number) => {
  const rotation = frame * 1.4;
  const t = (frame % 150) / 150;
  const ballAngle = -frame * 2 - Math.pow(1 - t, 2) * 900;
  return { rotation, ballAngle, ballDrop: interpolate(t, [0.7, 0.9], [0, 1], clamp) };
};

// Shared bankroll chart geometry (hook + bankroll use the same picture)
const BANK_Y = (zoom: number) => ({ min: interpolate(zoom, [0, 1], [-40, -450]), max: interpolate(zoom, [0, 1], [40, 250]) });
const xAxis = (xMax: number) => ({
  min: 0,
  max: xMax,
  ticks: xMax < 2000 ? [0, 50, 100] : [0, 5000, 10000],
  format: (v: number) => (v >= 1000 ? `${v / 1000}k` : `${v}`),
});
const yMoney = (lo: number, hi: number, zoomed: boolean) => ({ min: lo, max: hi, ticks: zoomed ? [-400, -200, 0, 200] : [-20, 0, 20], format: fmtMoney });
const maxSeries = (reveal: number) => ({ id: 'max', y: (x: number) => pathAt(maxPath, x), color: '#e5303d', width: 6, glow: true, area: true, revealTo: reveal, marker: 'Max' });
/** Where Max's end dot sits on stage, for camera push-ins (panel at stage 0,0). */
const markerOnStage = (chartH: number, x: number, xMax: number, yLo: number, yHi: number) => {
  const pad = { left: 190, right: 60, top: 40, bottom: 84 };
  const ih = chartH - pad.top - pad.bottom;
  return {
    x: pad.left + (x / xMax) * (STAGE_W - pad.left - pad.right),
    y: 74 + pad.top + (1 - (pathAt(maxPath, x) - yLo) / (yHi - yLo)) * ih,
  };
};

// ── 00 hook: the crash first, then rewind ────────────────────────
export const HookVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const total = beats[0].frames;
  const crashEnd = Math.round(total * 0.5);
  const rewindStart = total - 26;
  const forward = interpolate(frame, [4, crashEnd], [0, 10000], { ...clamp, easing: Easing.in(Easing.cubic) });
  const reveal = frame < rewindStart ? forward : interpolate(frame, [rewindStart, total - 4], [10000, 0], { ...clamp, easing: Easing.in(Easing.quad) });
  const rewinding = frame >= rewindStart;
  const { min, max } = BANK_Y(1);
  const chartH = 640;
  const m = markerOnStage(chartH, 10000, 10000, min, max);
  const t1 = spring({ frame: frame - 2, fps, config: { damping: 16 } });
  const t2 = spring({ frame: frame - Math.round(total * 0.42), fps, config: { damping: 12, stiffness: 150 } });
  const money = pathAt(maxPath, reveal);
  const settled = frame >= crashEnd && !rewinding;
  const jitter = rewinding ? Math.sin(frame * 3.1) * 6 : 0;

  return (
    <>
      <div style={{ position: 'absolute', left: theme.layout.gutter, right: theme.layout.gutter, top: 96, fontFamily: theme.fonts.sans, fontWeight: 900, letterSpacing: -2, lineHeight: 1.02 }}>
        <div style={{ fontSize: 84, color: theme.text, opacity: t1, transform: `translateY(${(1 - t1) * 40}px)` }}>Red or black: 50/50.</div>
        <div style={{ fontSize: 84, color: '#ff4d58', opacity: t2, transform: `translateY(${(1 - t2) * 40}px)`, textShadow: '0 0 40px rgba(193,18,31,0.5)' }}>So why is Max broke?</div>
      </div>
      <Stage style={{ top: theme.layout.stageTop + 40, height: STAGE_H - 40, overflow: 'hidden', borderRadius: 30 }}>
        <Camera
          keys={[
            { at: crashEnd - 6, scale: 1, x: STAGE_W / 2, y: STAGE_H / 2 },
            { at: crashEnd + 14, scale: 1.22, x: m.x - 60, y: m.y },
            { at: rewindStart, scale: 1.26, x: m.x - 60, y: m.y },
            { at: total - 6, scale: 1, x: STAGE_W / 2, y: STAGE_H / 2 },
          ]}
        >
          <div style={{ transform: `translateX(${jitter}px)`, filter: rewinding ? 'saturate(0.6) contrast(1.15)' : 'none' }}>
            <EditorPanel file="bankroll.plot" live status={rewinding ? '◀◀ rewind' : `spin ${fmtInt(reveal)}`} height={chartH + 74}>
              <LineChart width={STAGE_W} height={chartH} x={xAxis(10000)} y={yMoney(min, max, true)} refLines={[{ y: 0, color: theme.accent }]} series={[maxSeries(reveal)]} />
            </EditorPanel>
          </div>
        </Camera>
        {rewinding && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              borderRadius: 30,
              background: 'repeating-linear-gradient(0deg, rgba(253,240,213,0.06) 0 2px, transparent 2px 7px)',
              opacity: 0.6 + 0.4 * Math.sin(frame * 2),
            }}
          />
        )}
      </Stage>
      <Aside>
        <StatRow
          column
          width={ASIDE_W}
          delay={6}
          pulse={settled ? interpolate(frame, [crashEnd, crashEnd + 8, crashEnd + 24], [0, 1, 0.3], clamp) : 0}
          stats={[
            { label: 'spins', value: fmtInt(reveal) },
            { label: "Max's money", value: fmtMoney(money), tone: money < 0 ? 'bad' : 'neutral' },
          ]}
        />
      </Aside>
      <Sfx name="drop" at={Math.round(total * 0.18)} volume={0.5} />
      <Sfx name="thud" at={crashEnd} volume={0.8} />
      <Sfx name="rewind" at={rewindStart} volume={0.45} />
    </>
  );
};

// ── 01 table ─────────────────────────────────────────────────────
export const TableVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const { fps } = useVideoConfig();
  const { beat, local, frames, frame } = useBeat(beats);
  const enter = spring({ frame, fps, config: { damping: 16, stiffness: 90 } });
  const spin = spinState(frame);

  // beat 2: wheel shrinks to the top-left to make room for the results
  const shrink = beat >= 2 ? spring({ frame: local, fps, config: { damping: 20, stiffness: 100 } }) : 0;
  const size = interpolate(shrink, [0, 1], [600, 380]);
  const wx = interpolate(shrink, [0, 1], [60, 0]);

  // beat 1: "red or black": red, then black, then both back
  const hl = beat === 1 ? (local < frames * 0.22 ? 'red' : local < frames * 0.42 ? 'black' : null) : null;
  const hlAmount = beat === 1 ? interpolate(local, [0, 8, frames * 0.42 - 6, frames * 0.42], [0, 0.7, 0.7, 0], clamp) : 0;

  const chipAt = at(beats, 0, 0.55);
  const chipIn = spring({ frame: frame - chipAt, fps, config: { damping: 12, stiffness: 140 } });
  const splitIn = beat === 1 ? spring({ frame: local - Math.round(frames * 0.3), fps, config: { damping: 18 } }) : 0;

  // beat 2: ten spins land one by one
  const spinTime = (i: number) => beats[2].from + interpolate(i, [0, 10], [frames * 0.12, frames * 0.82]);
  const spinsShown = beat >= 2 ? Math.max(0, Math.min(10, Math.floor(interpolate(local, [frames * 0.12, frames * 0.82], [0, 10], clamp)))) : 0;
  const money = maxPath[spinsShown];

  return (
    <>
      <Stage>
        <div style={{ position: 'absolute', left: wx, top: 0, opacity: enter, transform: `scale(${interpolate(enter, [0, 1], [0.8, 1])})` }}>
          <Roulette size={size} {...spin} highlight={hl} highlightAmount={hlAmount} />
        </div>

        <div
          style={{
            position: 'absolute',
            right: beat >= 2 ? 30 : 0,
            top: beat >= 2 ? 110 : 470,
            display: 'flex',
            alignItems: 'center',
            gap: 18,
            opacity: chipIn,
            transform: `translateY(${interpolate(chipIn, [0, 1], [-120, 0])}px) rotate(${interpolate(chipIn, [0, 1], [-40, 0])}deg)`,
          }}
        >
          <Chip size={150} />
          <div style={{ fontFamily: theme.fonts.sans, fontWeight: 800, fontSize: 44, color: theme.muted }}>
            on <span style={{ color: '#ff5a63' }}>red</span>
          </div>
        </div>

        {beat === 1 && (
          <div style={{ position: 'absolute', left: 0, right: 0, top: 680, opacity: splitIn }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: theme.fonts.sans, fontWeight: 800, fontSize: 44, marginBottom: 14 }}>
              <span style={{ color: '#ff5a63' }}>red 50%</span>
              <span style={{ color: theme.cream }}>50% black</span>
            </div>
            <div style={{ display: 'flex', height: 22, borderRadius: 11, overflow: 'hidden', background: theme.hairline }}>
              <div style={{ width: `${50 * splitIn}%`, background: theme.primary }} />
              <div style={{ flex: 1 }} />
              <div style={{ width: `${50 * splitIn}%`, background: '#16181d' }} />
            </div>
          </div>
        )}

        {beat >= 2 && (
          <div style={{ position: 'absolute', left: 0, top: 440, width: STAGE_W, display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 18 }}>
            {maxOutcomes.slice(0, 10).map((o, i) => {
              const s = spring({ frame: frame - spinTime(i), fps, config: { damping: 12, stiffness: 180 } });
              const win = o === 'red';
              return (
                <div
                  key={i}
                  style={{
                    height: 120,
                    borderRadius: 22,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: theme.fonts.sans,
                    fontWeight: 900,
                    fontSize: 52,
                    color: theme.cream,
                    background: o === 'red' ? `linear-gradient(160deg, #d61f2c, ${theme.primaryDark})` : o === 'green' ? 'linear-gradient(160deg, #35b5a5, #1d7a6e)' : 'linear-gradient(160deg, #2a2d35, #0d0e11)',
                    border: `2px solid ${win ? 'rgba(253,240,213,0.35)' : 'rgba(102,155,188,0.3)'}`,
                    boxShadow: win ? '0 12px 30px rgba(193,18,31,0.35)' : '0 12px 30px rgba(0,0,0,0.35)',
                    opacity: s,
                    transform: `scale(${s}) translateY(${(1 - s) * -60}px)`,
                  }}
                >
                  {win ? '+1' : '−1'}
                </div>
              );
            })}
          </div>
        )}
      </Stage>
      {beat >= 2 && (
        <Aside>
          <StatRow
            column
            width={ASIDE_W}
            delay={beats[2].from + 6}
            stats={[
              { label: 'spins', value: fmtInt(spinsShown) },
              { label: "Max's money", value: fmtMoney(money), tone: money < 0 ? 'bad' : 'neutral' },
            ]}
          />
        </Aside>
      )}
      <Sfx name="chip" at={chipAt} />
      {Array.from({ length: 10 }, (_, i) => (
        <Sfx key={i} name="chip" at={spinTime(i)} volume={0.35} />
      ))}
    </>
  );
};

// ── 02 bankroll ──────────────────────────────────────────────────
export const BankrollVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const frame = useCurrentFrame();
  const chartH = 700;
  const zoom = interpolate(frame, [beats[1].from, beats[1].from + 36], [0, 1], { ...clamp, easing: ease });
  const crashEnd = beats[1].from + beats[1].frames - 30;
  const reveal =
    frame < beats[1].from
      ? interpolate(frame, [20, beats[0].frames - 40], [0, 100], clamp)
      : interpolate(frame, [beats[1].from + 10, crashEnd], [100, 10000], { ...clamp, easing: Easing.in(Easing.quad) });
  const xMax = interpolate(zoom, [0, 1], [100, 10000]);
  const y = BANK_Y(zoom);
  const money = pathAt(maxPath, reveal);
  const m = markerOnStage(chartH, 10000, 10000, -450, 250);
  // beat 2: a nudge on the question
  const shake = frame >= beats[2].from ? Math.sin(frame * 1.7) * interpolate(frame - beats[2].from, [0, 20], [6, 0], clamp) : 0;

  return (
    <>
      <Stage style={{ overflow: 'hidden', borderRadius: 30 }}>
        <Camera
          keys={[
            { at: crashEnd - 4, scale: 1, x: STAGE_W / 2, y: STAGE_H / 2 },
            { at: crashEnd + 16, scale: 1.2, x: m.x - 80, y: m.y },
            { at: beats[2].from + 18, scale: 1.2, x: m.x - 80, y: m.y },
            { at: beats[2].from + 40, scale: 1, x: STAGE_W / 2, y: STAGE_H / 2 },
          ]}
        >
          <div style={{ transform: `translateX(${shake}px)` }}>
            <EditorPanel file="bankroll.plot" live status={`spin ${fmtInt(reveal)}`} height={chartH + 74}>
              <LineChart width={STAGE_W} height={chartH} x={xAxis(xMax)} y={yMoney(y.min, y.max, zoom >= 0.5)} refLines={[{ y: 0, color: theme.accent }]} series={[maxSeries(reveal)]} />
            </EditorPanel>
          </div>
        </Camera>
      </Stage>
      <Aside>
        <StatRow
          column
          width={ASIDE_W}
          delay={10}
          pulse={interpolate(frame, [crashEnd, crashEnd + 8, crashEnd + 30], [0, 1, 0.25], clamp)}
          stats={[
            { label: 'spins', value: fmtInt(reveal) },
            { label: "Max's money", value: fmtMoney(money), tone: money < 0 ? 'bad' : 'neutral' },
          ]}
        />
      </Aside>
      <Sfx name="drop" at={beats[1].from + 12} volume={0.45} />
      <Sfx name="thud" at={crashEnd} volume={0.8} />
    </>
  );
};

// ── 03 players ───────────────────────────────────────────────────
export const PlayersVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const frame = useCurrentFrame();
  const chartH = 700;
  const othersIn = at(beats, 0, 0.45);
  const othersReveal = interpolate(frame, [othersIn, othersIn + 45], [0, 100], { ...clamp, easing: ease });
  const zoomStart = beats[2].from + 8;
  const zoom = interpolate(frame, [zoomStart, zoomStart + 36], [0, 1], { ...clamp, easing: ease });
  const end = beats[2].from + beats[2].frames - 30;
  const reveal = frame < zoomStart ? 100 : interpolate(frame, [zoomStart + 6, end], [100, 10000], { ...clamp, easing: Easing.in(Easing.quad) });
  const othersTo = frame < zoomStart ? othersReveal : reveal;
  const xMax = interpolate(zoom, [0, 1], [100, 10000]);
  const yMin = interpolate(zoom, [0, 1], [-40, -520]);
  const yMax = interpolate(zoom, [0, 1], [40, 260]);
  const showAhead = frame >= beats[1].from;
  const ahead = aheadAt(frame >= zoomStart ? reveal : 100);
  const markAhead = interpolate(frame, [beats[1].from, beats[1].from + 15, zoomStart, zoomStart + 20], [0, 1, 1, 0], clamp);
  const zeroAt = (() => {
    for (let f = zoomStart; f <= end; f++) {
      const r = interpolate(f, [zoomStart + 6, end], [100, 10000], { ...clamp, easing: Easing.in(Easing.quad) });
      if (aheadAt(r) === 0) return f;
    }
    return end;
  })();

  return (
    <>
      <Stage>
        <EditorPanel file="players.plot" live status={`${PLAYERS} players`} height={chartH + 74}>
          <LineChart
            width={STAGE_W}
            height={chartH}
            x={xAxis(xMax)}
            y={yMoney(yMin, yMax, zoom >= 0.5)}
            refLines={[{ y: 0, color: theme.accent }]}
            series={[
              ...players.slice(1).map((p, i) => {
                const lit = markAhead > 0 && pathAt(p, 100) > 0;
                return {
                  id: `p${i}`,
                  y: (x: number) => pathAt(p, x),
                  color: lit ? theme.cream : theme.accent,
                  width: lit ? 3 : 2.2,
                  opacity: othersTo > 0 ? (lit ? 0.75 : 0.32) : 0,
                  revealTo: othersTo,
                  samples: 160,
                };
              }),
              { ...maxSeries(reveal), area: false },
            ]}
          />
        </EditorPanel>
      </Stage>
      <Aside>
        <StatRow
          column
          width={ASIDE_W}
          delay={10}
          pulse={interpolate(frame, [zeroAt, zeroAt + 8, zeroAt + 30], [0, 1, 0.25], clamp)}
          stats={[
            { label: 'spins', value: fmtInt(reveal) },
            { label: 'players ahead', value: showAhead ? `${ahead} / ${PLAYERS}` : '…', tone: showAhead && ahead === 0 ? 'bad' : 'neutral' },
          ]}
        />
      </Aside>
      <Sfx name="whoosh" at={othersIn} volume={0.35} />
      <Sfx name="pop" at={beats[1].from + 4} />
      <Sfx name="thud" at={zeroAt} volume={0.8} />
    </>
  );
};

// ── 04 wheel, closer ─────────────────────────────────────────────
export const WheelVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const enter = spring({ frame, fps, config: { damping: 18, stiffness: 90 } });
  const tRed = at(beats, 0, 0.42);
  const tBlack = at(beats, 0, 0.56);
  const tGreen = at(beats, 0, 0.72);
  const hl = frame >= tGreen ? 'green' : frame >= tBlack ? 'black' : frame >= tRed ? 'red' : null;
  const hlAmount = interpolate(frame, [tRed, tRed + 8], [0, 0.6], clamp);
  // Spin down and park the zero at the top for the reveal
  const settle = interpolate(frame, [tRed - 30, tGreen], [0, 1], { ...clamp, easing: ease });
  const rotation = interpolate(settle, [0, 1], [frame * 0.8, Math.ceil((tGreen * 0.8) / 360) * 360]);
  const size = 780;
  const left = (STAGE_W - size) / 2;
  // zero pocket centre: radius ~385/500 of the wheel, straight up
  const zero = { x: left + size / 2, y: 10 + size / 2 - (385 / 500) * (size / 2) };
  const tiles: { label: string; value: string; at: number; color: string; key: 'red' | 'black' | 'green' }[] = [
    { label: 'red', value: '18', at: tRed, color: '#ff5a63', key: 'red' },
    { label: 'black', value: '18', at: tBlack, color: theme.cream, key: 'black' },
    { label: 'green', value: '1', at: tGreen, color: '#3cc9b8', key: 'green' },
  ];

  return (
    <>
      <Stage style={{ overflow: 'hidden', borderRadius: 40 }}>
        <Camera
          keys={[
            { at: tGreen - 2, scale: 1, x: STAGE_W / 2, y: STAGE_H / 2 },
            { at: tGreen + 22, scale: 2.3, x: zero.x, y: zero.y },
          ]}
        >
          <div style={{ position: 'absolute', left, top: 10, opacity: enter, transform: `scale(${interpolate(enter, [0, 1], [0.85, 1])})` }}>
            <Roulette size={size} rotation={rotation} ballAngle={rotation + 3 * (360 / 37)} ballDrop={1} highlight={hl} highlightAmount={hlAmount} />
          </div>
        </Camera>
        {frame >= tGreen + 16 && (
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: 40,
              textAlign: 'center',
              fontFamily: theme.fonts.sans,
              fontWeight: 900,
              fontSize: 76,
              color: '#3cc9b8',
              textShadow: '0 0 30px rgba(42,157,143,0.6)',
              opacity: interpolate(frame, [tGreen + 16, tGreen + 26], [0, 1], clamp),
            }}
          >
            the zero
          </div>
        )}
      </Stage>
      <Aside style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {tiles.map((t) => {
          const s = spring({ frame: frame - t.at, fps, config: { damping: 13, stiffness: 160 } });
          const on = s > 0.05;
          return (
            <div
              key={t.label}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 28px',
                borderRadius: 20,
                background: 'rgba(11,60,92,0.55)',
                border: `2px solid ${hl === t.key ? t.color : theme.hairline}`,
                opacity: Math.max(0.3, s),
                transform: `translateX(${interpolate(s, [0, 1], [30, 0])}px)`,
              }}
            >
              <span style={{ fontFamily: theme.fonts.mono, fontSize: 34, color: theme.muted }}>{t.label}</span>
              <span style={{ fontFamily: theme.fonts.sans, fontWeight: 900, fontSize: 66, color: on ? t.color : theme.faint }}>{on ? t.value : '?'}</span>
            </div>
          );
        })}
      </Aside>
      {tiles.map((t) => (
        <Sfx key={t.label} name="pop" at={t.at} />
      ))}
      <Sfx name="whoosh" at={tGreen} volume={0.4} />
    </>
  );
};

// ── 05 the house edge ────────────────────────────────────────────
export const EdgeVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const tWins = at(beats, 0, 0.5);
  const tLoses = at(beats, 0, 0.74);
  const tNet = at(beats, 1, 0.12);
  const tFrac = at(beats, 1, 0.42);
  const tEdge = at(beats, 1, 0.78);
  const tiny = frame >= beats[2].from;
  const pulse = tiny ? 1 + 0.05 * Math.sin((frame - beats[2].from) / 4) : 1;
  const edgeIn = spring({ frame: frame - tEdge, fps, config: { damping: 11, stiffness: 150 } });
  const eqIn = frame >= tFrac - 6;

  return (
    <>
      <Stage>
        <div style={{ position: 'absolute', left: 0, top: eqIn ? interpolate(frame, [tFrac - 6, tFrac + 10], [0, -40], clamp) : 0, transform: `scale(${eqIn ? interpolate(frame, [tFrac - 6, tFrac + 10], [1, 0.86], clamp) : 1})`, transformOrigin: '50% 0%' }}>
          <BalanceScale winsAt={tWins} lossesAt={tLoses} netAt={tNet} />
        </div>
        <div style={{ position: 'absolute', top: 560, opacity: tiny ? 0.55 : 1 }}>
          <EquationCard
            delay={tFrac - 6}
            lines={[
              {
                at: tFrac,
                size: 64,
                content: (
                  <>
                    <Frac n="−$1" d="37 spins" /> ≈ <Hl at={tFrac + 12}>−2.7¢</Hl>&nbsp;per spin
                  </>
                ),
              },
            ]}
          />
        </div>
      </Stage>
      <Aside style={{ display: 'flex', justifyContent: 'center', paddingTop: 70 }}>
        <div
          style={{
            padding: '22px 34px',
            borderRadius: 26,
            background: `linear-gradient(135deg, ${theme.primary}, ${theme.primaryDark})`,
            textAlign: 'center',
            opacity: edgeIn,
            transform: `scale(${interpolate(edgeIn, [0, 1], [1.8, 1]) * pulse}) rotate(${interpolate(edgeIn, [0, 1], [10, -3])}deg)`,
            boxShadow: `0 0 ${tiny ? 60 : 24}px rgba(193,18,31,0.6)`,
          }}
        >
          <div style={{ fontFamily: theme.fonts.mono, fontWeight: 700, fontSize: 30, letterSpacing: 3, color: theme.cream, opacity: 0.85 }}>THE HOUSE EDGE</div>
          <div style={{ fontFamily: theme.fonts.sans, fontWeight: 900, fontSize: 96, color: theme.cream, lineHeight: 1 }}>2.7%</div>
        </div>
      </Aside>
      {Array.from({ length: 6 }, (_, i) => (
        <Sfx key={`w${i}`} name="chip" at={tWins + i * 6} volume={0.4} />
      ))}
      {Array.from({ length: 6 }, (_, i) => (
        <Sfx key={`l${i}`} name="chip" at={tLoses + i * 6} volume={0.4} />
      ))}
      <Sfx name="pop" at={tNet} />
      <Sfx name="pop" at={tFrac} volume={0.4} />
      <Sfx name="thud" at={tEdge} volume={0.7} />
    </>
  );
};

// ── 06 casino scale ──────────────────────────────────────────────
export const CasinoVisual: React.FC<{ beats: Beats }> = ({ beats }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const chartH = 510;
  const logReveal = interpolate(frame, [15, beats[0].frames - 10], [4, 8], { ...clamp, easing: Easing.inOut(Easing.quad) });
  const n = Math.pow(10, logReveal);
  const bets = frame >= beats[0].frames - 10 ? CASINO_MAX : n;
  const tEq = at(beats, 1, 0.15);
  const punch = frame - beats[2].from;
  const back = punch >= 0 ? spring({ frame: punch, fps, config: { damping: 20 } }) : 0;
  const luck = spring({ frame: punch - 4, fps, config: { damping: 16 } });
  const strike = interpolate(punch, [beats[2].frames * 0.32, beats[2].frames * 0.42], [0, 1], clamp);
  const tStamp = Math.round(beats[2].frames * 0.5);
  const stamp = spring({ frame: punch - tStamp, fps, config: { damping: 10, stiffness: 200 } });
  const shake = punch >= tStamp ? Math.sin(punch * 2.3) * interpolate(punch - tStamp, [0, 12], [10, 0], clamp) : 0;
  const recede = { opacity: 1 - back * 0.8, filter: `blur(${back * 8}px)`, transform: `scale(${1 - back * 0.06})` };

  return (
    <>
      <Stage>
        <div style={recede}>
          <EditorPanel file="casino.plot" live status="$1 on red" height={chartH + 74}>
            <LineChart
              width={STAGE_W}
              height={chartH}
              x={{ min: CASINO_MIN, max: CASINO_MAX, scale: 'log', ticks: [1e4, 1e6, 1e8], format: (v) => (v >= 1e6 ? `${v / 1e6}M` : `${v / 1e3}k`) }}
              y={{ min: 0, max: 4.5, ticks: [1, 2, 3, 4], format: (v) => `${v}¢` }}
              refLines={[{ y: 100 / 37, label: '2.7¢ per bet', color: theme.cream }]}
              series={[{ id: 'casino', y: keptCents, color: '#e5303d', width: 6, glow: true, revealTo: n, marker: 'casino' }]}
            />
          </EditorPanel>
          <div style={{ position: 'absolute', top: chartH + 74 + 18, left: 0, opacity: frame >= tEq ? 1 : 0 }}>
            <EquationCard
              delay={tEq}
              lines={[
                {
                  at: tEq,
                  size: 50,
                  content: (
                    <>
                      <Frac n="$100 million" d="37" /> ≈ <Hl at={tEq + 16}>$2.7 million</Hl>
                    </>
                  ),
                },
              ]}
            />
          </div>
        </div>

        {punch >= 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 50, transform: `translate(${shake}px, ${shake * 0.6}px)` }}>
            <div style={{ position: 'relative', opacity: luck, transform: `translateY(${interpolate(luck, [0, 1], [40, 0])}px)` }}>
              <div style={{ fontFamily: theme.fonts.sans, fontWeight: 900, fontSize: 104, letterSpacing: -3, color: theme.text }}>It isn't luck.</div>
              <div style={{ position: 'absolute', left: -12, top: '52%', height: 12, borderRadius: 6, width: `${strike * 104}%`, background: theme.primary }} />
            </div>
            <div
              style={{
                padding: '20px 48px',
                border: `9px solid ${theme.primary}`,
                borderRadius: 26,
                fontFamily: theme.fonts.sans,
                fontWeight: 900,
                fontSize: 128,
                letterSpacing: 4,
                color: theme.primary,
                background: 'rgba(253,240,213,0.96)',
                opacity: stamp > 0.02 ? 1 : 0,
                transform: `scale(${interpolate(stamp, [0, 1], [2.4, 1])}) rotate(-7deg)`,
                boxShadow: '0 30px 80px rgba(0,10,20,0.6)',
              }}
            >
              PAYCHECK
            </div>
          </div>
        )}
      </Stage>
      <Aside style={{ ...recede, transformOrigin: '0 0' }}>
        <StatRow
          column
          width={ASIDE_W}
          delay={10}
          stats={[
            { label: 'bets played', value: fmtInt(bets) },
            { label: 'casino wins', value: `$${fmtInt(casinoWins(bets))}` },
          ]}
        />
      </Aside>
      <Sfx name="pop" at={tEq} />
      <Sfx name="whoosh" at={beats[2].from} volume={0.4} />
      <Sfx name="thud" at={beats[2].from + tStamp} volume={1} />
    </>
  );
};

export const visuals: Record<string, React.FC<{ beats: Beats }>> = {
  hook: HookVisual,
  table: TableVisual,
  bankroll: BankrollVisual,
  players: PlayersVisual,
  wheel: WheelVisual,
  edge: EdgeVisual,
  casino: CasinoVisual,
};
