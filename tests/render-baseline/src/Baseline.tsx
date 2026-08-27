/**
 * Render-baseline composition.
 *
 * One block per representative API. Each block is FRAMES_PER_BLOCK long and
 * the harness (scripts/render-baseline.mjs) renders the middle frame of each,
 * so a change in any block shows up as exactly one differing still.
 *
 * Determinism rules for this file:
 *   - only the bundled font (public/fonts/BaselineSans.ttf) — never system-ui
 *   - no Math.random / Date; use remotion's random(seed) if you need noise
 *   - no network assets
 */
import React from 'react';
import {
  AbsoluteFill,
  Audio,
  cancelRender,
  continueRender,
  delayRender,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  Series,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { glitch, zoomBlur } from '../../../lib/transitions';
import { AnimatedBackground, Vignette, FilmGrain, Label } from '../../../lib/components';
import { ThemeProvider, defaultThemeValues } from '../../../lib/theme';

export const FPS = 30;
export const FRAMES_PER_BLOCK = 30;

const FONT = 'BaselineSans';

// Load the bundled font once at module scope; rendering blocks until it is in.
const fontHandle = delayRender('Loading BaselineSans');
const fontFace = new FontFace(FONT, `url(${staticFile('fonts/BaselineSans.ttf')})`);
fontFace
  .load()
  .then((f) => {
    document.fonts.add(f);
    continueRender(fontHandle);
  })
  .catch((err) => cancelRender(err));

// Dark theme so lib/components render with contrast (default theme is light-on-white).
const theme = {
  ...defaultThemeValues,
  colors: { ...defaultThemeValues.colors, bgLight: '#0f172a', bgDark: '#020617', textDark: '#1e293b', textMedium: '#cbd5e1', textLight: '#94a3b8', divider: '#334155' },
  fonts: { ...defaultThemeValues.fonts, primary: FONT, mono: FONT },
};

// ─── Blocks ──────────────────────────────────────────────────

const Caption: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      position: 'absolute',
      left: 40,
      top: 28,
      fontFamily: FONT,
      fontSize: 28,
      color: '#e2e8f0',
      letterSpacing: 1,
    }}
  >
    {text}
  </div>
);

/** interpolate + spring + Easing on a bar chart */
const EasingBlock: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const bars = [
    interpolate(frame, [0, 25], [0, 1], { extrapolateRight: 'clamp' }),
    interpolate(frame, [0, 25], [0, 1], { extrapolateRight: 'clamp', easing: Easing.bezier(0.16, 1, 0.3, 1) }),
    interpolate(frame, [0, 25], [0, 1], { extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic) }),
    spring({ frame, fps, config: { damping: 8, stiffness: 120 } }),
    spring({ frame, fps, config: { damping: 200 } }),
  ];
  return (
    <AbsoluteFill style={{ backgroundColor: '#0f172a' }}>
      <Caption text="01 interpolate / spring / Easing" />
      <div style={{ position: 'absolute', left: 120, right: 120, bottom: 80, top: 120, display: 'flex', alignItems: 'flex-end', gap: 40 }}>
        {bars.map((v, i) => (
          <div key={i} style={{ flex: 1, height: `${Math.max(0, v) * 100}%`, backgroundColor: ['#3b82f6', '#60a5fa', '#a78bfa', '#f59e0b', '#10b981'][i], borderRadius: 8 }} />
        ))}
      </div>
    </AbsoluteFill>
  );
};

/** Sequence nesting + text layout in the bundled font */
const SequenceBlock: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: '#111827' }}>
    <Caption text="02 Sequence / text" />
    {[0, 1, 2].map((i) => (
      <Sequence key={i} from={i * 6} durationInFrames={FRAMES_PER_BLOCK - i * 6} layout="none">
        <SlideInLine index={i} />
      </Sequence>
    ))}
  </AbsoluteFill>
);

const SlideInLine: React.FC<{ index: number }> = ({ index }) => {
  const frame = useCurrentFrame();
  const x = interpolate(frame, [0, 12], [-400, 0], { extrapolateRight: 'clamp', easing: Easing.out(Easing.quad) });
  const sizes = [72, 48, 32];
  return (
    <div
      style={{
        position: 'absolute',
        left: 120,
        top: 160 + index * 150,
        transform: `translateX(${x}px)`,
        fontFamily: FONT,
        fontSize: sizes[index],
        fontWeight: index === 0 ? 700 : 400,
        color: '#f8fafc',
      }}
    >
      The quick brown fox jumps over {index === 0 ? 'the lazy dog' : `line ${index + 1} — 0123456789`}
    </div>
  );
};

/** OffthreadVideo on a committed testsrc clip (the toolkit's media choice) */
const VideoBlock: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: '#000' }}>
    <Caption text="03 OffthreadVideo" />
    <div style={{ position: 'absolute', left: 240, top: 120, width: 800, height: 450, overflow: 'hidden', borderRadius: 12, border: '4px solid #334155' }}>
      <OffthreadVideo src={staticFile('testsrc.mp4')} style={{ width: 800, height: 450 }} muted />
    </div>
  </AbsoluteFill>
);

/** Audio tag + a frame-driven "meter" so the still has a visual too */
const AudioBlock: React.FC = () => {
  const frame = useCurrentFrame();
  const level = Math.abs(Math.sin(frame / 3));
  return (
    <AbsoluteFill style={{ backgroundColor: '#1e1b4b' }}>
      <Caption text="04 Audio" />
      <Audio src={staticFile('tone.wav')} volume={0.5} />
      <div style={{ position: 'absolute', left: 120, right: 120, top: 320, height: 80, backgroundColor: '#312e81', borderRadius: 40, overflow: 'hidden' }}>
        <div style={{ width: `${level * 100}%`, height: '100%', backgroundColor: '#818cf8' }} />
      </div>
    </AbsoluteFill>
  );
};

/**
 * TransitionSeries with two toolkit transitions (lib/transitions).
 * Timeline (local frames): A 0-19, glitch overlaps 11-19, B 11-25, zoomBlur overlaps
 * 17-25, C 17-30. The sampled frame (15) sits at glitch progress 0.5.
 */
const TransitionBlock: React.FC = () => (
  <AbsoluteFill>
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={19}>
        <AbsoluteFill style={{ backgroundColor: '#dc2626' }} />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={glitch({ intensity: 0.8, slices: 8 })} timing={linearTiming({ durationInFrames: 8 })} />
      <TransitionSeries.Sequence durationInFrames={14}>
        <AbsoluteFill style={{ backgroundColor: '#2563eb' }} />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition presentation={zoomBlur({ direction: 'in' })} timing={linearTiming({ durationInFrames: 8 })} />
      <TransitionSeries.Sequence durationInFrames={13}>
        <AbsoluteFill style={{ backgroundColor: '#16a34a' }} />
      </TransitionSeries.Sequence>
    </TransitionSeries>
    <Caption text="05 TransitionSeries / lib glitch + zoomBlur" />
  </AbsoluteFill>
);

/** Shared components from lib/components */
const ComponentsBlock: React.FC = () => (
  <AbsoluteFill>
    <AnimatedBackground variant="tech" showGrid />
    <Label text="lib/components" jiraRef="BASE-1" size="lg" />
    <Vignette intensity={0.6} />
    <FilmGrain opacity={0.12} animate />
    <Caption text="06 AnimatedBackground / Label / Vignette / FilmGrain" />
  </AbsoluteFill>
);

export const BLOCKS: React.FC[] = [EasingBlock, SequenceBlock, VideoBlock, AudioBlock, TransitionBlock, ComponentsBlock];

// ─── Composition ─────────────────────────────────────────────

export const Baseline: React.FC<{ blocks: number }> = () => {
  return (
    <ThemeProvider theme={theme}>
      <AbsoluteFill style={{ fontFamily: FONT }}>
        <Series>
          {BLOCKS.map((Block, i) => (
            <Series.Sequence key={i} durationInFrames={FRAMES_PER_BLOCK}>
              <Block />
            </Series.Sequence>
          ))}
        </Series>
      </AbsoluteFill>
    </ThemeProvider>
  );
};
