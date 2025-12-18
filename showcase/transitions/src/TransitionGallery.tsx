/**
 * Transition Gallery
 *
 * A visual showcase of all available transitions.
 * Each transition is demonstrated with consistent before/after scenes,
 * labeled clearly for easy comparison.
 */
import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, Sequence } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { slide } from '@remotion/transitions/slide';
import { fade } from '@remotion/transitions/fade';
import { wipe } from '@remotion/transitions/wipe';
import { flip } from '@remotion/transitions/flip';
// Custom transitions from lib
import { glitch } from '../../../lib/transitions/presentations/glitch';
import { rgbSplit } from '../../../lib/transitions/presentations/rgb-split';
import { zoomBlur } from '../../../lib/transitions/presentations/zoom-blur';
import { lightLeak } from '../../../lib/transitions/presentations/light-leak';
import { clockWipe } from '../../../lib/transitions/presentations/clock-wipe';
import { pixelate } from '../../../lib/transitions/presentations/pixelate';
import { checkerboard } from '../../../lib/transitions/presentations/checkerboard';

// Scene colors for visual variety
const SCENE_A_COLOR = '#1a1a2e';
const SCENE_B_COLOR = '#e94560';

// Transition descriptions for context
const TRANSITION_NOTES: Record<string, string> = {
  'glitch()': 'Digital distortion with RGB shift',
  'rgbSplit()': 'Chromatic aberration effect',
  'zoomBlur()': 'Radial motion blur',
  'lightLeak()': 'Cinematic lens flare',
  'clockWipe()': 'Radial sweep reveal',
  'pixelate()': 'Mosaic dissolution',
  'checkerboard()': 'Grid squares reveal',
  'checkerboard(diagonal)': 'Diagonal wave pattern',
  'checkerboard(alternating)': 'True checkerboard pattern',
  'checkerboard(spiral)': 'Spiral from center',
  'checkerboard(center-out)': 'Radial grid reveal',
  'slide()': 'Push from direction',
  'fade()': 'Simple crossfade',
  'wipe()': 'Edge reveal',
  'flip()': '3D card flip',
};

// Consistent scene component
const GalleryScene: React.FC<{
  color: string;
  label: string;
  isAfter?: boolean;
}> = ({ color, label, isAfter = false }) => {
  const frame = useCurrentFrame();
  const labelOpacity = interpolate(frame, [0, 10], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const sceneName = isAfter ? 'Scene B' : 'Scene A';
  const note = TRANSITION_NOTES[label] || '';

  return (
    <AbsoluteFill
      style={{
        backgroundColor: color,
        fontFamily: "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      {/* Background grid pattern for visual texture */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Corner markers to show scene boundaries */}
      {['top-left', 'top-right', 'bottom-left', 'bottom-right'].map((corner) => (
        <div
          key={corner}
          style={{
            position: 'absolute',
            width: 40,
            height: 40,
            borderColor: 'rgba(255,255,255,0.2)',
            borderStyle: 'solid',
            borderWidth: 0,
            ...(corner.includes('top') ? { top: 30 } : { bottom: 30 }),
            ...(corner.includes('left') ? { left: 30 } : { right: 30 }),
            ...(corner.includes('top') && corner.includes('left') && { borderTopWidth: 2, borderLeftWidth: 2 }),
            ...(corner.includes('top') && corner.includes('right') && { borderTopWidth: 2, borderRightWidth: 2 }),
            ...(corner.includes('bottom') && corner.includes('left') && { borderBottomWidth: 2, borderLeftWidth: 2 }),
            ...(corner.includes('bottom') && corner.includes('right') && { borderBottomWidth: 2, borderRightWidth: 2 }),
          }}
        />
      ))}

      {/* Transition name label - top center */}
      <div
        style={{
          position: 'absolute',
          top: 50,
          left: 0,
          right: 0,
          textAlign: 'center',
          opacity: labelOpacity,
        }}
      >
        <span
          style={{
            fontSize: 32,
            fontWeight: 700,
            color: 'white',
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            padding: '14px 40px',
            borderRadius: 12,
            letterSpacing: '0.5px',
          }}
        >
          {label}
        </span>
      </div>

      {/* Main scene indicator - center */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        {/* Large letter */}
        <div
          style={{
            fontSize: 200,
            fontWeight: 900,
            color: 'rgba(255, 255, 255, 0.08)',
            letterSpacing: '-8px',
            lineHeight: 1,
          }}
        >
          {isAfter ? 'B' : 'A'}
        </div>

        {/* Scene name */}
        <div
          style={{
            fontSize: 48,
            fontWeight: 700,
            color: 'white',
            marginTop: -20,
            textTransform: 'uppercase',
            letterSpacing: '8px',
          }}
        >
          {sceneName}
        </div>
      </div>

      {/* Transition description - bottom center */}
      <div
        style={{
          position: 'absolute',
          bottom: 80,
          left: 0,
          right: 0,
          textAlign: 'center',
          opacity: labelOpacity * 0.7,
        }}
      >
        <span
          style={{
            fontSize: 20,
            fontWeight: 400,
            color: 'rgba(255, 255, 255, 0.6)',
            fontStyle: 'italic',
          }}
        >
          {note}
        </span>
      </div>

      {/* Side label showing transition direction */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          transform: 'translateY(-50%)',
          ...(isAfter ? { right: 40 } : { left: 40 }),
          writingMode: 'vertical-rl',
          textOrientation: 'mixed',
          fontSize: 14,
          fontWeight: 500,
          color: 'rgba(255, 255, 255, 0.3)',
          letterSpacing: '3px',
          textTransform: 'uppercase',
        }}
      >
        {isAfter ? 'Entering' : 'Exiting'}
      </div>
    </AbsoluteFill>
  );
};

// Single transition demo segment
const TransitionDemo: React.FC<{
  name: string;
  presentation: ReturnType<typeof glitch>;
  transitionDuration?: number;
  sceneADuration?: number;
  sceneBDuration?: number;
}> = ({ name, presentation, transitionDuration = 30, sceneADuration = 90, sceneBDuration = 90 }) => {
  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={sceneADuration}>
        <GalleryScene color={SCENE_A_COLOR} label={name} />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={presentation}
        timing={linearTiming({ durationInFrames: transitionDuration })}
      />
      <TransitionSeries.Sequence durationInFrames={sceneBDuration}>
        <GalleryScene color={SCENE_B_COLOR} label={name} isAfter />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};

// Intro slide
const IntroSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: 'clamp' });
  const subtitleOpacity = interpolate(frame, [15, 35], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#0f0f1a',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <h1
        style={{
          fontSize: 72,
          fontWeight: 700,
          color: 'white',
          margin: 0,
          opacity: titleOpacity,
          letterSpacing: '-2px',
        }}
      >
        Transitions Gallery
      </h1>
      <p
        style={{
          fontSize: 24,
          color: 'rgba(255, 255, 255, 0.6)',
          marginTop: 20,
          opacity: subtitleOpacity,
          fontWeight: 400,
        }}
      >
        claude-code-video-toolkit
      </p>
    </AbsoluteFill>
  );
};

// Define all transitions to showcase
// Transition durations: 45 frames = 1.5s for most, longer for complex effects
const TRANSITIONS = [
  { name: 'glitch()', presentation: glitch({ intensity: 0.9 }), duration: 45 },
  { name: 'rgbSplit()', presentation: rgbSplit({ direction: 'horizontal' }), duration: 45 },
  { name: 'zoomBlur()', presentation: zoomBlur({ direction: 'in' }), duration: 45 },
  { name: 'lightLeak()', presentation: lightLeak({ temperature: 'warm' }), duration: 60 },
  { name: 'clockWipe()', presentation: clockWipe({ direction: 'clockwise' }), duration: 50, sceneA: 60, sceneB: 90 },
  { name: 'pixelate()', presentation: pixelate({ maxBlockSize: 50 }), duration: 45 },
  { name: 'checkerboard(diagonal)', presentation: checkerboard({ pattern: 'diagonal', gridSize: 8 }), duration: 50 },
  { name: 'checkerboard(alternating)', presentation: checkerboard({ pattern: 'alternating', gridSize: 8 }), duration: 50 },
  { name: 'checkerboard(spiral)', presentation: checkerboard({ pattern: 'spiral', gridSize: 10 }), duration: 55 },
  { name: 'checkerboard(center-out)', presentation: checkerboard({ pattern: 'center-out', gridSize: 8, squareAnimation: 'scale' }), duration: 50 },
  { name: 'slide()', presentation: slide(), duration: 40 },
  { name: 'fade()', presentation: fade(), duration: 45 },
  { name: 'wipe()', presentation: wipe(), duration: 40 },
  { name: 'flip()', presentation: flip(), duration: 45 },
] as const;

// Calculate segment duration (scene A + scene B - overlap from transition)
const getSegmentDuration = (t: { duration: number; sceneA?: number; sceneB?: number }) => {
  const sceneA = t.sceneA ?? 90;
  const sceneB = t.sceneB ?? 90;
  return sceneA + sceneB - t.duration;
};

export const TransitionGallery: React.FC = () => {
  const introDuration = 60;

  let currentFrame = introDuration;
  const segments: { name: string; from: number; duration: number }[] = [];

  TRANSITIONS.forEach((t) => {
    const duration = getSegmentDuration(t);
    segments.push({ name: t.name, from: currentFrame, duration });
    currentFrame += duration;
  });

  return (
    <AbsoluteFill style={{ backgroundColor: '#0f0f1a' }}>
      {/* Intro */}
      <Sequence durationInFrames={introDuration}>
        <IntroSlide />
      </Sequence>

      {/* Each transition demo */}
      {TRANSITIONS.map((t, i) => (
        <Sequence
          key={t.name}
          from={segments[i].from}
          durationInFrames={segments[i].duration}
        >
          <TransitionDemo
            name={t.name}
            presentation={t.presentation}
            transitionDuration={t.duration}
            sceneADuration={'sceneA' in t ? t.sceneA : 90}
            sceneBDuration={'sceneB' in t ? t.sceneB : 90}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

// Export configuration
export const transitionGalleryConfig = {
  id: 'TransitionGallery',
  component: TransitionGallery,
  durationInFrames: 60 + TRANSITIONS.reduce(
    (acc, t) => acc + getSegmentDuration(t),
    0
  ),
  fps: 30,
  width: 1920,
  height: 1080,
};
