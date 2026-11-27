/**
 * Transition Gallery - Local copy for testing
 *
 * This is a local copy that resolves @remotion/transitions from
 * the template's node_modules. The source of truth is in lib/transitions/.
 */
import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, Sequence } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { slide } from '@remotion/transitions/slide';
import { fade } from '@remotion/transitions/fade';
import { wipe } from '@remotion/transitions/wipe';
import { flip } from '@remotion/transitions/flip';

// Import custom transitions from lib (relative path works for non-package imports)
import { glitch } from '../../../lib/transitions/presentations/glitch';
import { rgbSplit } from '../../../lib/transitions/presentations/rgb-split';
import { zoomBlur } from '../../../lib/transitions/presentations/zoom-blur';
import { lightLeak } from '../../../lib/transitions/presentations/light-leak';
import { clockWipe } from '../../../lib/transitions/presentations/clock-wipe';
import { pixelate } from '../../../lib/transitions/presentations/pixelate';

// Rich scene component with detailed content for better transition visibility
const GalleryScene: React.FC<{
  label: string;
  isAfter?: boolean;
}> = ({ label, isAfter = false }) => {
  const frame = useCurrentFrame();
  const labelOpacity = interpolate(frame, [0, 10], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Different visual styles for before/after
  const bgGradient = isAfter
    ? 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)'
    : 'linear-gradient(135deg, #2d132c 0%, #801336 50%, #c72c41 100%)';

  const accentColor = isAfter ? '#00d9ff' : '#ffd700';

  return (
    <AbsoluteFill
      style={{
        background: bgGradient,
        fontFamily: "'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      {/* Grid pattern for edge visibility */}
      <AbsoluteFill
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Decorative circles */}
      <div
        style={{
          position: 'absolute',
          top: '15%',
          left: '10%',
          width: 200,
          height: 200,
          borderRadius: '50%',
          border: `3px solid ${accentColor}`,
          opacity: 0.6,
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: '20%',
          right: '15%',
          width: 150,
          height: 150,
          borderRadius: '50%',
          backgroundColor: accentColor,
          opacity: 0.2,
        }}
      />

      {/* Decorative lines */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          right: '10%',
          width: 200,
          height: 4,
          backgroundColor: accentColor,
          opacity: 0.5,
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '32%',
          right: '10%',
          width: 120,
          height: 4,
          backgroundColor: 'white',
          opacity: 0.3,
        }}
      />

      {/* Main content card */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          padding: '60px 100px',
          backgroundColor: 'rgba(0,0,0,0.4)',
          borderRadius: 20,
          border: `2px solid ${accentColor}`,
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: 80,
            fontWeight: 800,
            color: 'white',
            letterSpacing: '-2px',
            marginBottom: 10,
          }}
        >
          {isAfter ? 'SCENE B' : 'SCENE A'}
        </div>
        <div
          style={{
            fontSize: 24,
            color: accentColor,
            fontWeight: 600,
            letterSpacing: '4px',
          }}
        >
          {isAfter ? 'DESTINATION' : 'ORIGIN'}
        </div>
      </div>

      {/* Transition name label */}
      <div
        style={{
          position: 'absolute',
          top: 40,
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
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            padding: '14px 40px',
            borderRadius: 12,
            letterSpacing: '1px',
            border: `1px solid ${accentColor}`,
          }}
        >
          {label}
        </span>
      </div>

      {/* Corner decorations */}
      <div style={{ position: 'absolute', top: 40, left: 40, color: accentColor, fontSize: 40 }}>◢</div>
      <div style={{ position: 'absolute', top: 40, right: 40, color: accentColor, fontSize: 40 }}>◣</div>
      <div style={{ position: 'absolute', bottom: 40, left: 40, color: accentColor, fontSize: 40 }}>◥</div>
      <div style={{ position: 'absolute', bottom: 40, right: 40, color: accentColor, fontSize: 40 }}>◤</div>

      {/* Sample text for edge detection */}
      <div
        style={{
          position: 'absolute',
          bottom: 100,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontSize: 16,
          color: 'rgba(255,255,255,0.4)',
          letterSpacing: '2px',
        }}
      >
        TRANSITION GALLERY • AGENT VIDEO TOOLKIT
      </div>
    </AbsoluteFill>
  );
};

// Single transition demo segment
const TransitionDemo: React.FC<{
  name: string;
  presentation: ReturnType<typeof glitch>;
  transitionDuration?: number;
}> = ({ name, presentation, transitionDuration = 60 }) => {
  const sceneBeforeDuration = 45; // 1.5 seconds before transition
  const sceneAfterDuration = 105; // 3.5 seconds after (includes 2s break before next)

  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={sceneBeforeDuration}>
        <GalleryScene label={name} />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={presentation}
        timing={linearTiming({ durationInFrames: transitionDuration })}
      />
      <TransitionSeries.Sequence durationInFrames={sceneAfterDuration}>
        <GalleryScene label={name} isAfter />
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
        agent-video-toolkit
      </p>
    </AbsoluteFill>
  );
};

// Define all transitions to showcase
const TRANSITIONS = [
  { name: 'glitch()', presentation: glitch({ intensity: 0.9 }), duration: 25 },
  { name: 'rgbSplit()', presentation: rgbSplit({ direction: 'horizontal' }), duration: 25 },
  { name: 'zoomBlur()', presentation: zoomBlur({ direction: 'in' }), duration: 25 },
  { name: 'lightLeak()', presentation: lightLeak({ temperature: 'warm' }), duration: 35 },
  { name: 'clockWipe()', presentation: clockWipe({ direction: 'clockwise' }), duration: 25 },
  { name: 'pixelate()', presentation: pixelate({ maxBlockSize: 50 }), duration: 25 },
  { name: 'slide()', presentation: slide(), duration: 20 },
  { name: 'fade()', presentation: fade(), duration: 25 },
  { name: 'wipe()', presentation: wipe(), duration: 20 },
  { name: 'flip()', presentation: flip(), duration: 25 },
];

// Calculate segment duration (scene + transition + scene, minus overlap)
const getSegmentDuration = (transitionDuration: number) => {
  const sceneBeforeDuration = 45;
  const sceneAfterDuration = 105;
  return sceneBeforeDuration + sceneAfterDuration - transitionDuration;
};

export const TransitionGallery: React.FC = () => {
  const introDuration = 60; // 2 seconds

  let currentFrame = introDuration;
  const segments: { name: string; from: number; duration: number }[] = [];

  TRANSITIONS.forEach((t) => {
    const duration = getSegmentDuration(t.duration);
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
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

// Export configuration for Remotion
export const transitionGalleryConfig = {
  id: 'TransitionGallery',
  component: TransitionGallery,
  durationInFrames: 60 + TRANSITIONS.reduce(
    (acc, t) => acc + getSegmentDuration(t.duration),
    0
  ),
  fps: 30,
  width: 1920,
  height: 1080,
};
