/**
 * Glitch Transition
 *
 * A digital distortion effect perfect for tech-focused videos.
 * Creates horizontal slice displacement, RGB channel separation,
 * and scan line artifacts for an authentic glitch aesthetic.
 *
 * Best for: Tech demos, cyberpunk themes, edgy reveals
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo, useState } from 'react';
import { AbsoluteFill, random, interpolate, useCurrentFrame } from 'remotion';

export type GlitchProps = {
  /** Intensity of the glitch effect (0-1). Default: 0.8 */
  intensity?: number;
  /** Number of horizontal slices. Default: 8 */
  slices?: number;
  /** Include RGB channel separation. Default: true */
  rgbShift?: boolean;
  /** Include scan lines overlay. Default: true */
  scanLines?: boolean;
};

const GlitchPresentation: React.FC<
  TransitionPresentationComponentProps<GlitchProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    intensity = 0.8,
    slices = 8,
    rgbShift = true,
    scanLines = true,
  } = passedProps;

  // Unique filter IDs for this instance
  const [filterId] = useState(() => `glitch-${String(random(null)).slice(2, 10)}`);

  // Get actual video frame for timing flicker effects
  const frame = useCurrentFrame();

  // Glitch intensity peaks in the middle of the transition
  const glitchIntensity = useMemo(() => {
    const peak = interpolate(presentationProgress, [0, 0.5, 1], [0, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    return peak * intensity;
  }, [presentationProgress, intensity]);

  // Flicker changes every 2-3 frames for rapid but not every-frame chaos
  const flickerFrame = Math.floor(frame / 2);

  // Generate deterministic slice offsets based on current progress
  const sliceOffsets = useMemo(() => {
    return Array.from({ length: slices }, (_, i) => {
      // Different offset per flicker frame for rapid movement
      const seed = `glitch-slice-${i}-${flickerFrame}`;
      const baseOffset = (random(seed) - 0.5) * 200 * glitchIntensity;
      // Aggressive flicker - some slices jump dramatically
      const jumpSeed = `jump-${i}-${flickerFrame}`;
      const jump = random(jumpSeed) > 0.4 ? (random(`${jumpSeed}-dir`) > 0.5 ? 2.5 : -2.5) : 1;
      return baseOffset * jump;
    });
  }, [slices, glitchIntensity, flickerFrame]);

  // RGB shift amount - much more aggressive
  const rgbShiftAmount = rgbShift ? glitchIntensity * 25 : 0;

  // Rapid RGB flicker
  const rgbFlicker = random(`rgb-${flickerFrame}`) > 0.3 ? 1 : 0.3;

  // Simple linear crossfade opacity
  const opacity = presentationDirection === 'exiting'
    ? interpolate(presentationProgress, [0, 1], [1, 0])
    : interpolate(presentationProgress, [0, 1], [0, 1]);

  const sliceHeight = 100 / slices;

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* Main content with slice displacement */}
      <AbsoluteFill style={{ opacity }}>
        {glitchIntensity < 0.01 ? (
          // TransitionSeries keeps the presentation mounted for the whole sequence
          // (progress 0 before an exit, 1 after an entry), so pass the scene through
          // untouched whenever the effect is idle.
          children
        ) : (
          // Full-size copies clipped to exact bands, so the slices line up without seams.
          sliceOffsets.map((offset, i) => (
            <AbsoluteFill
              key={i}
              style={{
                clipPath: `inset(${i * sliceHeight}% 0 ${Math.max(0, 100 - (i + 1) * sliceHeight)}% 0)`,
                transform: `translateX(${offset}px)`,
              }}
            >
              {children}
            </AbsoluteFill>
          ))
        )}
      </AbsoluteFill>

      {/* RGB channel separation overlay */}
      {rgbShift && glitchIntensity > 0.05 && (
        <>
          {/* Red channel - shifted left */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.6 * glitchIntensity * rgbFlicker,
              transform: `translateX(${-rgbShiftAmount}px) translateY(${(random(`rgb-y-${flickerFrame}`) - 0.5) * 10 * glitchIntensity}px)`,
              mixBlendMode: 'screen',
              filter: `url(#${filterId}-red)`,
            }}
          >
            {children}
          </AbsoluteFill>
          {/* Cyan channel - shifted right */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.6 * glitchIntensity * rgbFlicker,
              transform: `translateX(${rgbShiftAmount}px) translateY(${(random(`rgb-y2-${flickerFrame}`) - 0.5) * 10 * glitchIntensity}px)`,
              mixBlendMode: 'screen',
              filter: `url(#${filterId}-cyan)`,
            }}
          >
            {children}
          </AbsoluteFill>
        </>
      )}

      {/* Scan lines overlay */}
      {scanLines && glitchIntensity > 0.1 && (
        <AbsoluteFill
          style={{
            opacity: glitchIntensity * 0.4,
            background: `repeating-linear-gradient(
              0deg,
              transparent,
              transparent 2px,
              rgba(0, 0, 0, 0.4) 2px,
              rgba(0, 0, 0, 0.4) 4px
            )`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Noise overlay for texture */}
      {glitchIntensity > 0.15 && (
        <AbsoluteFill
          style={{
            opacity: glitchIntensity * 0.2,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
          }}
        />
      )}

      {/* Random glitch blocks - more of them, more aggressive */}
      {glitchIntensity > 0.15 && (
        <AbsoluteFill style={{ pointerEvents: 'none' }}>
          {Array.from({ length: 8 }, (_, i) => {
            const blockSeed = `block-${i}-${flickerFrame}`;
            const show = random(blockSeed) > 0.4;
            if (!show) return null;

            const x = random(`${blockSeed}-x`) * 100;
            const y = random(`${blockSeed}-y`) * 100;
            const w = 5 + random(`${blockSeed}-w`) * 40;
            const h = 1 + random(`${blockSeed}-h`) * 15;

            const colorChoice = random(`${blockSeed}-c`);
            let bgColor;
            if (colorChoice > 0.7) {
              bgColor = `rgba(255, 255, 255, ${glitchIntensity * 0.5})`;
            } else if (colorChoice > 0.4) {
              bgColor = `rgba(255, 0, 80, ${glitchIntensity * 0.6})`;
            } else {
              bgColor = `rgba(0, 255, 255, ${glitchIntensity * 0.6})`;
            }

            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: `${x}%`,
                  top: `${y}%`,
                  width: `${w}%`,
                  height: `${h}%`,
                  backgroundColor: bgColor,
                  mixBlendMode: 'screen',
                }}
              />
            );
          })}
        </AbsoluteFill>
      )}

      {/* White flash on peak glitch */}
      {glitchIntensity > 0.7 && random(`flash-${flickerFrame}`) > 0.6 && (
        <AbsoluteFill
          style={{
            backgroundColor: 'white',
            opacity: glitchIntensity * 0.15,
            mixBlendMode: 'overlay',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* SVG filters for RGB separation */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <filter id={`${filterId}-red`}>
            <feColorMatrix
              type="matrix"
              values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"
            />
          </filter>
          <filter id={`${filterId}-cyan`}>
            <feColorMatrix
              type="matrix"
              values="0 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0"
            />
          </filter>
        </defs>
      </svg>
    </AbsoluteFill>
  );
};

export const glitch = (
  props: GlitchProps = {}
): TransitionPresentation<GlitchProps> => {
  return { component: GlitchPresentation, props };
};
