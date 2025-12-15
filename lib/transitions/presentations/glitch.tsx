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
import React, { useMemo } from 'react';
import { AbsoluteFill, random, interpolate } from 'remotion';

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

  // For exiting scene, we reverse the effect
  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Glitch is most intense in the middle of the transition
  const glitchIntensity = useMemo(() => {
    const peak = interpolate(progress, [0, 0.5, 1], [0, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    return peak * intensity;
  }, [progress, intensity]);

  // Generate deterministic slice offsets
  const sliceOffsets = useMemo(() => {
    return Array.from({ length: slices }, (_, i) => {
      const seed = `glitch-slice-${i}`;
      const baseOffset = (random(seed) - 0.5) * 60 * glitchIntensity;
      // Add some temporal variation
      const flicker = random(`${seed}-${Math.floor(progress * 10)}`) > 0.7 ? 1.5 : 1;
      return baseOffset * flicker;
    });
  }, [slices, glitchIntensity, progress]);

  // RGB shift amounts
  const rgbShiftAmount = rgbShift ? glitchIntensity * 8 : 0;

  // Opacity for the entering/exiting effect
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.3], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.7, 1], [0, 1], { extrapolateLeft: 'clamp' });

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
    position: 'relative',
    overflow: 'hidden',
  }), []);

  const sliceHeight = 100 / slices;

  return (
    <AbsoluteFill style={containerStyle}>
      {/* Main content with slice displacement */}
      <AbsoluteFill style={{ opacity }}>
        {sliceOffsets.map((offset, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              top: `${i * sliceHeight}%`,
              left: 0,
              width: '100%',
              height: `${sliceHeight + 0.5}%`, // Slight overlap to prevent gaps
              overflow: 'hidden',
              transform: `translateX(${offset}px)`,
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: `-${i * sliceHeight}%`,
                left: 0,
                width: '100%',
                height: `${100 / sliceHeight * 100}%`,
              }}
            >
              {children}
            </div>
          </div>
        ))}
      </AbsoluteFill>

      {/* RGB channel separation overlay */}
      {rgbShift && glitchIntensity > 0.1 && (
        <>
          {/* Red channel */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.5 * glitchIntensity,
              transform: `translateX(${-rgbShiftAmount}px)`,
              mixBlendMode: 'screen',
              filter: 'url(#redChannel)',
            }}
          >
            {children}
          </AbsoluteFill>
          {/* Cyan channel */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.5 * glitchIntensity,
              transform: `translateX(${rgbShiftAmount}px)`,
              mixBlendMode: 'screen',
              filter: 'url(#cyanChannel)',
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
            opacity: glitchIntensity * 0.3,
            background: `repeating-linear-gradient(
              0deg,
              transparent,
              transparent 2px,
              rgba(0, 0, 0, 0.3) 2px,
              rgba(0, 0, 0, 0.3) 4px
            )`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Noise overlay for texture */}
      {glitchIntensity > 0.2 && (
        <AbsoluteFill
          style={{
            opacity: glitchIntensity * 0.15,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
          }}
        />
      )}

      {/* SVG filters for RGB separation */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <filter id="redChannel">
            <feColorMatrix
              type="matrix"
              values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"
            />
          </filter>
          <filter id="cyanChannel">
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
