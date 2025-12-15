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
      const baseOffset = (random(seed) - 0.5) * 80 * glitchIntensity;
      // Add temporal variation for flicker effect
      const flicker = random(`${seed}-${Math.floor(progress * 8)}`) > 0.6 ? 1.8 : 1;
      return baseOffset * flicker;
    });
  }, [slices, glitchIntensity, progress]);

  // RGB shift amounts
  const rgbShiftAmount = rgbShift ? glitchIntensity * 12 : 0;

  // Opacity for the entering/exiting effect
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.4], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.6, 1], [0, 1], { extrapolateLeft: 'clamp' });

  const sliceHeightPercent = 100 / slices;

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* Main content with slice displacement using clip-path */}
      <AbsoluteFill style={{ opacity }}>
        {sliceOffsets.map((offset, i) => {
          const topPercent = i * sliceHeightPercent;
          const bottomPercent = (i + 1) * sliceHeightPercent;

          return (
            <AbsoluteFill
              key={i}
              style={{
                clipPath: `polygon(0% ${topPercent}%, 100% ${topPercent}%, 100% ${bottomPercent}%, 0% ${bottomPercent}%)`,
                transform: `translateX(${offset}px)`,
              }}
            >
              {children}
            </AbsoluteFill>
          );
        })}
      </AbsoluteFill>

      {/* RGB channel separation overlay */}
      {rgbShift && glitchIntensity > 0.1 && (
        <>
          {/* Red channel - shifted left */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.4 * glitchIntensity,
              transform: `translateX(${-rgbShiftAmount}px)`,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{
              width: '100%',
              height: '100%',
              filter: 'saturate(2) hue-rotate(-30deg)',
              background: 'rgba(255, 0, 0, 0.3)',
              mixBlendMode: 'multiply',
            }}>
              {children}
            </div>
          </AbsoluteFill>
          {/* Cyan channel - shifted right */}
          <AbsoluteFill
            style={{
              opacity: opacity * 0.4 * glitchIntensity,
              transform: `translateX(${rgbShiftAmount}px)`,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{
              width: '100%',
              height: '100%',
              filter: 'saturate(2) hue-rotate(150deg)',
              background: 'rgba(0, 255, 255, 0.3)',
              mixBlendMode: 'multiply',
            }}>
              {children}
            </div>
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

      {/* Random block glitches */}
      {glitchIntensity > 0.3 && (
        <AbsoluteFill style={{ pointerEvents: 'none' }}>
          {Array.from({ length: 3 }, (_, i) => {
            const blockSeed = `block-${i}-${Math.floor(progress * 6)}`;
            const show = random(blockSeed) > 0.5;
            if (!show) return null;

            const x = random(`${blockSeed}-x`) * 80;
            const y = random(`${blockSeed}-y`) * 100;
            const w = 10 + random(`${blockSeed}-w`) * 30;
            const h = 2 + random(`${blockSeed}-h`) * 8;

            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: `${x}%`,
                  top: `${y}%`,
                  width: `${w}%`,
                  height: `${h}%`,
                  backgroundColor: random(`${blockSeed}-c`) > 0.5
                    ? `rgba(255, 0, 100, ${glitchIntensity * 0.5})`
                    : `rgba(0, 255, 200, ${glitchIntensity * 0.5})`,
                  mixBlendMode: 'screen',
                }}
              />
            );
          })}
        </AbsoluteFill>
      )}

      {/* Noise texture overlay */}
      {glitchIntensity > 0.2 && (
        <AbsoluteFill
          style={{
            opacity: glitchIntensity * 0.2,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
          }}
        />
      )}
    </AbsoluteFill>
  );
};

export const glitch = (
  props: GlitchProps = {}
): TransitionPresentation<GlitchProps> => {
  return { component: GlitchPresentation, props };
};
