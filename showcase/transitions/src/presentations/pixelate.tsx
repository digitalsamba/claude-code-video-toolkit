/**
 * Pixelate Transition
 *
 * Digital pixelation/mosaic effect that dissolves the scene into blocks.
 * Creates a retro gaming or digital artifact aesthetic.
 *
 * Best for: Tech themes, retro/gaming content, digital transformations
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo, useState } from 'react';
import { AbsoluteFill, interpolate, random } from 'remotion';

export type PixelateProps = {
  /** Maximum block size at peak pixelation. Default: 40 */
  maxBlockSize?: number;
  /** Include color posterization with pixelation. Default: true */
  posterize?: boolean;
  /** Pixelation pattern: 'uniform' or 'random'. Default: 'uniform' */
  pattern?: 'uniform' | 'random';
};

const PixelatePresentation: React.FC<
  TransitionPresentationComponentProps<PixelateProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    maxBlockSize = 40,
    posterize = true,
    pattern = 'uniform',
  } = passedProps;

  const [filterId] = useState(() => `pixelate-${String(random(null)).slice(2, 10)}`);

  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Pixelation intensity peaks in the middle
  const pixelIntensity = useMemo(() => {
    return interpolate(progress, [0, 0.5, 1], [0, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Block size calculation (starts small, gets big, then small again)
  const blockSize = useMemo(() => {
    const minSize = 1;
    return Math.max(minSize, Math.round(maxBlockSize * pixelIntensity));
  }, [maxBlockSize, pixelIntensity]);

  // Opacity for entering/exiting
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.5], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.5, 1], [0, 1], { extrapolateLeft: 'clamp' });

  // Posterization reduces color depth
  const posterizeLevels = posterize
    ? Math.max(2, Math.round(interpolate(pixelIntensity, [0, 1], [256, 4])))
    : 256;

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
    imageRendering: blockSize > 2 ? 'pixelated' : 'auto',
  }), [blockSize]);

  // For pattern === 'random', we add noise-based distortion
  const shouldApplyEffect = pixelIntensity > 0.05;

  return (
    <AbsoluteFill style={containerStyle}>
      <AbsoluteFill
        style={{
          opacity,
          filter: shouldApplyEffect ? `url(#${filterId})` : undefined,
        }}
      >
        {children}
      </AbsoluteFill>

      {/* Scanline overlay for CRT effect */}
      {pixelIntensity > 0.3 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.2,
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

      {/* Color banding effect */}
      {posterize && pixelIntensity > 0.2 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.15,
            background: `linear-gradient(
              180deg,
              rgba(0, 255, 0, 0.05) 0%,
              rgba(255, 0, 255, 0.05) 50%,
              rgba(0, 255, 255, 0.05) 100%
            )`,
            mixBlendMode: 'overlay',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* SVG filter for pixelation effect */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <filter id={filterId} x="0%" y="0%" width="100%" height="100%">
            {/* Pixelation via mosaic effect */}
            {blockSize > 1 && (
              <>
                {/* Scale down */}
                <feImage
                  result="scaled"
                  width={`${100 / blockSize}%`}
                  height={`${100 / blockSize}%`}
                  preserveAspectRatio="none"
                />
                {/* Create mosaic tiles */}
                <feMorphology
                  operator="dilate"
                  radius={Math.max(1, blockSize / 4)}
                  in="SourceGraphic"
                  result="dilated"
                />
                <feGaussianBlur
                  stdDeviation={blockSize / 2}
                  in="SourceGraphic"
                  result="blurred"
                />
                <feComponentTransfer in="blurred" result="posterized">
                  {posterize && posterizeLevels < 256 && (
                    <>
                      <feFuncR type="discrete" tableValues={generatePosterizeTable(posterizeLevels)} />
                      <feFuncG type="discrete" tableValues={generatePosterizeTable(posterizeLevels)} />
                      <feFuncB type="discrete" tableValues={generatePosterizeTable(posterizeLevels)} />
                    </>
                  )}
                </feComponentTransfer>
              </>
            )}

            {/* Add subtle noise for random pattern */}
            {pattern === 'random' && pixelIntensity > 0.3 && (
              <>
                <feTurbulence
                  type="fractalNoise"
                  baseFrequency={0.05}
                  numOctaves={1}
                  result="noise"
                />
                <feDisplacementMap
                  in="posterized"
                  in2="noise"
                  scale={pixelIntensity * 10}
                  xChannelSelector="R"
                  yChannelSelector="G"
                />
              </>
            )}
          </filter>
        </defs>
      </svg>
    </AbsoluteFill>
  );
};

// Generate posterization lookup table
function generatePosterizeTable(levels: number): string {
  const step = 1 / (levels - 1);
  return Array.from({ length: levels }, (_, i) => (i * step).toFixed(3)).join(' ');
}

export const pixelate = (
  props: PixelateProps = {}
): TransitionPresentation<PixelateProps> => {
  return { component: PixelatePresentation, props };
};
