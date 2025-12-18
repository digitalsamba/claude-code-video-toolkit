/**
 * Pixelate Transition
 *
 * Digital pixelation/mosaic effect that dissolves the scene into blocks.
 * Features randomized block reveals and glitchy artifacts.
 *
 * Best for: Tech themes, retro/gaming content, digital transformations
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo } from 'react';
import { AbsoluteFill, interpolate, random } from 'remotion';

export type PixelateProps = {
  /** Maximum block size at peak pixelation (pixels). Default: 60 */
  maxBlockSize?: number;
  /** Grid dimensions (e.g., 12 = 12x12 grid). Default: 12 */
  gridSize?: number;
  /** Add scanline overlay for CRT effect. Default: true */
  scanlines?: boolean;
  /** Add glitch artifacts during transition. Default: true */
  glitchArtifacts?: boolean;
  /** Randomness of block reveal (0-1). Default: 0.8 */
  randomness?: number;
};

// Generate pseudo-random value for a grid cell
const getCellRandom = (row: number, col: number, seed: number): number => {
  return random(`cell-${row}-${col}-${seed}`);
};

const PixelatePresentation: React.FC<
  TransitionPresentationComponentProps<PixelateProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    maxBlockSize = 60,
    gridSize = 12,
    scanlines = true,
    glitchArtifacts = true,
    randomness = 0.8,
  } = passedProps;

  const seed = 42;

  // For entering: 0→1, for exiting: treat as inverse
  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Pixelation intensity peaks in the middle
  const pixelIntensity = useMemo(() => {
    return interpolate(progress, [0, 0.4, 0.6, 1], [0, 1, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Block size grows then shrinks
  const blockSize = useMemo(() => {
    return Math.max(8, Math.round(maxBlockSize * pixelIntensity));
  }, [maxBlockSize, pixelIntensity]);

  // Heavy blur for mosaic effect
  const blurAmount = pixelIntensity * (maxBlockSize / 2.5);

  // Crossfade
  const opacity = presentationDirection === 'exiting'
    ? interpolate(presentationProgress, [0, 0.6, 1], [1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
    : interpolate(presentationProgress, [0, 0.4, 1], [0, 1, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

  // Generate grid cells with random reveal timing
  const cells = useMemo(() => {
    const result: Array<{
      row: number;
      col: number;
      revealOrder: number;
      hueShift: number;
    }> = [];

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const baseOrder = (row + col) / (gridSize * 2);
        const randOffset = getCellRandom(row, col, seed) * randomness;
        result.push({
          row,
          col,
          revealOrder: baseOrder * (1 - randomness) + randOffset,
          hueShift: getCellRandom(row, col, seed + 1) * 30 - 15,
        });
      }
    }
    return result;
  }, [gridSize, randomness, seed]);

  const cellSize = 100 / gridSize;

  // Glitch offset that changes during transition
  const glitchOffset = useMemo(() => {
    if (!glitchArtifacts || pixelIntensity < 0.3) return { x: 0, y: 0 };
    const intensity = (pixelIntensity - 0.3) / 0.7;
    return {
      x: Math.sin(progress * Math.PI * 8) * intensity * 15,
      y: Math.cos(progress * Math.PI * 6) * intensity * 8,
    };
  }, [glitchArtifacts, pixelIntensity, progress]);

  const shouldApplyEffect = pixelIntensity > 0.05;

  return (
    <AbsoluteFill style={{ backgroundColor: '#000' }}>
      {/* Main blurred content */}
      <AbsoluteFill
        style={{
          opacity,
          filter: shouldApplyEffect
            ? `blur(${blurAmount}px) saturate(140%) contrast(120%)`
            : undefined,
          transform: glitchArtifacts && pixelIntensity > 0.5
            ? `translate(${glitchOffset.x}px, ${glitchOffset.y}px)`
            : undefined,
        }}
      >
        {children}
      </AbsoluteFill>

      {/* Random block grid overlay */}
      {shouldApplyEffect && (
        <AbsoluteFill style={{ pointerEvents: 'none' }}>
          {cells.map(({ row, col, revealOrder, hueShift }) => {
            // Each cell reveals at different times
            const cellProgress = interpolate(
              pixelIntensity,
              [revealOrder * 0.5, revealOrder * 0.5 + 0.5],
              [0, 1],
              { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
            );

            if (cellProgress < 0.1) return null;

            const cellOpacity = cellProgress * 0.7;

            return (
              <div
                key={`${row}-${col}`}
                style={{
                  position: 'absolute',
                  left: `${col * cellSize}%`,
                  top: `${row * cellSize}%`,
                  width: `${cellSize}%`,
                  height: `${cellSize}%`,
                  backgroundColor: `hsla(${hueShift + 180}, 50%, 50%, ${cellOpacity * 0.15})`,
                  border: `1px solid rgba(0, 0, 0, ${cellOpacity})`,
                  boxSizing: 'border-box',
                }}
              />
            );
          })}
        </AbsoluteFill>
      )}

      {/* Pronounced grid lines */}
      {shouldApplyEffect && blockSize >= 8 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.8,
            backgroundImage: `
              linear-gradient(to right, rgba(0, 0, 0, 0.9) 2px, transparent 2px),
              linear-gradient(to bottom, rgba(0, 0, 0, 0.9) 2px, transparent 2px)
            `,
            backgroundSize: `${blockSize}px ${blockSize}px`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Glitch slices */}
      {glitchArtifacts && pixelIntensity > 0.4 && (
        <>
          {[0.15, 0.35, 0.55, 0.75, 0.9].map((pos, i) => {
            const sliceIntensity = (pixelIntensity - 0.4) / 0.6;
            const offset = Math.sin(progress * Math.PI * (4 + i)) * sliceIntensity * 20;
            const height = 3 + random(`slice-h-${i}`) * 8;

            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  top: `${pos * 100}%`,
                  height: `${height}%`,
                  transform: `translateX(${offset}px)`,
                  background: `linear-gradient(90deg,
                    transparent 0%,
                    rgba(255, 0, 128, ${sliceIntensity * 0.2}) 20%,
                    rgba(0, 255, 255, ${sliceIntensity * 0.2}) 80%,
                    transparent 100%
                  )`,
                  mixBlendMode: 'screen',
                  pointerEvents: 'none',
                }}
              />
            );
          })}
        </>
      )}

      {/* RGB split effect */}
      {glitchArtifacts && pixelIntensity > 0.5 && (
        <>
          <AbsoluteFill
            style={{
              opacity: (pixelIntensity - 0.5) * 0.4,
              transform: `translateX(${pixelIntensity * 8}px)`,
              filter: `blur(${blurAmount}px)`,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{ width: '100%', height: '100%', backgroundColor: 'rgba(255, 0, 0, 0.4)' }} />
          </AbsoluteFill>
          <AbsoluteFill
            style={{
              opacity: (pixelIntensity - 0.5) * 0.4,
              transform: `translateX(${-pixelIntensity * 8}px)`,
              filter: `blur(${blurAmount}px)`,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{ width: '100%', height: '100%', backgroundColor: 'rgba(0, 255, 255, 0.4)' }} />
          </AbsoluteFill>
        </>
      )}

      {/* Heavy scanlines */}
      {scanlines && pixelIntensity > 0.15 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.5,
            backgroundImage: `repeating-linear-gradient(
              0deg,
              transparent 0px,
              transparent 3px,
              rgba(0, 0, 0, 0.6) 3px,
              rgba(0, 0, 0, 0.6) 6px
            )`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Vignette */}
      {pixelIntensity > 0.2 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.7,
            background: `radial-gradient(
              ellipse at center,
              transparent 30%,
              rgba(0, 0, 0, 0.6) 100%
            )`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Noise overlay */}
      {pixelIntensity > 0.3 && (
        <AbsoluteFill
          style={{
            opacity: pixelIntensity * 0.15,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
          }}
        />
      )}
    </AbsoluteFill>
  );
};

export const pixelate = (
  props: PixelateProps = {}
): TransitionPresentation<PixelateProps> => {
  return { component: PixelatePresentation, props };
};
