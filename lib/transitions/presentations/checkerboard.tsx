/**
 * Checkerboard Transition
 *
 * Reveals the scene through a grid of squares with various patterns.
 * Classic video editing effect with modern flexibility.
 *
 * Best for: Playful reveals, retro aesthetics, creative transitions
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo } from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';

export type CheckerboardPattern =
  | 'sequential'    // Left-to-right, top-to-bottom
  | 'random'        // Randomized order
  | 'diagonal'      // Diagonal wave
  | 'alternating'   // True checkerboard - alternating squares first
  | 'spiral'        // From center outward in spiral
  | 'rows'          // Row by row
  | 'columns'       // Column by column
  | 'center-out'    // Radial from center
  | 'corners-in';   // From all corners to center

export type CheckerboardProps = {
  /** Grid size (e.g., 8 = 8x8 grid). Default: 8 */
  gridSize?: number;
  /** Reveal pattern. Default: 'diagonal' */
  pattern?: CheckerboardPattern;
  /** Stagger amount - how spread out the animation is (0-1). Default: 0.6 */
  stagger?: number;
  /** Individual square animation: 'fade' | 'scale' | 'flip'. Default: 'fade' */
  squareAnimation?: 'fade' | 'scale' | 'flip';
  /** Easing for individual squares. Default: ease-out */
  easing?: (t: number) => number;
};

// Generate order indices for each pattern
const generateOrder = (
  row: number,
  col: number,
  gridSize: number,
  pattern: CheckerboardPattern,
  seed: number
): number => {
  const total = gridSize * gridSize;
  const index = row * gridSize + col;
  const centerRow = (gridSize - 1) / 2;
  const centerCol = (gridSize - 1) / 2;

  switch (pattern) {
    case 'sequential':
      return index / total;

    case 'random':
      // Seeded pseudo-random based on position
      const hash = Math.sin(seed + index * 9999) * 10000;
      return (hash - Math.floor(hash));

    case 'diagonal':
      // Diagonal wave from top-left
      return (row + col) / (gridSize * 2 - 2);

    case 'alternating':
      // True checkerboard: alternating squares first (0-0.5), then others (0.5-1)
      const isAlternate = (row + col) % 2 === 0;
      const baseOrder = (row + col) / (gridSize * 2 - 2);
      return isAlternate ? baseOrder * 0.5 : 0.5 + baseOrder * 0.5;

    case 'spiral':
      // Spiral from center outward
      const distFromCenter = Math.max(
        Math.abs(row - centerRow),
        Math.abs(col - centerCol)
      );
      const maxDist = Math.max(centerRow, centerCol);
      const ring = distFromCenter / maxDist;
      // Add angle component for spiral effect
      const angle = Math.atan2(row - centerRow, col - centerCol);
      const normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
      return ring * 0.8 + normalizedAngle * 0.2;

    case 'rows':
      return row / (gridSize - 1);

    case 'columns':
      return col / (gridSize - 1);

    case 'center-out':
      // Radial from center
      const dist = Math.sqrt(
        Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2)
      );
      const maxRadius = Math.sqrt(2) * gridSize / 2;
      return dist / maxRadius;

    case 'corners-in':
      // From corners to center (inverse of center-out)
      const distCorners = Math.sqrt(
        Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2)
      );
      const maxRadiusCorners = Math.sqrt(2) * gridSize / 2;
      return 1 - (distCorners / maxRadiusCorners);

    default:
      return index / total;
  }
};

const CheckerboardPresentation: React.FC<
  TransitionPresentationComponentProps<CheckerboardProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    gridSize = 8,
    pattern = 'diagonal',
    stagger = 0.6,
    squareAnimation = 'fade',
    easing = Easing.out(Easing.cubic),
  } = passedProps;

  const isEntering = presentationDirection === 'entering';
  const progress = isEntering ? presentationProgress : 1 - presentationProgress;

  // Use a stable seed for random pattern
  const seed = 12345;

  // Generate grid cells
  const cells = useMemo(() => {
    const result: Array<{
      row: number;
      col: number;
      order: number;
    }> = [];

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        result.push({
          row,
          col,
          order: generateOrder(row, col, gridSize, pattern, seed),
        });
      }
    }

    return result;
  }, [gridSize, pattern, seed]);

  // Calculate cell size as percentage
  const cellSize = 100 / gridSize;

  return (
    <AbsoluteFill>
      {/* Base layer - for exiting scene or background */}
      {!isEntering && (
        <AbsoluteFill>
          {children}
        </AbsoluteFill>
      )}

      {/* Grid mask layer */}
      <AbsoluteFill style={{ overflow: 'hidden' }}>
        {cells.map(({ row, col, order }) => {
          // Calculate when this cell should animate
          // With stagger, cells animate in sequence
          // stagger=0 means all at once, stagger=1 means fully sequential
          const cellStart = order * stagger;
          const cellEnd = cellStart + (1 - stagger);

          // Individual cell progress
          const cellProgress = interpolate(
            progress,
            [cellStart, cellEnd],
            [0, 1],
            {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }
          );

          const easedProgress = easing(cellProgress);

          // Calculate animation values based on type
          let opacity = 1;
          let scale = 1;
          let rotateY = 0;

          switch (squareAnimation) {
            case 'fade':
              opacity = easedProgress;
              break;
            case 'scale':
              scale = easedProgress;
              opacity = easedProgress > 0 ? 1 : 0;
              break;
            case 'flip':
              rotateY = interpolate(easedProgress, [0, 1], [90, 0]);
              opacity = easedProgress > 0.1 ? 1 : 0;
              break;
          }

          // For exiting, we show the cell and hide it (inverse)
          if (!isEntering) {
            opacity = 1 - opacity;
            scale = scale === 0 ? 1 : 2 - scale;
            rotateY = -rotateY;
          }

          return (
            <div
              key={`${row}-${col}`}
              style={{
                position: 'absolute',
                left: `${col * cellSize}%`,
                top: `${row * cellSize}%`,
                width: `${cellSize}%`,
                height: `${cellSize}%`,
                overflow: 'hidden',
                opacity,
                transform: `scale(${scale}) perspective(500px) rotateY(${rotateY}deg)`,
                transformOrigin: 'center center',
              }}
            >
              {/* Clip the entering scene to this cell */}
              {isEntering && (
                <div
                  style={{
                    position: 'absolute',
                    left: `${-col * 100}%`,
                    top: `${-row * 100}%`,
                    width: `${gridSize * 100}%`,
                    height: `${gridSize * 100}%`,
                  }}
                >
                  {children}
                </div>
              )}
            </div>
          );
        })}
      </AbsoluteFill>

      {/* For entering, show full scene underneath with inverse mask */}
      {isEntering && progress < 0.01 && (
        <AbsoluteFill style={{ opacity: 0 }}>
          {children}
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

export const checkerboard = (
  props: CheckerboardProps = {}
): TransitionPresentation<CheckerboardProps> => {
  return { component: CheckerboardPresentation, props };
};
