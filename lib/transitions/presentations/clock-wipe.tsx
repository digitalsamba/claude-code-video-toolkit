/**
 * Clock Wipe Transition
 *
 * A radial wipe that reveals the scene like clock hands sweeping.
 * Classic transition with a playful, dynamic quality.
 *
 * Best for: Time-related content, reveals, playful videos
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo, useId } from 'react';
import { AbsoluteFill, interpolate } from 'remotion';

export type ClockWipeProps = {
  /** Starting angle in degrees. Default: 0 (12 o'clock) */
  startAngle?: number;
  /** Direction: 'clockwise' or 'counterclockwise'. Default: 'clockwise' */
  direction?: 'clockwise' | 'counterclockwise';
  /** Number of wipe segments (1 = single wipe, 2+ = multiple arms). Default: 1 */
  segments?: number;
};

const ClockWipePresentation: React.FC<
  TransitionPresentationComponentProps<ClockWipeProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    startAngle = 0,
    direction = 'clockwise',
    segments = 1,
  } = passedProps;

  // Use React's useId for truly unique clip path IDs
  const clipId = useId().replace(/:/g, '-');

  // The swept angle grows from 0 to 360/segments as progress goes 0 to 1
  const sweptAngle = useMemo(() => {
    const totalSweep = 360 / segments;
    return interpolate(presentationProgress, [0, 1], [0, totalSweep], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [presentationProgress, segments]);

  // Determine if this scene should be visible at all
  // Entering: hidden at start, visible at end
  // Exiting: visible at start, hidden at end
  const isEffectivelyHidden = useMemo(() => {
    if (presentationDirection === 'entering') {
      return sweptAngle <= 0.1; // No area swept yet = hide entering
    } else {
      return sweptAngle >= 359.9; // Fully swept = hide exiting
    }
  }, [presentationDirection, sweptAngle]);

  const isFullyVisible = useMemo(() => {
    if (presentationDirection === 'entering') {
      return sweptAngle >= 359.9; // Fully swept = show all of entering
    } else {
      return sweptAngle <= 0.1; // Nothing swept = show all of exiting
    }
  }, [presentationDirection, sweptAngle]);

  // Generate the SVG clip path
  const clipPath = useMemo(() => {
    const cx = 50;
    const cy = 50;
    const r = 75; // Large enough to cover corners of 100x100 square

    // Edge cases handled by opacity, but still need valid paths
    if (isEffectivelyHidden) {
      return 'M 0 0 Z'; // Point (will be hidden by opacity anyway)
    }

    if (isFullyVisible) {
      return 'M 0 0 L 100 0 L 100 100 L 0 100 Z'; // Full rectangle
    }

    const paths: string[] = [];

    if (presentationDirection === 'entering') {
      // ENTERING (Scene B): Show the SWEPT area (the pie that clock hand has passed)
      // This grows from nothing to full circle
      for (let i = 0; i < segments; i++) {
        const segmentOffset = i * 360 / segments;
        const pieStart = startAngle + segmentOffset;
        const pieEnd = pieStart + (direction === 'clockwise' ? sweptAngle : -sweptAngle);

        // Convert to radians (subtract 90 to make 0° = 12 o'clock)
        const startRad = (pieStart - 90) * Math.PI / 180;
        const endRad = (pieEnd - 90) * Math.PI / 180;

        const x1 = cx + r * Math.cos(startRad);
        const y1 = cy + r * Math.sin(startRad);
        const x2 = cx + r * Math.cos(endRad);
        const y2 = cy + r * Math.sin(endRad);

        const largeArc = sweptAngle > 180 ? 1 : 0;
        const sweepFlag = direction === 'clockwise' ? 1 : 0;

        paths.push(`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweepFlag} ${x2} ${y2} Z`);
      }
    } else {
      // EXITING (Scene A): Show the REMAINING area (what clock hand hasn't passed)
      // This shrinks from full circle to nothing
      const remainingAngle = (360 / segments) - sweptAngle;

      for (let i = 0; i < segments; i++) {
        const segmentOffset = i * 360 / segments;
        // Remaining starts where swept ends
        const pieStart = startAngle + segmentOffset + (direction === 'clockwise' ? sweptAngle : -sweptAngle);
        const pieEnd = startAngle + segmentOffset + (direction === 'clockwise' ? 360 / segments : -360 / segments);

        const startRad = (pieStart - 90) * Math.PI / 180;
        const endRad = (pieEnd - 90) * Math.PI / 180;

        const x1 = cx + r * Math.cos(startRad);
        const y1 = cy + r * Math.sin(startRad);
        const x2 = cx + r * Math.cos(endRad);
        const y2 = cy + r * Math.sin(endRad);

        const largeArc = remainingAngle > 180 ? 1 : 0;
        const sweepFlag = direction === 'clockwise' ? 1 : 0;

        paths.push(`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweepFlag} ${x2} ${y2} Z`);
      }
    }

    return paths.join(' ');
  }, [presentationDirection, sweptAngle, startAngle, direction, segments, isEffectivelyHidden, isFullyVisible]);

  // Use opacity as belt-and-suspenders for edge cases
  // This ensures hidden scenes are truly hidden even if clip path fails
  const opacity = isEffectivelyHidden ? 0 : 1;

  return (
    <AbsoluteFill>
      <AbsoluteFill
        style={{
          clipPath: isFullyVisible ? undefined : `url(#${clipId})`,
          WebkitClipPath: isFullyVisible ? undefined : `url(#${clipId})`,
          opacity,
        }}
      >
        {children}
      </AbsoluteFill>

      {/* SVG clip path definition - only needed when not fully visible */}
      {!isFullyVisible && !isEffectivelyHidden && (
        <svg style={{ position: 'absolute', width: 0, height: 0 }}>
          <defs>
            <clipPath id={clipId} clipPathUnits="objectBoundingBox">
              <path
                d={clipPath}
                transform="scale(0.01)"
              />
            </clipPath>
          </defs>
        </svg>
      )}
    </AbsoluteFill>
  );
};

export const clockWipe = (
  props: ClockWipeProps = {}
): TransitionPresentation<ClockWipeProps> => {
  return { component: ClockWipePresentation, props };
};
