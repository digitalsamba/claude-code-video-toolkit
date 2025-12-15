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
import React, { useMemo, useState } from 'react';
import { AbsoluteFill, interpolate, random } from 'remotion';

export type ClockWipeProps = {
  /** Starting angle in degrees. Default: 0 (12 o'clock) */
  startAngle?: number;
  /** Direction: 'clockwise' or 'counterclockwise'. Default: 'clockwise' */
  direction?: 'clockwise' | 'counterclockwise';
  /** Number of wipe segments (1 = single wipe, 2+ = multiple arms). Default: 1 */
  segments?: number;
  /** Include soft edge blur. Default: true */
  softEdge?: boolean;
};

const ClockWipePresentation: React.FC<
  TransitionPresentationComponentProps<ClockWipeProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    startAngle = 0,
    direction = 'clockwise',
    segments = 1,
    softEdge = true,
  } = passedProps;

  const [clipId] = useState(() => `clock-wipe-${String(random(null)).slice(2, 10)}`);

  const progress = presentationDirection === 'exiting'
    ? presentationProgress
    : presentationProgress;

  // Calculate the sweep angle
  const sweepAngle = useMemo(() => {
    const totalSweep = 360 / segments;
    return interpolate(progress, [0, 1], [0, totalSweep], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress, segments]);

  // For exiting, we use the inverse clip
  const isRevealing = presentationDirection === 'entering';

  // Generate the SVG path for the pie slice(s)
  const generateClipPath = () => {
    const paths: string[] = [];
    const cx = 50; // Center X (percentage)
    const cy = 50; // Center Y (percentage)
    const r = 75;  // Radius (large enough to cover corners)

    for (let i = 0; i < segments; i++) {
      const segmentStartAngle = startAngle + (i * 360 / segments);
      const adjustedSweep = direction === 'clockwise' ? sweepAngle : -sweepAngle;
      const endAngle = segmentStartAngle + adjustedSweep;

      // Convert angles to radians (SVG uses different coordinate system)
      const startRad = (segmentStartAngle - 90) * Math.PI / 180;
      const endRad = (endAngle - 90) * Math.PI / 180;

      // Calculate arc endpoints
      const x1 = cx + r * Math.cos(startRad);
      const y1 = cy + r * Math.sin(startRad);
      const x2 = cx + r * Math.cos(endRad);
      const y2 = cy + r * Math.sin(endRad);

      // Determine if we need the large arc flag
      const largeArc = Math.abs(sweepAngle) > 180 ? 1 : 0;
      const sweepFlag = direction === 'clockwise' ? 1 : 0;

      // Create pie slice path
      const path = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} ${sweepFlag} ${x2} ${y2} Z`;
      paths.push(path);
    }

    return paths.join(' ');
  };

  // Opacity for smooth transition
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0.8, 1], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
    : interpolate(progress, [0, 0.2], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
  }), []);

  return (
    <AbsoluteFill style={containerStyle}>
      {/* Clipped content */}
      <AbsoluteFill
        style={{
          clipPath: isRevealing ? `url(#${clipId})` : undefined,
          WebkitClipPath: isRevealing ? `url(#${clipId})` : undefined,
          opacity: isRevealing ? opacity : 1,
        }}
      >
        {children}
      </AbsoluteFill>

      {/* For exiting, show content disappearing */}
      {!isRevealing && (
        <AbsoluteFill
          style={{
            clipPath: `url(#${clipId}-inverse)`,
            WebkitClipPath: `url(#${clipId}-inverse)`,
            opacity,
          }}
        >
          {children}
        </AbsoluteFill>
      )}

      {/* Soft edge glow effect */}
      {softEdge && sweepAngle > 5 && sweepAngle < 355 && (
        <AbsoluteFill
          style={{
            opacity: 0.3,
            pointerEvents: 'none',
          }}
        >
          <svg width="100%" height="100%" style={{ position: 'absolute' }}>
            <defs>
              <radialGradient id={`${clipId}-glow`}>
                <stop offset="0%" stopColor="white" stopOpacity="0" />
                <stop offset="90%" stopColor="white" stopOpacity="0.5" />
                <stop offset="100%" stopColor="white" stopOpacity="0" />
              </radialGradient>
            </defs>
            {/* Edge glow line */}
            {(() => {
              const edgeAngle = startAngle + (direction === 'clockwise' ? sweepAngle : -sweepAngle);
              const edgeRad = (edgeAngle - 90) * Math.PI / 180;
              const cx = 50;
              const cy = 50;
              const r = 75;
              const x = cx + r * Math.cos(edgeRad);
              const y = cy + r * Math.sin(edgeRad);
              return (
                <line
                  x1={`${cx}%`}
                  y1={`${cy}%`}
                  x2={`${x}%`}
                  y2={`${y}%`}
                  stroke="rgba(255, 255, 255, 0.5)"
                  strokeWidth="4"
                  filter="blur(3px)"
                />
              );
            })()}
          </svg>
        </AbsoluteFill>
      )}

      {/* SVG clip path definitions */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <clipPath id={clipId} clipPathUnits="objectBoundingBox">
            <path
              d={generateClipPath()}
              transform="scale(0.01)"
            />
          </clipPath>
          {/* Inverse clip for exiting */}
          <clipPath id={`${clipId}-inverse`} clipPathUnits="objectBoundingBox">
            <path
              d={`M 0 0 L 100 0 L 100 100 L 0 100 Z ${generateClipPath()}`}
              transform="scale(0.01)"
              clipRule="evenodd"
            />
          </clipPath>
        </defs>
      </svg>
    </AbsoluteFill>
  );
};

export const clockWipe = (
  props: ClockWipeProps = {}
): TransitionPresentation<ClockWipeProps> => {
  return { component: ClockWipePresentation, props };
};
