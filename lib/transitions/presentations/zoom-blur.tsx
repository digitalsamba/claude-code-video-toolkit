/**
 * Zoom Blur Transition
 *
 * Radial motion blur combined with scale for high-energy transitions.
 * Creates a sense of speed, impact, and forward momentum.
 *
 * Best for: CTAs, reveals, action sequences, energetic moments
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo } from 'react';
import { AbsoluteFill, interpolate } from 'remotion';

export type ZoomBlurProps = {
  /** Direction: 'in' zooms toward viewer, 'out' zooms away. Default: 'in' */
  direction?: 'in' | 'out';
  /** Maximum blur amount in pixels. Default: 20 */
  blurAmount?: number;
  /** Scale multiplier at peak. Default: 1.15 */
  scaleAmount?: number;
  /** Origin point for zoom. Default: 'center' */
  origin?: 'center' | 'top' | 'bottom' | 'left' | 'right';
};

const ZoomBlurPresentation: React.FC<
  TransitionPresentationComponentProps<ZoomBlurProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    direction = 'in',
    blurAmount = 20,
    scaleAmount = 1.15,
    origin = 'center',
  } = passedProps;

  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Effect intensity peaks in the middle then settles
  const effectIntensity = useMemo(() => {
    return interpolate(progress, [0, 0.4, 1], [0, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Scale animation
  const scale = useMemo(() => {
    if (direction === 'in') {
      // Start small, zoom in
      return interpolate(
        progress,
        [0, 0.5, 1],
        [1 / scaleAmount, scaleAmount, 1],
        { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
      );
    } else {
      // Start big, zoom out
      return interpolate(
        progress,
        [0, 0.5, 1],
        [scaleAmount, 1 / scaleAmount, 1],
        { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
      );
    }
  }, [progress, direction, scaleAmount]);

  // Blur tracks with scale movement
  const blur = blurAmount * effectIntensity;

  // Opacity
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.5], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.5, 1], [0, 1], { extrapolateLeft: 'clamp' });

  // Transform origin based on setting
  const transformOrigin = useMemo(() => {
    switch (origin) {
      case 'top': return 'center top';
      case 'bottom': return 'center bottom';
      case 'left': return 'left center';
      case 'right': return 'right center';
      default: return 'center center';
    }
  }, [origin]);

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  }), []);

  const contentStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
    transform: `scale(${scale})`,
    transformOrigin,
    filter: blur > 0.5 ? `blur(${blur}px)` : undefined,
    opacity,
  }), [scale, transformOrigin, blur, opacity]);

  return (
    <AbsoluteFill style={containerStyle}>
      <div style={contentStyle}>
        {children}
      </div>

      {/* Radial light streak overlay for extra energy */}
      {effectIntensity > 0.3 && (
        <AbsoluteFill
          style={{
            opacity: effectIntensity * 0.4,
            background: `radial-gradient(
              ellipse at ${origin === 'center' ? '50% 50%' : origin === 'top' ? '50% 0%' : origin === 'bottom' ? '50% 100%' : origin === 'left' ? '0% 50%' : '100% 50%'},
              rgba(255, 255, 255, 0.3) 0%,
              rgba(255, 255, 255, 0.1) 30%,
              transparent 70%
            )`,
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
          }}
        />
      )}
    </AbsoluteFill>
  );
};

export const zoomBlur = (
  props: ZoomBlurProps = {}
): TransitionPresentation<ZoomBlurProps> => {
  return { component: ZoomBlurPresentation, props };
};
