/**
 * Light Leak Transition
 *
 * Cinematic light leak/lens flare effect that washes over the scene.
 * Creates warmth, nostalgia, and organic film-like quality.
 *
 * Best for: Emotional moments, celebrations, warm transitions, film aesthetic
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo } from 'react';
import { AbsoluteFill, interpolate, random } from 'remotion';

export type LightLeakProps = {
  /** Color temperature: 'warm' (orange/gold), 'cool' (blue/cyan), 'rainbow'. Default: 'warm' */
  temperature?: 'warm' | 'cool' | 'rainbow';
  /** Direction the light enters from. Default: 'right' */
  direction?: 'left' | 'right' | 'top' | 'bottom' | 'center';
  /** Intensity of the overexposure. Default: 0.8 */
  intensity?: number;
  /** Include lens flare artifacts. Default: true */
  flareArtifacts?: boolean;
};

const LightLeakPresentation: React.FC<
  TransitionPresentationComponentProps<LightLeakProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    temperature = 'warm',
    direction = 'right',
    intensity = 0.8,
    flareArtifacts = true,
  } = passedProps;

  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Light leak sweeps across the scene
  const leakProgress = useMemo(() => {
    return interpolate(progress, [0, 0.6, 1], [0, 1, 0.2], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Scene exposure (brightens during transition)
  const exposure = useMemo(() => {
    return interpolate(progress, [0, 0.4, 0.6, 1], [1, 1.3, 1.3, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Opacity for entering/exiting
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.6], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.4, 1], [0, 1], { extrapolateLeft: 'clamp' });

  // Color gradients based on temperature
  const getGradientColors = () => {
    switch (temperature) {
      case 'warm':
        return {
          primary: 'rgba(255, 180, 80, 0.9)',
          secondary: 'rgba(255, 120, 50, 0.7)',
          tertiary: 'rgba(255, 220, 150, 0.5)',
        };
      case 'cool':
        return {
          primary: 'rgba(100, 180, 255, 0.9)',
          secondary: 'rgba(150, 220, 255, 0.7)',
          tertiary: 'rgba(200, 240, 255, 0.5)',
        };
      case 'rainbow':
        return {
          primary: 'rgba(255, 100, 150, 0.8)',
          secondary: 'rgba(255, 200, 100, 0.6)',
          tertiary: 'rgba(100, 200, 255, 0.5)',
        };
    }
  };

  const colors = getGradientColors();

  // Calculate gradient position based on direction
  const getGradientPosition = () => {
    const pos = leakProgress * 150 - 50; // -50 to 100
    switch (direction) {
      case 'left':
        return `linear-gradient(90deg, ${colors.primary} ${pos}%, ${colors.secondary} ${pos + 20}%, ${colors.tertiary} ${pos + 40}%, transparent ${pos + 60}%)`;
      case 'right':
        return `linear-gradient(270deg, ${colors.primary} ${pos}%, ${colors.secondary} ${pos + 20}%, ${colors.tertiary} ${pos + 40}%, transparent ${pos + 60}%)`;
      case 'top':
        return `linear-gradient(180deg, ${colors.primary} ${pos}%, ${colors.secondary} ${pos + 20}%, ${colors.tertiary} ${pos + 40}%, transparent ${pos + 60}%)`;
      case 'bottom':
        return `linear-gradient(0deg, ${colors.primary} ${pos}%, ${colors.secondary} ${pos + 20}%, ${colors.tertiary} ${pos + 40}%, transparent ${pos + 60}%)`;
      case 'center':
        return `radial-gradient(ellipse at center, ${colors.primary} ${pos * 0.5}%, ${colors.secondary} ${pos * 0.7}%, ${colors.tertiary} ${pos}%, transparent ${pos + 30}%)`;
    }
  };

  // Flare artifact positions (deterministic)
  const flarePositions = useMemo(() => {
    return Array.from({ length: 5 }, (_, i) => ({
      x: random(`flare-x-${i}`) * 100,
      y: random(`flare-y-${i}`) * 100,
      size: 20 + random(`flare-size-${i}`) * 60,
      delay: random(`flare-delay-${i}`) * 0.3,
    }));
  }, []);

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
  }), []);

  return (
    <AbsoluteFill style={containerStyle}>
      {/* Main content with exposure adjustment */}
      <AbsoluteFill
        style={{
          opacity,
          filter: `brightness(${exposure})`,
        }}
      >
        {children}
      </AbsoluteFill>

      {/* Light leak gradient overlay */}
      <AbsoluteFill
        style={{
          background: getGradientPosition(),
          opacity: intensity * leakProgress,
          mixBlendMode: 'screen',
          pointerEvents: 'none',
        }}
      />

      {/* Soft glow overlay */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at ${direction === 'left' ? '20%' : direction === 'right' ? '80%' : '50%'} 50%, ${colors.tertiary}, transparent 70%)`,
          opacity: intensity * leakProgress * 0.5,
          mixBlendMode: 'overlay',
          pointerEvents: 'none',
        }}
      />

      {/* Lens flare artifacts */}
      {flareArtifacts && leakProgress > 0.2 && (
        <AbsoluteFill style={{ pointerEvents: 'none' }}>
          {flarePositions.map((flare, i) => {
            const flareOpacity = interpolate(
              progress,
              [flare.delay, flare.delay + 0.3, 0.7, 1],
              [0, 0.6, 0.6, 0],
              { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
            );
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: `${flare.x}%`,
                  top: `${flare.y}%`,
                  width: flare.size,
                  height: flare.size,
                  borderRadius: '50%',
                  background: `radial-gradient(circle, ${temperature === 'warm' ? 'rgba(255, 255, 200, 0.8)' : temperature === 'cool' ? 'rgba(200, 240, 255, 0.8)' : 'rgba(255, 200, 255, 0.8)'}, transparent)`,
                  opacity: flareOpacity * intensity,
                  transform: 'translate(-50%, -50%)',
                  mixBlendMode: 'screen',
                }}
              />
            );
          })}

          {/* Hexagonal flare (anamorphic style) */}
          <div
            style={{
              position: 'absolute',
              left: direction === 'right' ? '70%' : direction === 'left' ? '30%' : '50%',
              top: '50%',
              width: 200,
              height: 30,
              background: `linear-gradient(90deg, transparent, ${colors.tertiary}, transparent)`,
              opacity: leakProgress * intensity * 0.7,
              transform: 'translate(-50%, -50%) rotate(-5deg)',
              filter: 'blur(10px)',
              mixBlendMode: 'screen',
            }}
          />
        </AbsoluteFill>
      )}

      {/* Film grain for authenticity */}
      <AbsoluteFill
        style={{
          opacity: leakProgress * 0.1,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          mixBlendMode: 'overlay',
          pointerEvents: 'none',
        }}
      />
    </AbsoluteFill>
  );
};

export const lightLeak = (
  props: LightLeakProps = {}
): TransitionPresentation<LightLeakProps> => {
  return { component: LightLeakPresentation, props };
};
