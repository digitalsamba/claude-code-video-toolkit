import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import type { Theme } from './themes';

export const CRTOverlay: React.FC<{ theme: Theme }> = ({ theme }) => {
  const frame = useCurrentFrame();

  const sweepY = interpolate(frame, [0, 9], [-20, 120], {
    extrapolateRight: 'clamp',
  });
  const sweepOpacity = interpolate(frame, [0, 4, 9], [0, 1, 0], {
    extrapolateRight: 'clamp',
  });

  const powerOn = interpolate(frame, [0, 9], [0.1, 1], {
    extrapolateRight: 'clamp',
    extrapolateLeft: 'clamp',
  });

  const flicker = frame % 90 === 0 || frame % 137 === 0 ? 0.85 : 1;

  const scanAlpha = Math.min(0.08 * theme.crt.scanlineOpacity, 0.12);

  return (
    <>
      <AbsoluteFill style={{ background: '#000', opacity: 1 - powerOn, pointerEvents: 'none' }} />

      <AbsoluteFill
        style={{
          backgroundImage: `repeating-linear-gradient(to bottom, rgba(255,255,255,${scanAlpha}) 0px, rgba(255,255,255,${scanAlpha}) 1px, transparent 1px, transparent 3px)`,
          mixBlendMode: 'overlay',
          opacity: flicker,
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: `${sweepY}%`,
          height: 4,
          background: `linear-gradient(to bottom, transparent, ${theme.crt.sweepColor}, transparent)`,
          filter: 'blur(2px)',
          opacity: sweepOpacity,
          pointerEvents: 'none',
        }}
      />

      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,${theme.crt.vignetteStrength}) 100%)`,
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 60,
          background: 'linear-gradient(to bottom, rgba(255,255,255,0.05), transparent)',
          pointerEvents: 'none',
        }}
      />
    </>
  );
};
