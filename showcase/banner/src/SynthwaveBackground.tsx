import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import type { Theme } from './themes';

export const SynthwaveBackground: React.FC<{ theme: Theme }> = ({ theme }) => {
  const frame = useCurrentFrame();

  const fadeIn = interpolate(frame, [8, 25], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const gridScroll = (frame * 1.2) % 60;

  return (
    <AbsoluteFill style={{ background: theme.background, overflow: 'hidden' }}>
      {theme.sun.enabled && (
        <>
          {/* Sun horizon glow */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '42%',
              width: 620,
              height: 620,
              transform: 'translate(-50%, -50%)',
              borderRadius: '50%',
              background: `radial-gradient(circle at 50% 50%, ${theme.sun.inner} 0%, ${theme.sun.mid} 28%, ${theme.sun.outer} 60%, ${theme.background} 82%)`,
              filter: 'blur(2px)',
              opacity: fadeIn * 0.85,
            }}
          />
          {/* Sun bands — synthwave stripes cutting the sun */}
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: '42%',
              width: 620,
              height: 620,
              transform: 'translate(-50%, -50%)',
              borderRadius: '50%',
              overflow: 'hidden',
              opacity: fadeIn * 0.9,
              WebkitMaskImage:
                'radial-gradient(circle at 50% 50%, black 0%, black 48%, transparent 52%)',
              maskImage:
                'radial-gradient(circle at 50% 50%, black 0%, black 48%, transparent 52%)',
            }}
          >
            {[0, 1, 2, 3, 4].map((i) => (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  top: `${62 + i * 7}%`,
                  height: `${3 + i * 1.2}%`,
                  background: theme.sun.bandColor,
                }}
              />
            ))}
          </div>
        </>
      )}

      {/* Perspective grid */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: '-10%',
          right: '-10%',
          height: '52%',
          perspective: '340px',
          perspectiveOrigin: '50% 0%',
          opacity: fadeIn,
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            transform: 'rotateX(58deg)',
            transformOrigin: '50% 0%',
            backgroundImage: `
              linear-gradient(to right, ${theme.grid.vertical} ${theme.grid.thickness}px, transparent ${theme.grid.thickness}px),
              linear-gradient(to bottom, ${theme.grid.horizontal} ${theme.grid.thickness}px, transparent ${theme.grid.thickness}px)
            `,
            backgroundSize: '60px 60px',
            backgroundPosition: `0 ${gridScroll}px, 0 ${gridScroll}px`,
          }}
        />
        {/* Horizon haze */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '18%',
            background: `linear-gradient(to bottom, ${theme.background} 0%, ${theme.grid.vertical} 50%, transparent 100%)`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
