import { AbsoluteFill, useCurrentFrame } from 'remotion';
import { theme } from '../config/theme';

/**
 * Living navy backdrop: two slow aurora glows (crimson, steel blue), a drifting
 * dot grid that fades toward the edges, and a soft vignette. Persistent across
 * scenes so cuts never flash the background.
 */
export const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const t = frame / 30;
  const ax = 760 + Math.sin(t * 0.21) * 120;
  const ay = 380 + Math.cos(t * 0.17) * 90;
  const bx = 260 + Math.cos(t * 0.13) * 140;
  const by = 1380 + Math.sin(t * 0.19) * 120;

  return (
    <AbsoluteFill style={{ background: theme.bg, overflow: 'hidden' }}>
      <AbsoluteFill
        style={{
          background: `linear-gradient(180deg, ${theme.bgDeep} 0%, ${theme.bg} 38%, ${theme.bg} 70%, ${theme.bgDeep} 100%)`,
        }}
      />
      <AbsoluteFill
        style={{
          background: [
            `radial-gradient(620px 620px at ${ax}px ${ay}px, rgba(193, 18, 31, 0.16), transparent 70%)`,
            `radial-gradient(760px 760px at ${bx}px ${by}px, rgba(102, 155, 188, 0.16), transparent 70%)`,
          ].join(','),
        }}
      />
      <AbsoluteFill
        style={{
          backgroundImage: 'radial-gradient(rgba(253, 240, 213, 0.16) 1.6px, transparent 1.8px)',
          backgroundSize: '44px 44px',
          backgroundPosition: `${(t * 3) % 44}px ${(t * 6) % 44}px`,
          maskImage: 'radial-gradient(ellipse 75% 60% at 50% 45%, black 20%, transparent 85%)',
          WebkitMaskImage: 'radial-gradient(ellipse 75% 60% at 50% 45%, black 20%, transparent 85%)',
          opacity: 0.55,
        }}
      />
      <AbsoluteFill
        style={{ background: 'radial-gradient(ellipse 90% 70% at 50% 45%, transparent 55%, rgba(0, 12, 22, 0.55) 100%)' }}
      />
    </AbsoluteFill>
  );
};
