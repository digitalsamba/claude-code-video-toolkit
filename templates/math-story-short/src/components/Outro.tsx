import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { brand } from '../config/brand';
import { theme } from '../config/theme';

/** End card under Max (MaxLayer walks him to centre stage): wordmark, tagline, follow pill. */
export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = spring({ frame: frame - 8, fps, config: { damping: 16 } });
  const b = spring({ frame: frame - 16, fps, config: { damping: 16 } });
  const c = spring({ frame: frame - 26, fps, config: { damping: 12, stiffness: 160 } });
  const [first, second] = ['MaxLearns', 'Math'];

  return (
    <div style={{ position: 'absolute', left: 0, right: 0, top: 1270, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22 }}>
      <div style={{ fontFamily: theme.fonts.sans, fontWeight: 900, fontSize: 104, letterSpacing: -3, opacity: a, transform: `translateY(${interpolate(a, [0, 1], [40, 0])}px)` }}>
        <span style={{ color: theme.text }}>{first}</span>
        <span style={{ color: theme.primary }}>{second}</span>
      </div>
      <div style={{ fontFamily: theme.fonts.mono, fontSize: 34, color: theme.muted, opacity: b }}>{brand.tagline}</div>
      <div
        style={{
          marginTop: 20,
          padding: '18px 36px',
          borderRadius: 999,
          background: theme.cream,
          color: theme.bg,
          fontFamily: theme.fonts.sans,
          fontWeight: 800,
          fontSize: 40,
          opacity: c,
          transform: `scale(${interpolate(c, [0, 1], [0.7, 1])})`,
          boxShadow: '0 20px 50px rgba(0,10,20,0.45)',
        }}
      >
        Follow {brand.handle} for the next story
      </div>
    </div>
  );
};
