import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';

/** Stacked fraction in the math serif. */
export const Frac: React.FC<{ n: React.ReactNode; d: React.ReactNode }> = ({ n, d }) => (
  <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', verticalAlign: 'middle', margin: '0 10px' }}>
    <span style={{ padding: '0 10px 4px' }}>{n}</span>
    <span style={{ alignSelf: 'stretch', height: 3, background: 'currentColor', borderRadius: 2 }} />
    <span style={{ padding: '4px 10px 0' }}>{d}</span>
  </span>
);

/** Crimson result with a marker sweep underneath, drawn in from frame `at`. */
export const Hl: React.FC<{ at: number; children: React.ReactNode }> = ({ at, children }) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame, [at, at + 14], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <span style={{ position: 'relative', color: p > 0.2 ? '#ff5a63' : theme.text, whiteSpace: 'nowrap' }}>
      <span
        style={{
          position: 'absolute',
          left: -6,
          right: -6,
          bottom: -6,
          height: 7,
          borderRadius: 4,
          background: theme.primary,
          transform: `scaleX(${p})`,
          transformOrigin: 'left',
        }}
      />
      <span style={{ position: 'relative' }}>{children}</span>
    </span>
  );
};

export interface EquationLine {
  /** Frame (relative to the card's Sequence) the line appears on */
  at: number;
  content: React.ReactNode;
  /** Small mono caption under the line */
  note?: string;
  size?: number;
}

/**
 * Glass card that builds a derivation line by line. Each line rises in with
 * a de-blur; earlier lines dim slightly so the eye lands on the newest step.
 */
export const EquationCard: React.FC<{ lines: EquationLine[]; width?: number; delay?: number }> = ({ lines, width = 960, delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = spring({ frame: frame - delay, fps, config: { damping: 20, stiffness: 120 } });
  const shown = lines.filter((l) => frame >= l.at);
  return (
    <div
      style={{
        width,
        padding: '30px 40px 30px 52px',
        borderRadius: 26,
        position: 'relative',
        background: 'linear-gradient(180deg, rgba(11,60,92,0.7), rgba(0,36,58,0.75))',
        border: `1.5px solid ${theme.hairline}`,
        boxShadow: '0 30px 70px rgba(0,10,20,0.45)',
        opacity: enter,
        transform: `translateY(${interpolate(enter, [0, 1], [40, 0])}px)`,
        overflow: 'hidden',
      }}
    >
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 8, background: `linear-gradient(${theme.primary}, ${theme.primaryDark})` }} />
      {shown.map((l, i) => {
        const s = spring({ frame: frame - l.at, fps, config: { damping: 18, stiffness: 130 } });
        const isLast = i === shown.length - 1;
        return (
          <div
            key={i}
            style={{
              fontFamily: theme.fonts.math,
              fontSize: l.size ?? 54,
              color: theme.text,
              lineHeight: 1.25,
              margin: '10px 0',
              opacity: s * (isLast ? 1 : 0.62),
              filter: `blur(${interpolate(s, [0, 1], [8, 0])}px)`,
              transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)`,
              display: 'flex',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 4,
            }}
          >
            {l.content}
            {l.note && (
              <span style={{ width: '100%', fontFamily: theme.fonts.mono, fontSize: 26, color: theme.muted, marginTop: 6 }}>{l.note}</span>
            )}
          </div>
        );
      })}
    </div>
  );
};
