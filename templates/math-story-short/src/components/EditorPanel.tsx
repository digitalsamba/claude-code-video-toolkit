import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';

export interface EditorPanelProps {
  /** Filename in the tab, e.g. "bankroll.plot" */
  file: string;
  /** Right side of the title bar: a live readout or a tag */
  status?: React.ReactNode;
  /** Pulse a "live" dot next to the status */
  live?: boolean;
  width?: number;
  height?: number;
  /** Frames before the panel springs in */
  delay?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

/**
 * The code-editor window the charts live in (the brand's signature frame).
 * Glass body, tab with a file glyph, traffic-light dots in brand colours.
 */
export const EditorPanel: React.FC<EditorPanelProps> = ({
  file,
  status,
  live,
  width = 960,
  height = 620,
  delay = 0,
  style,
  children,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 20, stiffness: 120 } });
  const pulse = 0.5 + 0.5 * Math.sin(frame / 6);

  return (
    <div
      style={{
        width,
        height,
        borderRadius: 30,
        background: 'linear-gradient(180deg, rgba(11, 60, 92, 0.82) 0%, rgba(0, 36, 58, 0.88) 100%)',
        border: `1.5px solid ${theme.faint}`,
        boxShadow:
          '0 40px 90px rgba(0, 10, 20, 0.55), 0 0 0 1px rgba(0,0,0,0.15), inset 0 1px 0 rgba(253, 240, 213, 0.09)',
        overflow: 'hidden',
        opacity: s,
        transform: `translateY(${interpolate(s, [0, 1], [60, 0])}px) scale(${interpolate(s, [0, 1], [0.94, 1])})`,
        display: 'flex',
        flexDirection: 'column',
        ...style,
      }}
    >
      <div
        style={{
          height: 74,
          display: 'flex',
          alignItems: 'center',
          padding: '0 26px',
          gap: 12,
          background: 'rgba(0, 24, 40, 0.45)',
          borderBottom: `1px solid ${theme.hairline}`,
          flexShrink: 0,
        }}
      >
        {[theme.primary, theme.cream, theme.accent].map((c, i) => (
          <div key={i} style={{ width: 16, height: 16, borderRadius: 8, background: c, opacity: i === 0 ? 0.95 : 0.6 }} />
        ))}
        <div
          style={{
            marginLeft: 18,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '10px 20px',
            borderRadius: 12,
            background: 'rgba(102, 155, 188, 0.12)',
            border: `1px solid ${theme.hairline}`,
            fontFamily: theme.fonts.mono,
            fontSize: 32,
            color: theme.text,
          }}
        >
          <svg width="22" height="22" viewBox="0 0 24 24">
            <path d="M3 20 L8 12 L12 16 L20 5" fill="none" stroke={theme.primary} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {file}
        </div>
        <div style={{ flex: 1 }} />
        {status !== undefined && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: theme.fonts.mono, fontSize: 30, color: theme.muted }}>
            {live && (
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 6,
                  background: theme.primary,
                  boxShadow: `0 0 ${6 + pulse * 10}px ${theme.primary}`,
                  opacity: 0.6 + pulse * 0.4,
                }}
              />
            )}
            {status}
          </div>
        )}
      </div>
      <div style={{ position: 'relative', flex: 1 }}>{children}</div>
    </div>
  );
};
