import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';

export interface Stat {
  label: string;
  value: string;
  /** 'bad' tints crimson, 'good' cream-bright, default cream */
  tone?: 'bad' | 'good' | 'neutral';
}

/**
 * Glass tiles with a mono label over a big tabular number. `column` stacks
 * them for the aside zone beside Max (see Aside).
 */
export const StatRow: React.FC<{ stats: Stat[]; delay?: number; width?: number; column?: boolean; pulse?: number }> = ({
  stats,
  delay = 0,
  width = 960,
  column,
  pulse = 0,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div style={{ display: 'flex', flexDirection: column ? 'column' : 'row', gap: column ? 16 : 20, width }}>
      {stats.map((s, i) => {
        const e = spring({ frame: frame - delay - i * 4, fps, config: { damping: 18, stiffness: 140 } });
        const bad = s.tone === 'bad';
        return (
          <div
            key={s.label}
            style={{
              flex: 1,
              padding: '22px 28px',
              borderRadius: 22,
              background: bad ? 'linear-gradient(135deg, rgba(193,18,31,0.22), rgba(120,0,0,0.18))' : 'rgba(11, 60, 92, 0.55)',
              border: `1.5px solid ${bad ? 'rgba(193,18,31,0.55)' : theme.hairline}`,
              opacity: e,
              display: column ? 'flex' : 'block',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              transform: `translateY(${interpolate(e, [0, 1], [30, 0])}px) scale(${bad ? 1 + pulse * 0.06 : 1})`,
              boxShadow: bad && pulse > 0 ? `0 0 ${40 * pulse}px rgba(193,18,31,0.6)` : 'none',
            }}
          >
            <div style={{ fontFamily: theme.fonts.mono, fontSize: 30, color: theme.muted, letterSpacing: 0.5, whiteSpace: 'nowrap', marginRight: 16 }}>{s.label}</div>
            <div
              style={{
                fontFamily: theme.fonts.sans,
                fontWeight: 800,
                fontSize: column ? (s.value.length > 9 ? 56 : 68) : 64,
                letterSpacing: -1.5,
                fontVariantNumeric: 'tabular-nums',
                color: bad ? '#ff5a63' : theme.text,
                marginTop: 4,
                textShadow: bad ? '0 0 24px rgba(193,18,31,0.5)' : 'none',
              }}
            >
              {s.value}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export const fmtMoney = (v: number) => `${v < 0 ? '−' : v > 0 ? '+' : ''}$${Math.abs(Math.round(v)).toLocaleString('en-US')}`;
export const fmtInt = (v: number) => Math.round(v).toLocaleString('en-US');
