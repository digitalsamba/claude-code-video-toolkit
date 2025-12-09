import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from 'remotion';
import { useTheme } from '../../config/theme';
import type { StatsContent } from '../../config/types';

interface StatsSlideProps {
  content: StatsContent;
}

export const StatsSlide: React.FC<StatsSlideProps> = ({ content }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = useTheme();

  // Header animation
  const headerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const headerY = interpolate(frame, [0, 25], [20, 0], {
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 60,
        }}
      >
        {/* Header */}
        {content.headline && (
          <h2
            style={{
              fontSize: theme.typography.h2.size - 8,
              fontWeight: theme.typography.h2.weight,
              fontFamily: theme.fonts.primary,
              color: theme.colors.textDark,
              margin: 0,
              opacity: headerOpacity,
              transform: `translateY(${headerY}px)`,
            }}
          >
            {content.headline}
          </h2>
        )}

        {/* Stats Cards */}
        <div style={{ display: 'flex', gap: 40 }}>
          {content.stats.map((stat, i) => {
            const delay = 30 + i * 20;
            const cardScale = spring({
              frame: Math.max(0, frame - delay),
              fps,
              config: { damping: 10, stiffness: 100, mass: 0.8 },
            });

            const cardOpacity = interpolate(
              frame,
              [delay, delay + 15],
              [0, 1],
              { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
            );

            const color = stat.color || theme.colors.primary;

            return (
              <div
                key={i}
                style={{
                  opacity: cardOpacity,
                  transform: `scale(${cardScale})`,
                  background: `linear-gradient(135deg, ${color}15, ${color}05)`,
                  border: `1px solid ${color}40`,
                  borderRadius: 24,
                  padding: '50px 60px',
                  textAlign: 'center',
                  minWidth: 280,
                }}
              >
                {/* Icon */}
                {stat.icon && (
                  <div style={{ fontSize: 48, marginBottom: 20 }}>
                    {stat.icon}
                  </div>
                )}

                {/* Value */}
                <div
                  style={{
                    fontSize: 72,
                    fontWeight: 700,
                    fontFamily: theme.fonts.primary,
                    color: color,
                    lineHeight: 1,
                  }}
                >
                  {stat.value}
                  {stat.unit && (
                    <span
                      style={{
                        fontSize: 32,
                        fontWeight: 400,
                        marginLeft: 4,
                      }}
                    >
                      {stat.unit}
                    </span>
                  )}
                </div>

                {/* Label */}
                <div
                  style={{
                    fontSize: 20,
                    fontFamily: theme.fonts.primary,
                    color: theme.colors.textMedium,
                    marginTop: 16,
                    fontWeight: 500,
                  }}
                >
                  {stat.label}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
