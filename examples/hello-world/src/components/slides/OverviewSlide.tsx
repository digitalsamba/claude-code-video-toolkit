import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { useTheme } from '../../config/theme';
import { sprintConfig } from '../../config/sprint-config';

export const OverviewSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const theme = useTheme();
  const { overview } = sprintConfig;

  // Fade in header and title
  const opacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const styles = {
    container: {
      backgroundColor: 'transparent',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 80,
      fontFamily: theme.fonts.primary,
    },
    header: {
      color: theme.colors.primary,
      fontSize: 34,
      fontWeight: 500,
      marginBottom: 20,
      textTransform: 'uppercase' as const,
      letterSpacing: 3,
    },
    title: {
      color: theme.colors.textDark,
      fontSize: 78,
      fontWeight: 700,
      margin: '0 0 60px 0',
      textAlign: 'center' as const,
    },
    list: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: 34,
    },
    item: {
      display: 'flex',
      alignItems: 'center',
      gap: 24,
    },
    bullet: {
      width: 14,
      height: 14,
      borderRadius: 7,
      backgroundColor: theme.colors.primary,
      flexShrink: 0,
    },
    text: {
      color: theme.colors.textMedium,
      fontSize: 44,
    },
    highlight: {
      color: theme.colors.primary,
      fontWeight: 600,
    },
  };

  return (
    <AbsoluteFill style={{ ...styles.container, opacity }}>
      <p style={styles.header}>Sprint Overview</p>
      <h1 style={styles.title}>{overview.title}</h1>

      <div style={styles.list}>
        {overview.items.map((item, index) => {
          // Stagger animation - spread across the sequence
          const itemStart = 40 + index * 45; // Start at ~1.3s, 1.5s apart
          const itemEnd = itemStart + 20; // ~0.7 second animation each
          const itemOpacity = interpolate(
            frame,
            [itemStart, itemEnd],
            [0, 1],
            { extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
          );
          const itemX = interpolate(
            frame,
            [itemStart, itemEnd],
            [-30, 0],
            { extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
          );

          return (
            <div
              key={index}
              style={{
                ...styles.item,
                opacity: itemOpacity,
                transform: `translateX(${itemX}px)`,
              }}
            >
              <div style={styles.bullet} />
              <span style={styles.text}>
                {item.text}
                <span style={styles.highlight}>{item.highlight}</span>
              </span>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
