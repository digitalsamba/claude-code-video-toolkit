import { AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig, getStaticFiles } from 'remotion';
import { useTheme } from '../../config/theme';
import { sprintConfig } from '../../config/sprint-config';

export const SummarySlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = useTheme();
  const { summary } = sprintConfig;

  const staticFiles = getStaticFiles();
  const screenshotPath = summary.screenshotFile ? `images/${summary.screenshotFile}` : null;
  const hasScreenshot = screenshotPath && staticFiles.some((f) => f.name === screenshotPath);

  const styles = {
    container: {
      backgroundColor: 'transparent',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: theme.fonts.primary,
    },
    header: {
      color: theme.colors.primary,
      fontSize: 34,
      fontWeight: 500,
      marginBottom: 24,
      textTransform: 'uppercase' as const,
      letterSpacing: 3,
    },
    title: {
      color: theme.colors.textDark,
      fontSize: 72,
      fontWeight: 700,
      margin: '0 0 80px 0',
    },
    statsContainer: {
      display: 'flex',
      gap: 120,
    },
    stat: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
    },
    statNumber: {
      color: theme.colors.primary,
      fontSize: 140,
      fontWeight: 700,
      lineHeight: 1,
    },
    statLabel: {
      color: theme.colors.textLight,
      fontSize: 32,
      marginTop: 16,
    },
    screenshotOverlay: {
      position: 'absolute' as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: theme.colors.bgOverlay,
    },
    screenshot: {
      height: 350,
      borderRadius: theme.borderRadius.lg,
      boxShadow: `0 12px 60px rgba(0, 0, 0, 0.2)`,
      border: `2px solid ${theme.colors.divider}`,
    },
  };

  // Phase 1: Stats (frames 0-150, ~5 seconds)
  const statsOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const statsScale = interpolate(frame, [0, 25], [0.9, 1], {
    extrapolateRight: 'clamp',
  });

  // Phase 2: Screenshot overlay (starts at frame 150)
  const screenshotStart = 150;
  const screenshotProgress = spring({
    frame: frame - screenshotStart,
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const showScreenshot = frame >= screenshotStart && hasScreenshot;
  const screenshotScale = interpolate(screenshotProgress, [0, 1], [0.8, 1]);
  const screenshotOpacity = interpolate(screenshotProgress, [0, 0.3], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Stats fade out as screenshot comes in
  const statsFadeOut = showScreenshot
    ? interpolate(frame, [screenshotStart, screenshotStart + 20], [1, 0], {
        extrapolateRight: 'clamp',
      })
    : 1;

  return (
    <AbsoluteFill style={styles.container}>
      {/* Stats layer */}
      <div
        style={{
          opacity: statsOpacity * statsFadeOut,
          transform: `scale(${statsScale})`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <p style={styles.header}>Sprint Complete</p>
        <h1 style={styles.title}>Release Summary</h1>

        <div style={styles.statsContainer}>
          {summary.stats.map((stat, index) => {
            // Count up animation with stagger
            const countStart = 40 + index * 10;
            const countEnd = countStart + 60;
            const currentValue = Math.round(
              interpolate(frame, [countStart, countEnd], [0, stat.value], {
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              })
            );

            // Individual stat spring in
            const statSpring = spring({
              frame: frame - (20 + index * 10),
              fps,
              config: { damping: 12, stiffness: 100 },
            });

            return (
              <div
                key={index}
                style={{
                  ...styles.stat,
                  opacity: statSpring,
                  transform: `scale(${0.5 + statSpring * 0.5})`,
                }}
              >
                <span style={styles.statNumber}>{currentValue}</span>
                <span style={styles.statLabel}>{stat.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Screenshot overlay with spring animation */}
      {showScreenshot && screenshotPath && (
        <div style={{ ...styles.screenshotOverlay, opacity: screenshotOpacity }}>
          <Img
            src={staticFile(screenshotPath)}
            style={{
              ...styles.screenshot,
              transform: `scale(${screenshotScale})`,
            }}
          />
        </div>
      )}
    </AbsoluteFill>
  );
};
