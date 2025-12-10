import { AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  header: {
    color: "#ea580c",
    fontSize: 34,
    fontWeight: 500,
    marginBottom: 24,
    textTransform: "uppercase" as const,
    letterSpacing: 3,
  },
  title: {
    color: "#1e293b",
    fontSize: 72,
    fontWeight: 700,
    margin: "0 0 80px 0",
  },
  statsContainer: {
    display: "flex",
    gap: 120,
  },
  stat: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
  },
  statNumber: {
    color: "#ea580c",
    fontSize: 140,
    fontWeight: 700,
    lineHeight: 1,
  },
  statLabel: {
    color: "#64748b",
    fontSize: 32,
    marginTop: 16,
  },
  screenshotOverlay: {
    position: "absolute" as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255, 255, 255, 0.95)",
  },
  screenshot: {
    height: 350,
    borderRadius: 16,
    boxShadow: "0 12px 60px rgba(0, 0, 0, 0.2)",
    border: "2px solid #e2e8f0",
  },
};

// Stats with their target values and stagger delays
const stats = [
  { value: 1, label: "App Rename", delay: 0 },
  { value: 9, label: "Bug Fixes", delay: 10 },
  { value: 6, label: "Screenshare Fixes", delay: 20 },
];

export const SummarySlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Phase 1: Stats (frames 0-150, ~5 seconds)
  const statsOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  const statsScale = interpolate(frame, [0, 25], [0.9, 1], {
    extrapolateRight: "clamp",
  });

  // Phase 2: Screenshot overlay (starts at frame 150)
  const screenshotStart = 150;
  const screenshotProgress = spring({
    frame: frame - screenshotStart,
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const showScreenshot = frame >= screenshotStart;
  const screenshotScale = interpolate(screenshotProgress, [0, 1], [0.8, 1]);
  const screenshotOpacity = interpolate(screenshotProgress, [0, 0.3], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Stats fade out as screenshot comes in
  const statsFadeOut = showScreenshot
    ? interpolate(frame, [screenshotStart, screenshotStart + 20], [1, 0], {
        extrapolateRight: "clamp",
      })
    : 1;

  return (
    <AbsoluteFill style={styles.container}>
      {/* Stats layer */}
      <div
        style={{
          opacity: statsOpacity * statsFadeOut,
          transform: `scale(${statsScale})`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <p style={styles.header}>Sprint Complete</p>
        <h1 style={styles.title}>Release Summary</h1>

        <div style={styles.statsContainer}>
          {stats.map((stat, index) => {
            // Count up animation with stagger
            const countStart = 40 + stat.delay;
            const countEnd = countStart + 60;
            const currentValue = Math.round(
              interpolate(frame, [countStart, countEnd], [0, stat.value], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })
            );

            // Individual stat spring in
            const statSpring = spring({
              frame: frame - (20 + stat.delay),
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
      {showScreenshot && (
        <div style={{ ...styles.screenshotOverlay, opacity: screenshotOpacity }}>
          <Img
            src={staticFile("images/4.0.2-pending-release.png")}
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
