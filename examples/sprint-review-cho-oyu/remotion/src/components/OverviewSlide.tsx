import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    padding: 80,
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  header: {
    color: "#ea580c",
    fontSize: 34,
    fontWeight: 500,
    marginBottom: 20,
    textTransform: "uppercase" as const,
    letterSpacing: 3,
  },
  title: {
    color: "#1e293b",
    fontSize: 78,
    fontWeight: 700,
    margin: "0 0 60px 0",
    textAlign: "center" as const,
  },
  list: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 34,
  },
  item: {
    display: "flex",
    alignItems: "center",
    gap: 24,
  },
  bullet: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: "#ea580c",
    flexShrink: 0,
  },
  text: {
    color: "#334155",
    fontSize: 44,
  },
  highlight: {
    color: "#ea580c",
    fontWeight: 600,
  },
};

const items = [
  { text: "App renamed to ", highlight: "'Digital Samba'" },
  { text: "Screen Sharing: ", highlight: "6 stability fixes" },
  { text: "Connection: ", highlight: "Sleep/lock & background persistence" },
  { text: "Deep Linking: ", highlight: "Warm start fix" },
];

export const OverviewSlide: React.FC = () => {
  const frame = useCurrentFrame();

  // Fade in header and title
  const opacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ ...styles.container, opacity }}>
      <p style={styles.header}>Sprint Overview</p>
      <h1 style={styles.title}>What's New in v4.0.2</h1>

      <div style={styles.list}>
        {items.map((item, index) => {
          // Stagger animation - spread across ~10 seconds of the 15s sequence
          const itemStart = 60 + index * 75; // Start at 2s, 2.5s apart
          const itemEnd = itemStart + 30; // 1 second animation each
          const itemOpacity = interpolate(
            frame,
            [itemStart, itemEnd],
            [0, 1],
            { extrapolateRight: "clamp", extrapolateLeft: "clamp" }
          );
          const itemX = interpolate(
            frame,
            [itemStart, itemEnd],
            [-30, 0],
            { extrapolateRight: "clamp", extrapolateLeft: "clamp" }
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
