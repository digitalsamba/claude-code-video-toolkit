import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, getStaticFiles } from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  title: {
    color: "#1e293b",
    fontSize: 88,
    fontWeight: 700,
    margin: 0,
    textAlign: "center" as const,
  },
  sprintLine: {
    color: "#1e293b",
    fontSize: 52,
    fontWeight: 500,
    margin: "24px 0 10px 0",
  },
  date: {
    color: "#64748b",
    fontSize: 34,
    margin: "0 0 48px 0",
  },
  sprintName: {
    color: "#ea580c",
    fontWeight: 600,
  },
  sectionTitle: {
    color: "#1e293b",
    fontSize: 48,
    fontWeight: 700,
    margin: "0 0 10px 0",
  },
  version: {
    color: "#64748b",
    fontSize: 38,
    fontWeight: 500,
    margin: 0,
  },
  logo: {
    width: 200,
    height: 200,
    marginBottom: 48,
  },
};

export const TitleSlide: React.FC = () => {
  const frame = useCurrentFrame();

  // Scale up animation (0.95 → 1.0)
  const scale = interpolate(frame, [0, 30], [0.95, 1], {
    extrapolateRight: "clamp",
  });

  // Staggered fade timings (each element starts 6 frames after previous)
  const logoOpacity = interpolate(frame, [0, 18], [0, 1], { extrapolateRight: "clamp" });
  const titleOpacity = interpolate(frame, [6, 24], [0, 1], { extrapolateRight: "clamp" });
  const sprintOpacity = interpolate(frame, [12, 30], [0, 1], { extrapolateRight: "clamp" });
  const dateOpacity = interpolate(frame, [18, 36], [0, 1], { extrapolateRight: "clamp" });
  const sectionOpacity = interpolate(frame, [24, 42], [0, 1], { extrapolateRight: "clamp" });
  const versionOpacity = interpolate(frame, [30, 48], [0, 1], { extrapolateRight: "clamp" });

  const staticFiles = getStaticFiles();
  const hasLogo = staticFiles.some((f) => f.name === "images/ds-logo.png");

  return (
    <AbsoluteFill style={styles.container}>
      <div style={{ transform: `scale(${scale})`, display: "flex", flexDirection: "column", alignItems: "center" }}>
        {hasLogo ? (
          <Img src={staticFile("images/ds-logo.png")} style={{ ...styles.logo, opacity: logoOpacity }} />
        ) : (
          <div style={{ ...styles.logo, opacity: logoOpacity, backgroundColor: "#ea580c", borderRadius: 32, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 48, fontWeight: 700 }}>DS</div>
        )}
        <h1 style={{ ...styles.title, opacity: titleOpacity }}>Digital Samba Mobile</h1>
        <p style={{ ...styles.sprintLine, opacity: sprintOpacity }}>Sprint Review : <span style={styles.sprintName}>Cho Oyu</span></p>
        <p style={{ ...styles.date, opacity: dateOpacity }}>24th Nov – 8th Dec</p>
        <p style={{ ...styles.sectionTitle, opacity: sectionOpacity }}>iOS Embedded App Update</p>
        <p style={{ ...styles.version, opacity: versionOpacity }}>Version 4.0.2 (BUILD 233)</p>
      </div>
    </AbsoluteFill>
  );
};
