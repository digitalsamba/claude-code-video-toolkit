import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  getStaticFiles,
} from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    padding: 60,
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
    fontSize: 64,
    fontWeight: 700,
    margin: "0 0 50px 0",
  },
  content: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 40,
  },
  stepsRow: {
    display: "flex",
    justifyContent: "center",
    gap: 100,
  },
  stepContainer: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 28,
  },
  appRow: {
    display: "flex",
    alignItems: "flex-start",
    gap: 40,
  },
  appCard: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 14,
    transformOrigin: "top center",
  },
  appIcon: {
    width: 130,
    height: 130,
    borderRadius: 28,
    backgroundColor: "#f1f5f9",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 40,
    color: "#64748b",
    border: "2px solid #e2e8f0",
  },
  appIconImg: {
    width: 130,
    height: 130,
    borderRadius: 28,
    objectFit: "cover" as const,
  },
  appLabel: {
    color: "#1e293b",
    fontSize: 22,
    fontWeight: 500,
    textAlign: "center" as const,
    maxWidth: 180,
  },
  appSublabel: {
    color: "#64748b",
    fontSize: 17,
    marginTop: 2,
  },
  rowLabel: {
    color: "#ea580c",
    fontSize: 20,
    fontWeight: 600,
    minWidth: 100,
    textTransform: "uppercase" as const,
    letterSpacing: 2,
  },
  arrow: {
    fontSize: 42,
    color: "#ea580c",
  },
  upgradesRow: {
    display: "flex",
    justifyContent: "center",
    gap: 80,
  },
  upgradeBox: {
    backgroundColor: "#f8fafc",
    borderRadius: 20,
    padding: "28px 48px",
    border: "2px solid #e2e8f0",
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 14,
  },
  upgradeLabel: {
    color: "#64748b",
    fontSize: 22,
    fontWeight: 500,
  },
  upgradeVersions: {
    display: "flex",
    alignItems: "center",
    gap: 24,
  },
  version: {
    color: "#ea580c",
    fontSize: 42,
    fontWeight: 700,
    fontFamily: "monospace",
  },
  versionSecondary: {
    color: "#64748b",
    fontSize: 36,
    fontWeight: 600,
    fontFamily: "monospace",
  },
  versionArrow: {
    color: "#22c55e",
    fontSize: 36,
  },
  upgradeSubtext: {
    color: "#94a3b8",
    fontSize: 18,
    marginTop: 8,
    fontFamily: "monospace",
  },
};

const ImageOrPlaceholder: React.FC<{
  src: string;
  alt: string;
  style: React.CSSProperties;
}> = ({ src, alt, style }) => {
  const staticFiles = getStaticFiles();
  const hasImage = staticFiles.some((f) => f.name === src);

  if (hasImage) {
    return <Img src={staticFile(src)} style={style} />;
  }

  return (
    <div style={styles.appIcon}>
      <span>{alt === "before" ? "Old" : "New"}</span>
    </div>
  );
};

export const AppRenameSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animation sequence:
  // 1. Title (0-20)
  // 2. Step 1 - Legacy rename (20-70)
  // 3. React Native upgrade box with slow count (80-180)
  // 4. Step 2 - Embedded rename (190-260)
  // 5. App Version box (270-290)

  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Step 1: Legacy app animations
  const step1Opacity = interpolate(frame, [20, 35], [0, 1], {
    extrapolateRight: "clamp",
  });

  const legacyBeforeScale = spring({
    frame: frame - 25,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const arrow1Opacity = interpolate(frame, [45, 55], [0, 1], {
    extrapolateRight: "clamp",
  });

  const legacyAfterScale = spring({
    frame: frame - 55,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  // React Native upgrade - appears after Step 1, before Step 2
  const rnUpgradeOpacity = interpolate(frame, [80, 95], [0, 1], {
    extrapolateRight: "clamp",
  });

  const rnUpgradeY = interpolate(frame, [80, 95], [20, 0], {
    extrapolateRight: "clamp",
  });

  // Slow version count (0.68 to 0.77) over ~3 seconds
  const versionProgress = interpolate(frame, [100, 190], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const currentVersion = 0.68 + (0.77 - 0.68) * versionProgress;

  // Legacy app version (1.35.0 to 1.36.0) - animates alongside RN upgrade
  const legacyVersionProgress = interpolate(frame, [120, 190], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const legacyMinorVersion = 35 + (36 - 35) * legacyVersionProgress;

  // Step 2: Embedded app animations - starts after RN upgrade completes
  const step2Opacity = interpolate(frame, [200, 215], [0, 1], {
    extrapolateRight: "clamp",
  });

  const embeddedBeforeScale = spring({
    frame: frame - 210,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const arrow2Opacity = interpolate(frame, [230, 240], [0, 1], {
    extrapolateRight: "clamp",
  });

  const embeddedAfterScale = spring({
    frame: frame - 240,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  // App version box - appears after Step 2 completes
  const appVersionOpacity = interpolate(frame, [270, 285], [0, 1], {
    extrapolateRight: "clamp",
  });

  const appVersionY = interpolate(frame, [270, 285], [20, 0], {
    extrapolateRight: "clamp",
  });

  // App version animation (4.0.1 to 4.0.2) - counts up after box appears
  const appPatchVersion = interpolate(frame, [290, 350], [1, 2], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={styles.container}>
      <div style={{ opacity: titleOpacity, textAlign: "center" as const }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 20 }}>
          <p style={styles.header}>Legacy App Migration</p>
          <span style={{ color: "#64748b", fontSize: 24, fontFamily: "monospace" }}>
            SAMBA-8117
          </span>
        </div>
        <h1 style={styles.title}>Making Room for Digital Samba</h1>
      </div>

      <div style={styles.content}>
        {/* Steps side by side */}
        <div style={styles.stepsRow}>
          {/* Step 1: Legacy Angular app rename */}
          <div style={{ ...styles.stepContainer, opacity: step1Opacity }}>
            <span style={styles.rowLabel}>Step 1</span>
            <div style={styles.appRow}>
              <div
                style={{
                  ...styles.appCard,
                  transform: `scale(${legacyBeforeScale})`,
                }}
              >
                <Img src={staticFile("images/legacy.png")} style={styles.appIconImg} />
                <div style={styles.appLabel}>Digital Samba</div>
                <div style={styles.appSublabel}>Angular App</div>
              </div>

              <div style={{ ...styles.arrow, opacity: arrow1Opacity }}>→</div>

              <div
                style={{
                  ...styles.appCard,
                  transform: `scale(${legacyAfterScale})`,
                }}
              >
                <Img src={staticFile("images/legacy.png")} style={styles.appIconImg} />
                <div style={styles.appLabel}>Digital Samba (legacy)</div>
                <div style={styles.appSublabel}>Renamed</div>
              </div>
            </div>
          </div>

          {/* Step 2: Embedded app gets the name */}
          <div style={{ ...styles.stepContainer, opacity: step2Opacity }}>
            <span style={styles.rowLabel}>Step 2</span>
            <div style={styles.appRow}>
              <div
                style={{
                  ...styles.appCard,
                  transform: `scale(${embeddedBeforeScale})`,
                }}
              >
                <Img src={staticFile("images/ds-logo.png")} style={styles.appIconImg} />
                <div style={styles.appLabel}>Digital Samba E</div>
                <div style={styles.appSublabel}>v4.0.1</div>
              </div>

              <div style={{ ...styles.arrow, opacity: arrow2Opacity }}>→</div>

              <div
                style={{
                  ...styles.appCard,
                  transform: `scale(${embeddedAfterScale})`,
                }}
              >
                <Img src={staticFile("images/ds-logo.png")} style={styles.appIconImg} />
                <div style={styles.appLabel}>Digital Samba</div>
                <div style={styles.appSublabel}>v4.0.2</div>
              </div>
            </div>
          </div>
        </div>

        {/* Version upgrades row - showing correlation */}
        <div style={styles.upgradesRow}>
          {/* RN upgrade - appears after Step 1, enables Step 2 */}
          <div
            style={{
              ...styles.upgradeBox,
              opacity: rnUpgradeOpacity,
              transform: `translateY(${rnUpgradeY}px)`,
            }}
          >
            <div style={styles.upgradeLabel}>React Native Upgrade</div>
            <div style={styles.upgradeVersions}>
              <span style={styles.versionSecondary}>0.68</span>
              <span style={styles.versionArrow}>→</span>
              <span style={styles.versionSecondary}>
                {currentVersion.toFixed(2)}
              </span>
            </div>
            <div style={styles.upgradeSubtext}>
              Legacy App: 1.{Math.round(legacyMinorVersion)}.0
            </div>
          </div>

          {/* App version - appears after Step 2 */}
          <div
            style={{
              ...styles.upgradeBox,
              opacity: appVersionOpacity,
              transform: `translateY(${appVersionY}px)`,
            }}
          >
            <div style={styles.upgradeLabel}>App Version</div>
            <div style={styles.upgradeVersions}>
              <span style={styles.version}>4.0.1</span>
              <span style={styles.versionArrow}>→</span>
              <span style={styles.version}>
                4.0.{Math.round(appPatchVersion)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
