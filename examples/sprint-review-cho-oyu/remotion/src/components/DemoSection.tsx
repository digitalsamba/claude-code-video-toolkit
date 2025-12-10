import { AbsoluteFill, OffthreadVideo, staticFile, getStaticFiles } from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
  },
  placeholder: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  placeholderIcon: {
    fontSize: 80,
    marginBottom: 24,
    opacity: 0.5,
  },
  placeholderText: {
    color: "#64748b",
    fontSize: 24,
  },
  placeholderFile: {
    color: "#ea580c",
    fontSize: 20,
    marginTop: 8,
    fontFamily: "monospace",
  },
  label: {
    position: "absolute" as const,
    bottom: 48,
    left: 48,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    color: "#1e293b",
    padding: "16px 32px",
    borderRadius: 12,
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontSize: 32,
    fontWeight: 500,
    boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
  },
  jira: {
    color: "#ea580c",
    marginLeft: 16,
    fontSize: 24,
  },
};

interface DemoSectionProps {
  videoFile: string;
  label?: string;
  jiraRef?: string;
  startFrom?: number;
  playbackRate?: number;
}

export const DemoSection: React.FC<DemoSectionProps> = ({
  videoFile,
  label,
  jiraRef,
  startFrom = 0,
  playbackRate = 1,
}) => {
  // Check if video file exists
  const staticFiles = getStaticFiles();
  const videoPath = `demos/${videoFile}`;
  const hasVideo = staticFiles.some((f) => f.name === videoPath);

  return (
    <AbsoluteFill style={styles.container}>
      {hasVideo ? (
        <OffthreadVideo
          src={staticFile(videoPath)}
          startFrom={startFrom}
          playbackRate={playbackRate}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
          }}
        />
      ) : (
        <div style={styles.placeholder}>
          <div style={styles.placeholderIcon}>🎬</div>
          <div style={styles.placeholderText}>Video placeholder</div>
          <div style={styles.placeholderFile}>public/demos/{videoFile}</div>
        </div>
      )}

      {label && (
        <div style={styles.label}>
          {label}
          {jiraRef && <span style={styles.jira}>{jiraRef}</span>}
        </div>
      )}
    </AbsoluteFill>
  );
};
