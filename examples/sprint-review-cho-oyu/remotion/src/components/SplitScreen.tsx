import { AbsoluteFill, OffthreadVideo, staticFile, getStaticFiles } from "remotion";

const styles = {
  container: {
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "row" as const,
  },
  panel: {
    flex: 1,
    display: "flex",
    flexDirection: "column" as const,
    position: "relative" as const,
  },
  divider: {
    width: 4,
    backgroundColor: "#e2e8f0",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "contain" as const,
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
    fontSize: 48,
    marginBottom: 16,
    opacity: 0.5,
  },
  placeholderText: {
    color: "#64748b",
    fontSize: 16,
  },
  placeholderFile: {
    color: "#ea580c",
    fontSize: 14,
    marginTop: 8,
    fontFamily: "monospace",
  },
  label: {
    position: "absolute" as const,
    top: 24,
    left: 24,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    color: "#1e293b",
    padding: "14px 24px",
    borderRadius: 10,
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontSize: 32,
    fontWeight: 500,
    boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
  },
  bottomLabel: {
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

interface SplitScreenProps {
  leftVideo: string;
  rightVideo: string;
  leftLabel?: string;
  rightLabel?: string;
  bottomLabel?: string;
  jiraRef?: string;
  startFrom?: number;
  leftStartFrom?: number;
  rightStartFrom?: number;
  playbackRate?: number;
}

const VideoOrPlaceholder: React.FC<{
  videoFile: string;
  startFrom: number;
  playbackRate: number;
}> = ({ videoFile, startFrom, playbackRate }) => {
  const staticFiles = getStaticFiles();
  const videoPath = `demos/${videoFile}`;
  const hasVideo = staticFiles.some((f) => f.name === videoPath);

  if (hasVideo) {
    return (
      <OffthreadVideo
        src={staticFile(videoPath)}
        startFrom={startFrom}
        playbackRate={playbackRate}
        style={styles.video}
      />
    );
  }

  return (
    <div style={styles.placeholder}>
      <div style={styles.placeholderIcon}>🎬</div>
      <div style={styles.placeholderText}>Video placeholder</div>
      <div style={styles.placeholderFile}>{videoFile}</div>
    </div>
  );
};

export const SplitScreen: React.FC<SplitScreenProps> = ({
  leftVideo,
  rightVideo,
  leftLabel = "iPhone",
  rightLabel = "Remote View",
  bottomLabel,
  jiraRef,
  startFrom = 0,
  leftStartFrom,
  rightStartFrom,
  playbackRate = 1,
}) => {
  // Allow individual overrides, fall back to shared startFrom
  const leftOffset = leftStartFrom ?? startFrom;
  const rightOffset = rightStartFrom ?? startFrom;

  return (
    <AbsoluteFill style={styles.container}>
      {/* Left panel - iPhone */}
      <div style={styles.panel}>
        <VideoOrPlaceholder videoFile={leftVideo} startFrom={leftOffset} playbackRate={playbackRate} />
        <div style={styles.label}>{leftLabel}</div>
      </div>

      <div style={styles.divider} />

      {/* Right panel - Browser/Remote */}
      <div style={styles.panel}>
        <VideoOrPlaceholder videoFile={rightVideo} startFrom={rightOffset} playbackRate={playbackRate} />
        <div style={styles.label}>{rightLabel}</div>
      </div>

      {/* Bottom label */}
      {bottomLabel && (
        <div style={styles.bottomLabel}>
          {bottomLabel}
          {jiraRef && <span style={styles.jira}>{jiraRef}</span>}
        </div>
      )}
    </AbsoluteFill>
  );
};
