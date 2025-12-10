import { AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, interpolate, Sequence } from "remotion";

const styles = {
  video: {
    width: "100%",
    height: "100%",
    objectFit: "contain" as const,
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
  speedOverlay: {
    position: "absolute" as const,
    top: 40,
    left: 48,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    color: "#fff",
    padding: "12px 24px",
    borderRadius: 8,
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontSize: 24,
    fontWeight: 600,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  speedValue: {
    color: "#e8c500",
    fontSize: 28,
    fontWeight: 700,
  },
  timerOverlay: {
    position: "absolute" as const,
    top: 40,
    right: 40,
    backgroundColor: "#000",
    padding: "24px 40px",
    borderRadius: 4,
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    gap: 8,
    boxShadow: "0 0 60px rgba(0, 0, 0, 0.8)",
  },
  timerTime: {
    color: "#e8c500",
    fontSize: 96,
    fontWeight: 700,
    fontFamily: "'Courier New', monospace",
    letterSpacing: 8,
    textShadow: "0 0 30px rgba(232, 197, 0, 0.9), 0 0 60px rgba(232, 197, 0, 0.5), 0 0 90px rgba(232, 197, 0, 0.3)",
  },
  timerLabel: {
    color: "#665500",
    fontSize: 16,
    textTransform: "uppercase" as const,
    letterSpacing: 4,
    textShadow: "0 0 10px rgba(232, 197, 0, 0.3)",
  },
};

interface TimelapseDemoProps {
  videoFile: string;
  label: string;
  jiraRef?: string;
  videoDurationSeconds: number;
  sequenceDurationSeconds: number;
  outroVideoFile?: string; // Separate video file for normal speed outro
  outroSeconds?: number; // Duration of outro in seconds
}

export const TimelapseDemo: React.FC<TimelapseDemoProps> = ({
  videoFile,
  label,
  jiraRef,
  videoDurationSeconds,
  sequenceDurationSeconds,
  outroVideoFile,
  outroSeconds = 0,
}) => {
  const frame = useCurrentFrame();
  const FPS = 30;
  const sequenceDurationFrames = sequenceDurationSeconds * FPS;

  // Calculate split between timelapse and normal speed outro
  const outroSequenceFrames = outroSeconds * FPS;
  const timelapseSequenceFrames = sequenceDurationFrames - outroSequenceFrames;

  const timelapseVideoSeconds = videoDurationSeconds - outroSeconds;
  const timelapsePlaybackRate = timelapseVideoSeconds / (timelapseSequenceFrames / FPS);

  // Which phase are we in?
  const isTimelapse = frame < timelapseSequenceFrames;

  // Timer: show elapsed video time
  let elapsedVideoSeconds: number;
  if (isTimelapse) {
    elapsedVideoSeconds = (frame / timelapseSequenceFrames) * timelapseVideoSeconds;
  } else {
    const outroFrame = frame - timelapseSequenceFrames;
    elapsedVideoSeconds = timelapseVideoSeconds + (outroFrame / FPS);
  }

  const minutes = Math.floor(elapsedVideoSeconds / 60);
  const seconds = Math.floor(elapsedVideoSeconds % 60);
  const timeDisplay = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

  // Timer fade in/out
  const timerOpacity = interpolate(
    frame,
    [0, 30, sequenceDurationFrames - 30, sequenceDurationFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill>
      {/* Timelapse portion - Sequence limits it to first N frames */}
      <Sequence durationInFrames={timelapseSequenceFrames}>
        <OffthreadVideo
          src={staticFile(`demos/${videoFile}`)}
          playbackRate={timelapsePlaybackRate}
          style={styles.video}
        />
      </Sequence>

      {/* Normal speed outro - Sequence from={X} makes child receive frames starting at 0 */}
      {outroVideoFile && (
        <Sequence from={timelapseSequenceFrames} durationInFrames={outroSequenceFrames}>
          <OffthreadVideo
            src={staticFile(`demos/${outroVideoFile}`)}
            style={styles.video}
          />
        </Sequence>
      )}

      {/* Speed indicator - only during timelapse */}
      <div style={{ ...styles.speedOverlay, opacity: isTimelapse ? timerOpacity : 0 }}>
        <span style={styles.speedValue}>{timelapsePlaybackRate.toFixed(1)}x</span>
        <span>speed</span>
      </div>

      <div style={{ ...styles.timerOverlay, opacity: timerOpacity }}>
        <span style={styles.timerTime}>{timeDisplay}</span>
        <span style={styles.timerLabel}>elapsed</span>
      </div>

      <div style={styles.label}>
        {label}
        {jiraRef && <span style={styles.jira}>{jiraRef}</span>}
      </div>
    </AbsoluteFill>
  );
};
