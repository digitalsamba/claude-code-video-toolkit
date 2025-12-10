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
  skipOverlay: {
    position: "absolute" as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "#000",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  skipText: {
    color: "#665500",
    fontSize: 32,
    fontFamily: "system-ui, -apple-system, sans-serif",
    letterSpacing: 4,
    textTransform: "uppercase" as const,
  },
};

// Sequence timing (in frames at 30fps):
// - Intro: 0-150 (5s) - 60s video at 12x, timer 0:00→1:00
// - Skip: 150-180 (1s) - fade transition, timer jumps 1:00→6:00
// - Return: 180-270 (3s) - 14s video at ~4.7x, timer 6:00→6:14
// - Outro: 270-450 (6s) - 6s video at 1x, timer 6:14→6:20

const INTRO_FRAMES = 150; // 5s
const SKIP_FRAMES = 30; // 1s
const RETURN_FRAMES = 90; // 3s
const OUTRO_FRAMES = 180; // 6s

const INTRO_END = INTRO_FRAMES;
const SKIP_END = INTRO_END + SKIP_FRAMES;
const RETURN_END = SKIP_END + RETURN_FRAMES;
const TOTAL_FRAMES = RETURN_END + OUTRO_FRAMES; // 450 frames = 15s

interface TimelapseDemoWithSkipProps {
  label: string;
  jiraRef?: string;
}

export const TimelapseDemoWithSkip: React.FC<TimelapseDemoWithSkipProps> = ({
  label,
  jiraRef,
}) => {
  const frame = useCurrentFrame();
  const FPS = 30;

  // Determine current phase
  const isIntro = frame < INTRO_END;
  const isSkip = frame >= INTRO_END && frame < SKIP_END;
  const isReturn = frame >= SKIP_END && frame < RETURN_END;
  const isOutro = frame >= RETURN_END;

  // Calculate elapsed video time based on phase
  let elapsedVideoSeconds: number;
  let currentSpeed: number;

  if (isIntro) {
    // 0:00 to 1:00 over 5 seconds
    elapsedVideoSeconds = (frame / INTRO_FRAMES) * 60;
    currentSpeed = 12;
  } else if (isSkip) {
    // Quick jump from 1:00 to 6:00 over 1 second
    const skipProgress = (frame - INTRO_END) / SKIP_FRAMES;
    elapsedVideoSeconds = 60 + skipProgress * (360 - 60);
    currentSpeed = 0; // No video showing
  } else if (isReturn) {
    // 6:00 to 6:14 over 3 seconds
    const returnProgress = (frame - SKIP_END) / RETURN_FRAMES;
    elapsedVideoSeconds = 360 + returnProgress * 14;
    currentSpeed = 4.7;
  } else {
    // 6:14 to 6:20 over 6 seconds (normal speed)
    const outroProgress = (frame - RETURN_END) / OUTRO_FRAMES;
    elapsedVideoSeconds = 374 + outroProgress * 6;
    currentSpeed = 1;
  }

  const minutes = Math.floor(elapsedVideoSeconds / 60);
  const seconds = Math.floor(elapsedVideoSeconds % 60);
  const timeDisplay = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

  // Timer visibility
  const timerOpacity = interpolate(
    frame,
    [0, 30, TOTAL_FRAMES - 30, TOTAL_FRAMES],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Skip overlay fade
  const skipFadeIn = interpolate(
    frame,
    [INTRO_END - 15, INTRO_END],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const skipFadeOut = interpolate(
    frame,
    [SKIP_END, SKIP_END + 15],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const skipOpacity = isSkip ? 1 : (frame < SKIP_END ? skipFadeIn : skipFadeOut);
  const showSkipOverlay = frame >= INTRO_END - 15 && frame < SKIP_END + 15;

  // Speed indicator visibility (hide during skip and outro)
  const showSpeed = (isIntro || isReturn) && currentSpeed > 1;

  return (
    <AbsoluteFill>
      {/* Intro segment: 60s video at 12x speed */}
      <Sequence durationInFrames={INTRO_FRAMES}>
        <OffthreadVideo
          src={staticFile("demos/background-screenshare-intro.mp4")}
          playbackRate={12}
          style={styles.video}
        />
      </Sequence>

      {/* Return segment: 14s video at ~4.7x speed */}
      <Sequence from={SKIP_END} durationInFrames={RETURN_FRAMES}>
        <OffthreadVideo
          src={staticFile("demos/background-screenshare-return.mp4")}
          playbackRate={4.67}
          style={styles.video}
        />
      </Sequence>

      {/* Outro segment: 6s video at normal speed */}
      <Sequence from={RETURN_END} durationInFrames={OUTRO_FRAMES}>
        <OffthreadVideo
          src={staticFile("demos/background-screenshare-outro.mp4")}
          style={styles.video}
        />
      </Sequence>

      {/* Skip overlay */}
      {showSkipOverlay && (
        <div style={{ ...styles.skipOverlay, opacity: skipOpacity }}>
          <span style={styles.skipText}>time passing...</span>
        </div>
      )}

      {/* Speed indicator */}
      {showSpeed && (
        <div style={{ ...styles.speedOverlay, opacity: timerOpacity }}>
          <span style={styles.speedValue}>{currentSpeed.toFixed(1)}x</span>
          <span>speed</span>
        </div>
      )}

      {/* Timer */}
      <div style={{ ...styles.timerOverlay, opacity: timerOpacity }}>
        <span style={styles.timerTime}>{timeDisplay}</span>
        <span style={styles.timerLabel}>elapsed</span>
      </div>

      {/* Label */}
      <div style={styles.label}>
        {label}
        {jiraRef && <span style={styles.jira}>{jiraRef}</span>}
      </div>
    </AbsoluteFill>
  );
};
