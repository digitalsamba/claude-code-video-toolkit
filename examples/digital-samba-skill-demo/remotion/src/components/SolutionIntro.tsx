import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const SolutionIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Question text animation
  const questionOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  const questionY = interpolate(frame, [0, 25], [20, 0], {
    extrapolateRight: "clamp",
  });

  // Terminal animation
  const terminalScale = spring({
    frame: Math.max(0, frame - 30),
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const terminalOpacity = interpolate(frame, [30, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Typing animation for terminal
  const terminalText = "claude --skill digital-samba";
  const typedChars = Math.floor(
    interpolate(frame, [60, 120], [0, terminalText.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );

  // Cursor blink
  const cursorOpacity = Math.sin(frame * 0.3) > 0 ? 1 : 0;

  // Glow effect
  const glowIntensity = interpolate(frame, [100, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 60,
        }}
      >
        {/* Question */}
        <h2
          style={{
            fontSize: 52,
            fontWeight: 500,
            color: "#ffffff",
            margin: 0,
            opacity: questionOpacity,
            transform: `translateY(${questionY}px)`,
            textAlign: "center",
          }}
        >
          What if <span style={{ color: "#D97706" }}>AI</span> understood
          <br />
          your video API?
        </h2>

        {/* Terminal Window */}
        <div
          style={{
            opacity: terminalOpacity,
            transform: `scale(${terminalScale})`,
            background: "#1a1a1a",
            borderRadius: 16,
            padding: 30,
            border: "1px solid #333",
            width: 700,
            boxShadow: `
              0 20px 60px rgba(0,0,0,0.5),
              0 0 ${60 * glowIntensity}px rgba(0, 102, 255, ${0.3 * glowIntensity})
            `,
          }}
        >
          {/* Window controls */}
          <div
            style={{
              display: "flex",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: "#FF5F56",
              }}
            />
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: "#FFBD2E",
              }}
            />
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: "#27C93F",
              }}
            />
          </div>

          {/* Terminal content */}
          <div
            style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 24,
            }}
          >
            <span style={{ color: "#00D4AA" }}>$</span>{" "}
            <span style={{ color: "#ffffff" }}>
              {terminalText.slice(0, typedChars)}
            </span>
            <span
              style={{
                color: "#ffffff",
                opacity: cursorOpacity,
              }}
            >
              _
            </span>
          </div>
        </div>

        {/* Tagline */}
        <p
          style={{
            fontSize: 28,
            color: "#666",
            margin: 0,
            opacity: interpolate(frame, [140, 160], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          Introducing the{" "}
          <span style={{ color: "#0066FF" }}>Digital Samba</span> skill for
          Claude Code
        </p>
      </div>
    </AbsoluteFill>
  );
};
