import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

// Simulated Claude building app - terminal output lines
const TERMINAL_LINES = [
  { type: "prompt", text: "$ claude" },
  { type: "input", text: "> Build an interview room app with Digital Samba" },
  { type: "output", text: "" },
  { type: "claude", text: "I'll create an interview room application using the Digital Samba SDK." },
  { type: "claude", text: "Let me set up the project structure..." },
  { type: "output", text: "" },
  { type: "action", text: "Creating src/app/page.tsx..." },
  { type: "action", text: "Creating src/app/room/[id]/page.tsx..." },
  { type: "action", text: "Creating src/lib/digital-samba.ts..." },
  { type: "action", text: "Creating src/components/RoomCard.tsx..." },
  { type: "output", text: "" },
  { type: "claude", text: "Setting up JWT authentication with your API keys..." },
  { type: "action", text: "Creating src/app/api/token/route.ts..." },
  { type: "output", text: "" },
  { type: "claude", text: "Adding the Digital Samba Embedded SDK integration..." },
  { type: "action", text: "Installing @anthropic-ai/sdk..." },
  { type: "success", text: "✓ Interview room app created successfully!" },
];

export const BuildingApp: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Header animation
  const headerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Terminal window animation
  const terminalScale = spring({
    frame: Math.max(0, frame - 20),
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const terminalOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Calculate visible lines based on frame
  const visibleLines = Math.floor(
    interpolate(frame, [50, 800], [0, TERMINAL_LINES.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );

  // Creative freedom callout
  const calloutOpacity = interpolate(frame, [450, 480], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const calloutScale = spring({
    frame: Math.max(0, frame - 450),
    fps,
    config: { damping: 15, stiffness: 100 },
  });

  return (
    <AbsoluteFill
      style={{
        padding: 60,
      }}
    >
      {/* Header */}
      <div
        style={{
          opacity: headerOpacity,
          textAlign: "center",
          marginBottom: 40,
        }}
      >
        <h2
          style={{
            fontSize: 42,
            fontWeight: 600,
            color: "#ffffff",
            margin: 0,
          }}
        >
          Claude Builds the App
        </h2>
      </div>

      <div
        style={{
          display: "flex",
          gap: 40,
          height: "calc(100% - 120px)",
        }}
      >
        {/* Terminal Window */}
        <div
          style={{
            flex: 1,
            opacity: terminalOpacity,
            transform: `scale(${terminalScale})`,
            background: "#1a1a1a",
            borderRadius: 16,
            border: "1px solid #333",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            boxShadow: "0 30px 80px rgba(0,0,0,0.5)",
          }}
        >
          {/* Window header */}
          <div
            style={{
              background: "#2a2a2a",
              padding: "12px 16px",
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}
          >
            <div style={{ display: "flex", gap: 8 }}>
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
            <span style={{ color: "#666", fontSize: 14 }}>
              Claude Code — Building Interview Room
            </span>
          </div>

          {/* Terminal content */}
          <div
            style={{
              flex: 1,
              padding: 24,
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 16,
              lineHeight: 1.8,
              overflowY: "auto",
            }}
          >
            {TERMINAL_LINES.slice(0, visibleLines).map((line, i) => (
              <div key={i} style={getLineStyle(line.type)}>
                {line.text}
              </div>
            ))}

            {/* Cursor */}
            {visibleLines < TERMINAL_LINES.length && (
              <span
                style={{
                  color: "#00D4AA",
                  opacity: Math.sin(frame * 0.3) > 0 ? 1 : 0,
                }}
              >
                _
              </span>
            )}
          </div>
        </div>

        {/* Creative Freedom Callout */}
        <div
          style={{
            width: 400,
            opacity: calloutOpacity,
            transform: `scale(${calloutScale})`,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              background: "linear-gradient(135deg, rgba(0, 102, 255, 0.15), rgba(0, 212, 170, 0.15))",
              border: "1px solid rgba(0, 102, 255, 0.3)",
              borderRadius: 20,
              padding: 40,
            }}
          >
            <div
              style={{
                fontSize: 48,
                marginBottom: 20,
              }}
            >
              🎨
            </div>
            <h3
              style={{
                fontSize: 28,
                fontWeight: 600,
                color: "#fff",
                margin: 0,
                marginBottom: 16,
              }}
            >
              Creative Freedom
            </h3>
            <p
              style={{
                fontSize: 18,
                color: "#aaa",
                margin: 0,
                lineHeight: 1.6,
              }}
            >
              We gave Claude creative freedom to interpret what an online
              interview experience should look like.
            </p>
          </div>

          {/* Tech Stack */}
          <div
            style={{
              marginTop: 30,
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <TechBadge label="Next.js 15" color="#000" />
            <TechBadge label="Digital Samba SDK" color="#0066FF" />
            <TechBadge label="TypeScript" color="#3178C6" />
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

function getLineStyle(type: string): React.CSSProperties {
  const baseStyle: React.CSSProperties = { minHeight: 24 };

  switch (type) {
    case "prompt":
      return { ...baseStyle, color: "#00D4AA" };
    case "input":
      return { ...baseStyle, color: "#D97706", paddingLeft: 20 };
    case "claude":
      return { ...baseStyle, color: "#0066FF" };
    case "action":
      return { ...baseStyle, color: "#888", paddingLeft: 20 };
    case "success":
      return { ...baseStyle, color: "#27C93F", fontWeight: 600 };
    default:
      return { ...baseStyle, color: "#666" };
  }
}

const TechBadge: React.FC<{ label: string; color: string }> = ({
  label,
  color,
}) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      gap: 12,
      background: "rgba(255,255,255,0.05)",
      borderRadius: 8,
      padding: "10px 16px",
    }}
  >
    <div
      style={{
        width: 12,
        height: 12,
        borderRadius: 4,
        background: color,
      }}
    />
    <span style={{ color: "#fff", fontSize: 16 }}>{label}</span>
  </div>
);
