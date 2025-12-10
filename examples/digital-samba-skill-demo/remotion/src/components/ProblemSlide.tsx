import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

// Simulated complex API code lines
const CODE_LINES = [
  "const jwt = await generateToken({",
  "  teamId: process.env.TEAM_ID,",
  "  apiKey: process.env.API_KEY,",
  "  privateKey: fs.readFileSync('./key.pem'),",
  "  expiresIn: '1h',",
  "  claims: { room: roomId, role: 'host' }",
  "});",
  "",
  "const room = await fetch('/api/rooms', {",
  "  method: 'POST',",
  "  headers: { Authorization: `Bearer ${jwt}` },",
  "  body: JSON.stringify({",
  "    name: roomName,",
  "    settings: {",
  "      recording: true,",
  "      maxParticipants: 10,",
  "      ...",
];

export const ProblemSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Code block reveal
  const codeBlockOpacity = interpolate(frame, [15, 35], [0, 1], {
    extrapolateRight: "clamp",
  });

  const codeBlockX = interpolate(frame, [15, 45], [-50, 0], {
    extrapolateRight: "clamp",
  });

  // Problem icons animation (right side)
  const iconAnimations = [60, 90, 120, 150].map((startFrame, i) => ({
    opacity: interpolate(frame, [startFrame, startFrame + 15], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
    scale: spring({
      frame: Math.max(0, frame - startFrame),
      fps,
      config: { damping: 10, stiffness: 100 },
    }),
  }));

  const problems = [
    { icon: "JWT", text: "JWT Authentication" },
    { icon: "WS", text: "WebSocket Handling" },
    { icon: "UI", text: "Video UI Components" },
    { icon: "ERR", text: "Error Management" },
  ];

  return (
    <AbsoluteFill style={{ padding: 80 }}>
      {/* Title */}
      <h2
        style={{
          fontSize: 56,
          fontWeight: 600,
          color: "#ffffff",
          marginBottom: 60,
          opacity: titleOpacity,
          textAlign: "center",
        }}
      >
        Video conferencing integration is{" "}
        <span style={{ color: "#FF6B6B" }}>complex</span>...
      </h2>

      <div
        style={{
          display: "flex",
          gap: 80,
          alignItems: "flex-start",
          justifyContent: "center",
        }}
      >
        {/* Code Block */}
        <div
          style={{
            opacity: codeBlockOpacity,
            transform: `translateX(${codeBlockX}px)`,
            background: "#1a1a1a",
            borderRadius: 16,
            padding: 30,
            border: "1px solid #333",
            width: 700,
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
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

          {/* Code lines */}
          <pre
            style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 16,
              lineHeight: 1.6,
              margin: 0,
              color: "#888",
            }}
          >
            {CODE_LINES.map((line, i) => {
              const lineOpacity = interpolate(
                frame,
                [30 + i * 3, 35 + i * 3],
                [0, 1],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
              );
              return (
                <div key={i} style={{ opacity: lineOpacity }}>
                  {highlightCode(line)}
                </div>
              );
            })}
          </pre>
        </div>

        {/* Problem Cards */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 20,
          }}
        >
          {problems.map((problem, i) => (
            <div
              key={i}
              style={{
                opacity: iconAnimations[i].opacity,
                transform: `scale(${iconAnimations[i].scale})`,
                display: "flex",
                alignItems: "center",
                gap: 16,
                background: "rgba(255, 107, 107, 0.1)",
                border: "1px solid rgba(255, 107, 107, 0.3)",
                borderRadius: 12,
                padding: "16px 24px",
              }}
            >
              <div
                style={{
                  width: 50,
                  height: 50,
                  borderRadius: 10,
                  background: "#FF6B6B",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 18,
                  fontWeight: 700,
                  color: "#fff",
                }}
              >
                {problem.icon}
              </div>
              <span
                style={{
                  color: "#fff",
                  fontSize: 20,
                  fontWeight: 500,
                }}
              >
                {problem.text}
              </span>
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Simple syntax highlighting using React elements
function highlightCode(line: string): React.ReactNode {
  const tokens: React.ReactNode[] = [];
  let remaining = line;
  let keyIndex = 0;

  const patterns: [RegExp, string][] = [
    [/^(const|await|async|return|if|else|try|catch)/, "#c586c0"],
    [/^('.*?'|".*?"|`.*?`)/, "#ce9178"],
    [/^(process\.env\.\w+|Bearer)/, "#9cdcfe"],
    [/^(\w+)(?=\s*:)/, "#9cdcfe"], // property names
    [/^(true|false|null|undefined|\d+)/, "#b5cea8"],
  ];

  while (remaining.length > 0) {
    let matched = false;

    for (const [pattern, color] of patterns) {
      const match = remaining.match(pattern);
      if (match) {
        tokens.push(
          <span key={keyIndex++} style={{ color }}>
            {match[0]}
          </span>
        );
        remaining = remaining.slice(match[0].length);
        matched = true;
        break;
      }
    }

    if (!matched) {
      // Take next character as plain text
      const nextSpecial = remaining.slice(1).search(/const|await|async|return|if|else|try|catch|'|"|`|process\.env|\d|true|false|null|undefined/);
      const take = nextSpecial === -1 ? remaining.length : nextSpecial + 1;
      tokens.push(remaining.slice(0, take));
      remaining = remaining.slice(take);
    }
  }

  return tokens;
}
