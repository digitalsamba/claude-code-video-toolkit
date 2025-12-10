import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Img,
  staticFile,
} from "remotion";

export const TitleSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo animations
  const dsLogoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const dsLogoOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Title animation (staggered)
  const titleOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleY = interpolate(frame, [20, 45], [30, 0], {
    extrapolateRight: "clamp",
  });

  // Subtitle animation
  const subtitleOpacity = interpolate(frame, [45, 65], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const subtitleY = interpolate(frame, [45, 70], [20, 0], {
    extrapolateRight: "clamp",
  });

  // Accent line animation
  const lineWidth = interpolate(frame, [35, 60], [0, 400], {
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
          gap: 40,
        }}
      >
        {/* Dual Logos: Claude + Digital Samba */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 60,
            opacity: dsLogoOpacity,
            transform: `scale(${dsLogoScale})`,
          }}
        >
          {/* Claude Logo */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            <Img
              src={staticFile("images/claude-icon.webp")}
              style={{
                width: 80,
                height: 80,
                objectFit: "contain",
              }}
            />
            <span
              style={{
                fontSize: 36,
                fontWeight: 600,
                color: "#D97706",
              }}
            >
              Claude
            </span>
          </div>

          {/* Plus sign */}
          <span
            style={{
              fontSize: 48,
              color: "#444",
              fontWeight: 300,
            }}
          >
            +
          </span>

          {/* Digital Samba Logo */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            <Img
              src={staticFile("images/ds-logo.png")}
              style={{
                width: 80,
                height: 80,
                objectFit: "contain",
              }}
            />
            <span
              style={{
                fontSize: 36,
                fontWeight: 600,
                color: "#0066FF",
              }}
            >
              Digital Samba
            </span>
          </div>
        </div>

        {/* Main Title */}
        <h1
          style={{
            fontSize: 72,
            fontWeight: 700,
            color: "#ffffff",
            margin: 0,
            textAlign: "center",
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            letterSpacing: "-1px",
          }}
        >
          Integrate Video Conferencing
          <br />
          <span style={{ color: "#0066FF" }}>in Minutes</span>
        </h1>

        {/* Accent Line */}
        <div
          style={{
            width: lineWidth,
            height: 3,
            background: "linear-gradient(90deg, #0066FF, #00D4AA)",
            borderRadius: 2,
          }}
        />

        {/* Subtitle */}
        <p
          style={{
            fontSize: 28,
            color: "#888",
            margin: 0,
            opacity: subtitleOpacity,
            transform: `translateY(${subtitleY}px)`,
          }}
        >
          Build video features into your apps faster with Digital Samba and Claude
        </p>
      </div>
    </AbsoluteFill>
  );
};
