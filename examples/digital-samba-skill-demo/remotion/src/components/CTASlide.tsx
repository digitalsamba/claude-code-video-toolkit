import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Img,
  staticFile,
} from "remotion";

export const CTASlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo animation
  const logoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const logoOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Title animation
  const titleOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleY = interpolate(frame, [20, 45], [20, 0], {
    extrapolateRight: "clamp",
  });

  // Links animation
  const link1Opacity = interpolate(frame, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const link2Opacity = interpolate(frame, [70, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Glow pulse
  const glowPulse =
    0.3 + Math.sin(frame * 0.05) * 0.1;

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
        background: `
          radial-gradient(ellipse at 50% 40%, rgba(0, 102, 255, 0.15) 0%, transparent 60%),
          #0a0a0a
        `,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 50,
        }}
      >
        {/* Logo */}
        <div
          style={{
            transform: `scale(${logoScale})`,
            opacity: logoOpacity,
            filter: `drop-shadow(0 0 ${60 * glowPulse}px rgba(0, 102, 255, ${glowPulse}))`,
          }}
        >
          <Img
            src={staticFile("images/ds-logo.png")}
            style={{
              width: 150,
              height: 150,
              objectFit: "contain",
            }}
          />
        </div>

        {/* Title */}
        <h2
          style={{
            fontSize: 56,
            fontWeight: 600,
            color: "#ffffff",
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            textAlign: "center",
          }}
        >
          Start Building Today
        </h2>

        {/* Links */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 24,
            alignItems: "center",
          }}
        >
          {/* GitHub Link */}
          <div
            style={{
              opacity: link1Opacity,
              display: "flex",
              alignItems: "center",
              gap: 16,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 12,
              padding: "16px 32px",
            }}
          >
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="#fff"
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            <div>
              <div style={{ color: "#888", fontSize: 14 }}>Get the skill</div>
              <div
                style={{
                  color: "#fff",
                  fontSize: 22,
                  fontFamily: "JetBrains Mono, monospace",
                }}
              >
                github.com/digitalsamba/digital-samba-skill
              </div>
            </div>
          </div>

          {/* Website Link */}
          <div
            style={{
              opacity: link2Opacity,
              display: "flex",
              alignItems: "center",
              gap: 16,
              background: "linear-gradient(135deg, rgba(0, 102, 255, 0.15), rgba(0, 212, 170, 0.1))",
              border: "1px solid rgba(0, 102, 255, 0.3)",
              borderRadius: 12,
              padding: "16px 32px",
            }}
          >
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 6,
                background: "#0066FF",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 16,
                color: "#fff",
              }}
            >
              DS
            </div>
            <div>
              <div style={{ color: "#888", fontSize: 14 }}>
                Learn more
              </div>
              <div
                style={{
                  color: "#fff",
                  fontSize: 22,
                  fontFamily: "JetBrains Mono, monospace",
                }}
              >
                digitalsamba.com
              </div>
            </div>
          </div>
        </div>

        {/* Tagline */}
        <p
          style={{
            fontSize: 24,
            color: "#555",
            margin: 0,
            marginTop: 20,
            opacity: interpolate(frame, [100, 120], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          From zero to video calls in minutes
        </p>
      </div>
    </AbsoluteFill>
  );
};
