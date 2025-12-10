import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  OffthreadVideo,
  staticFile,
} from "remotion";

export const SkillInstallDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Header animation
  const headerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Video frame animation
  const videoScale = spring({
    frame: Math.max(0, frame - 15),
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const videoOpacity = interpolate(frame, [15, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Caption animation
  const captionOpacity = interpolate(frame, [45, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
        padding: 60,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 40,
          width: "100%",
        }}
      >
        {/* Header */}
        <div
          style={{
            opacity: headerOpacity,
            textAlign: "center",
          }}
        >
          <h2
            style={{
              fontSize: 42,
              fontWeight: 600,
              color: "#ffffff",
              margin: 0,
              marginBottom: 12,
            }}
          >
            Simple Installation
          </h2>
          <p
            style={{
              fontSize: 24,
              color: "#666",
              margin: 0,
            }}
          >
            Get the Digital Samba skill in seconds
          </p>
        </div>

        {/* Video Container */}
        <div
          style={{
            opacity: videoOpacity,
            transform: `scale(${videoScale})`,
            borderRadius: 16,
            overflow: "hidden",
            border: "2px solid #333",
            boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
            position: "relative",
          }}
        >
          {/* Browser-style header */}
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
            <div
              style={{
                background: "#1a1a1a",
                borderRadius: 6,
                padding: "6px 16px",
                color: "#666",
                fontSize: 14,
                fontFamily: "monospace",
              }}
            >
              Terminal — claude-code
            </div>
          </div>

          {/* Video - sped up to show more of the installation */}
          <OffthreadVideo
            src={staticFile("demos/skill-install.mp4")}
            style={{
              width: 1008,
              height: 626,
              display: "block",
            }}
            playbackRate={2.25}
          />
        </div>

        {/* Caption */}
        <div
          style={{
            opacity: captionOpacity,
            display: "flex",
            alignItems: "center",
            gap: 16,
            background: "rgba(0, 102, 255, 0.1)",
            border: "1px solid rgba(0, 102, 255, 0.3)",
            borderRadius: 12,
            padding: "16px 32px",
          }}
        >
          <div
            style={{
              fontSize: 28,
            }}
          >
            ⚡
          </div>
          <span
            style={{
              color: "#fff",
              fontSize: 20,
            }}
          >
            Full API documentation and examples included
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
