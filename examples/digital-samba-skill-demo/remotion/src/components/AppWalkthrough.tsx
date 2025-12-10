import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  OffthreadVideo,
  staticFile,
} from "remotion";

// Flow steps that appear alongside the video
const FLOW_STEPS = [
  { time: 0, label: "Home Page", desc: "Dual-card UI for role selection" },
  { time: 200, label: "Create Room", desc: "Simple form to start interview" },
  { time: 400, label: "Room Code", desc: "Shareable code for candidates" },
  { time: 600, label: "Join Flow", desc: "Candidate enters room code" },
  { time: 900, label: "Video Call", desc: "Live connection established" },
];

export const AppWalkthrough: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Header animation
  const headerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Video container animation
  const videoScale = spring({
    frame: Math.max(0, frame - 15),
    fps,
    config: { damping: 12, stiffness: 80 },
  });

  const videoOpacity = interpolate(frame, [15, 35], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Determine current step based on frame
  const currentStepIndex = FLOW_STEPS.findIndex(
    (step, i) =>
      frame >= step.time &&
      (i === FLOW_STEPS.length - 1 || frame < FLOW_STEPS[i + 1].time)
  );

  return (
    <AbsoluteFill
      style={{
        padding: 50,
      }}
    >
      {/* Header */}
      <div
        style={{
          opacity: headerOpacity,
          textAlign: "center",
          marginBottom: 30,
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
          The Interview Room App
        </h2>
        <p
          style={{
            fontSize: 22,
            color: "#666",
            margin: "12px 0 0 0",
          }}
        >
          Built entirely by Claude using the Digital Samba skill
        </p>
      </div>

      <div
        style={{
          display: "flex",
          gap: 40,
          height: "calc(100% - 130px)",
          alignItems: "center",
        }}
      >
        {/* Video Container */}
        <div
          style={{
            flex: 1,
            opacity: videoOpacity,
            transform: `scale(${videoScale})`,
            borderRadius: 16,
            overflow: "hidden",
            border: "2px solid #333",
            boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
            position: "relative",
            maxHeight: 700,
          }}
        >
          {/* Browser chrome */}
          <div
            style={{
              background: "#2a2a2a",
              padding: "10px 16px",
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
                flex: 1,
                background: "#1a1a1a",
                borderRadius: 6,
                padding: "6px 16px",
                color: "#888",
                fontSize: 14,
              }}
            >
              localhost:3000
            </div>
          </div>

          {/* Video */}
          <OffthreadVideo
            src={staticFile("demos/app-walkthrough.mp4")}
            style={{
              width: "100%",
              height: "auto",
              display: "block",
            }}
          />
        </div>

        {/* Flow Steps Sidebar */}
        <div
          style={{
            width: 340,
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          {FLOW_STEPS.map((step, i) => {
            const isActive = i === currentStepIndex;
            const isPast = i < currentStepIndex;
            const stepOpacity = interpolate(
              frame,
              [step.time, step.time + 30],
              [0.3, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={i}
                style={{
                  opacity: isPast ? 0.5 : stepOpacity,
                  background: isActive
                    ? "linear-gradient(135deg, rgba(0, 102, 255, 0.2), rgba(0, 212, 170, 0.1))"
                    : "rgba(255,255,255,0.03)",
                  border: isActive
                    ? "1px solid rgba(0, 102, 255, 0.5)"
                    : "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                  padding: 20,
                  transition: "all 0.3s",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    marginBottom: 8,
                  }}
                >
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: "50%",
                      background: isActive ? "#0066FF" : isPast ? "#00D4AA" : "#333",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#fff",
                    }}
                  >
                    {isPast ? "✓" : i + 1}
                  </div>
                  <span
                    style={{
                      color: isActive ? "#fff" : "#888",
                      fontSize: 18,
                      fontWeight: 600,
                    }}
                  >
                    {step.label}
                  </span>
                </div>
                <p
                  style={{
                    color: isActive ? "#aaa" : "#555",
                    fontSize: 14,
                    margin: 0,
                    paddingLeft: 44,
                  }}
                >
                  {step.desc}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
