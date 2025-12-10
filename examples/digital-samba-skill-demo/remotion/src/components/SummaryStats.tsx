import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

const STATS = [
  {
    value: "5",
    unit: "min",
    label: "To Working App",
    icon: "⚡",
    color: "#0066FF",
  },
  {
    value: "0",
    unit: "",
    label: "Infrastructure Setup",
    icon: "☁️",
    color: "#00D4AA",
  },
  {
    value: "100%",
    unit: "",
    label: "SDK Control",
    icon: "🎛️",
    color: "#D97706",
  },
];

export const SummaryStats: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Header animation
  const headerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  const headerY = interpolate(frame, [0, 25], [20, 0], {
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
        {/* Header */}
        <h2
          style={{
            fontSize: 48,
            fontWeight: 600,
            color: "#ffffff",
            margin: 0,
            opacity: headerOpacity,
            transform: `translateY(${headerY}px)`,
          }}
        >
          What You Get
        </h2>

        {/* Stats Cards */}
        <div
          style={{
            display: "flex",
            gap: 40,
          }}
        >
          {STATS.map((stat, i) => {
            const delay = 30 + i * 20;
            const cardScale = spring({
              frame: Math.max(0, frame - delay),
              fps,
              config: { damping: 10, stiffness: 100, mass: 0.8 },
            });

            const cardOpacity = interpolate(
              frame,
              [delay, delay + 15],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={i}
                style={{
                  opacity: cardOpacity,
                  transform: `scale(${cardScale})`,
                  background: `linear-gradient(135deg, ${stat.color}15, ${stat.color}05)`,
                  border: `1px solid ${stat.color}40`,
                  borderRadius: 24,
                  padding: "50px 60px",
                  textAlign: "center",
                  minWidth: 280,
                }}
              >
                {/* Icon */}
                <div
                  style={{
                    fontSize: 48,
                    marginBottom: 20,
                  }}
                >
                  {stat.icon}
                </div>

                {/* Value */}
                <div
                  style={{
                    fontSize: 72,
                    fontWeight: 700,
                    color: stat.color,
                    lineHeight: 1,
                  }}
                >
                  {stat.value}
                  <span
                    style={{
                      fontSize: 32,
                      fontWeight: 400,
                      marginLeft: 4,
                    }}
                  >
                    {stat.unit}
                  </span>
                </div>

                {/* Label */}
                <div
                  style={{
                    fontSize: 20,
                    color: "#888",
                    marginTop: 16,
                    fontWeight: 500,
                  }}
                >
                  {stat.label}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
