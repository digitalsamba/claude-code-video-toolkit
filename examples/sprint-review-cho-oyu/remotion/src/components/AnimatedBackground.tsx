import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

interface AnimatedBackgroundProps {
  variant?: "subtle" | "tech" | "warm";
}

export const AnimatedBackground: React.FC<AnimatedBackgroundProps> = ({
  variant = "subtle",
}) => {
  const frame = useCurrentFrame();

  const colors = {
    subtle: {
      bg: "#ffffff",
      shape1: "rgba(234, 88, 12, 0.03)",
      shape2: "rgba(30, 41, 59, 0.02)",
      shape3: "rgba(234, 88, 12, 0.02)",
    },
    tech: {
      bg: "#f8fafc",
      shape1: "rgba(234, 88, 12, 0.04)",
      shape2: "rgba(100, 116, 139, 0.03)",
      shape3: "rgba(34, 197, 94, 0.02)",
    },
    warm: {
      bg: "#fffbf7",
      shape1: "rgba(234, 88, 12, 0.05)",
      shape2: "rgba(234, 88, 12, 0.03)",
      shape3: "rgba(251, 146, 60, 0.03)",
    },
  };

  const c = colors[variant];

  // Slow, organic movements
  const rotation1 = interpolate(frame, [0, 900], [0, 360], {
    extrapolateRight: "extend",
  });
  const rotation2 = interpolate(frame, [0, 1200], [360, 0], {
    extrapolateRight: "extend",
  });

  const float1Y = Math.sin(frame * 0.008) * 30;
  const float2Y = Math.cos(frame * 0.006) * 40;
  const float3X = Math.sin(frame * 0.005) * 50;

  const scale1 = 1 + Math.sin(frame * 0.004) * 0.1;
  const scale2 = 1 + Math.cos(frame * 0.003) * 0.08;

  return (
    <AbsoluteFill style={{ backgroundColor: c.bg, overflow: "hidden" }}>
      {/* Large slow-moving circle top-right */}
      <div
        style={{
          position: "absolute",
          top: -200 + float1Y,
          right: -150,
          width: 600,
          height: 600,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${c.shape1} 0%, transparent 70%)`,
          transform: `rotate(${rotation1}deg) scale(${scale1})`,
        }}
      />

      {/* Medium shape bottom-left */}
      <div
        style={{
          position: "absolute",
          bottom: -100 + float2Y,
          left: -100 + float3X,
          width: 500,
          height: 500,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${c.shape2} 0%, transparent 70%)`,
          transform: `rotate(${rotation2}deg) scale(${scale2})`,
        }}
      />

      {/* Subtle accent shape center-left */}
      <div
        style={{
          position: "absolute",
          top: "40%",
          left: -200 + float3X * 0.5,
          width: 400,
          height: 400,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${c.shape3} 0%, transparent 60%)`,
          transform: `scale(${scale1 * 0.9})`,
        }}
      />

      {/* Fine grid pattern overlay for tech feel */}
      {variant === "tech" && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage: `
              linear-gradient(rgba(100, 116, 139, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(100, 116, 139, 0.03) 1px, transparent 1px)
            `,
            backgroundSize: "60px 60px",
          }}
        />
      )}
    </AbsoluteFill>
  );
};
