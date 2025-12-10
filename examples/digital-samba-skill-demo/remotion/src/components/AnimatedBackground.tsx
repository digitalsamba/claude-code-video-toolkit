import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

// Floating shapes for subtle background animation
export const AnimatedBackground: React.FC = () => {
  const frame = useCurrentFrame();

  // Generate floating shapes
  const shapes = Array.from({ length: 8 }, (_, i) => {
    const baseX = (i * 250 + 100) % 1920;
    const baseY = (i * 180 + 50) % 1080;
    const size = 100 + (i % 3) * 50;
    const speed = 0.3 + (i % 4) * 0.1;
    const phase = i * 0.8;

    const x = baseX + Math.sin((frame * speed * 0.02) + phase) * 30;
    const y = baseY + Math.cos((frame * speed * 0.015) + phase) * 20;
    const rotation = frame * speed * 0.5 + i * 45;
    const opacity = 0.03 + (i % 3) * 0.01;

    return { x, y, size, rotation, opacity, isCircle: i % 2 === 0 };
  });

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      {/* Gradient overlay */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          background: `
            radial-gradient(ellipse at 20% 30%, rgba(0, 102, 255, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 70%, rgba(0, 212, 170, 0.08) 0%, transparent 50%)
          `,
        }}
      />

      {/* Floating shapes */}
      {shapes.map((shape, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: shape.x,
            top: shape.y,
            width: shape.size,
            height: shape.size,
            borderRadius: shape.isCircle ? "50%" : "20%",
            border: `1px solid ${i % 2 === 0 ? "#0066FF" : "#00D4AA"}`,
            opacity: shape.opacity,
            transform: `rotate(${shape.rotation}deg)`,
          }}
        />
      ))}

      {/* Grid lines */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
          `,
          backgroundSize: "100px 100px",
        }}
      />
    </AbsoluteFill>
  );
};
