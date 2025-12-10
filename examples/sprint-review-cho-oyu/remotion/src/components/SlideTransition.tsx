import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

interface SlideTransitionProps {
  children: React.ReactNode;
  durationInFrames: number;
  transitionDuration?: number; // frames for fade in/out
  style?: "fade" | "zoom" | "slide-up" | "blur-fade";
}

export const SlideTransition: React.FC<SlideTransitionProps> = ({
  children,
  durationInFrames,
  transitionDuration = 15,
  style = "zoom",
}) => {
  const frame = useCurrentFrame();

  // Fade in at start
  const fadeIn = interpolate(frame, [0, transitionDuration], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Fade out at end
  const fadeOut = interpolate(
    frame,
    [durationInFrames - transitionDuration, durationInFrames],
    [1, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  const opacity = Math.min(fadeIn, fadeOut);

  // Style-specific transforms
  let transform = "";
  let filter = "";

  if (style === "zoom") {
    const scaleIn = interpolate(frame, [0, transitionDuration], [0.97, 1], {
      extrapolateRight: "clamp",
    });
    const scaleOut = interpolate(
      frame,
      [durationInFrames - transitionDuration, durationInFrames],
      [1, 1.02],
      {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }
    );
    const scale = frame < durationInFrames / 2 ? scaleIn : scaleOut;
    transform = `scale(${scale})`;
  }

  if (style === "slide-up") {
    const yIn = interpolate(frame, [0, transitionDuration], [20, 0], {
      extrapolateRight: "clamp",
    });
    const yOut = interpolate(
      frame,
      [durationInFrames - transitionDuration, durationInFrames],
      [0, -20],
      {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }
    );
    const y = frame < durationInFrames / 2 ? yIn : yOut;
    transform = `translateY(${y}px)`;
  }

  if (style === "blur-fade") {
    const blurIn = interpolate(frame, [0, transitionDuration], [8, 0], {
      extrapolateRight: "clamp",
    });
    const blurOut = interpolate(
      frame,
      [durationInFrames - transitionDuration, durationInFrames],
      [0, 8],
      {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }
    );
    const blur = frame < durationInFrames / 2 ? blurIn : blurOut;
    filter = `blur(${blur}px)`;
  }

  return (
    <AbsoluteFill
      style={{
        opacity,
        transform: transform || undefined,
        filter: filter || undefined,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};
