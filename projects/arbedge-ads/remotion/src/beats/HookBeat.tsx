import React from "react";
import { Audio, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { interpolate, spring } from "remotion";
import { brand, font } from "../lib/brand";
import type { HookBeat as HookBeatProps } from "../lib/types";

export const HookBeat: React.FC<{ beat: HookBeatProps }> = ({ beat }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const lines = beat.headline.split("\n");
  const FRAMES_PER_LINE = 10;

  const logoOpacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: "clamp",
  });

  const sublineDelay = lines.length * FRAMES_PER_LINE + 8;
  const sublineOpacity = interpolate(
    frame,
    [sublineDelay, sublineDelay + 18],
    [0, 1],
    { extrapolateRight: "clamp" }
  );

  // Fade out near end
  const fadeOutStart = durationInFrames - 12;
  const globalOpacity = interpolate(frame, [fadeOutStart, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: brand.bg,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
        opacity: globalOpacity,
      }}
    >
      {/* Radial blue glow from top-center */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(29,78,216,0.22) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />

      {/* Logo */}
      <div
        style={{
          position: "absolute",
          top: 72,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: font,
          fontSize: 32,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: brand.white,
          opacity: logoOpacity,
        }}
      >
        Arb<span style={{ color: brand.blue400 }}>Edge</span>
      </div>

      {/* Headline — line by line */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 4,
          padding: "0 56px",
          textAlign: "center",
        }}
      >
        {lines.map((line, i) => {
          const delay = i * FRAMES_PER_LINE;
          const progress = spring({
            frame: frame - delay,
            fps,
            config: { damping: 16, stiffness: 100, mass: 0.9 },
            durationInFrames: 30,
          });

          // Alternate emphasis: short lines (one word) get extra-large treatment
          const isEmphasis = line.split(" ").length === 1;
          const fontSize = isEmphasis ? 144 : 100;

          return (
            <div
              key={i}
              style={{
                fontFamily: font,
                fontSize,
                fontWeight: 900,
                letterSpacing: "-0.045em",
                lineHeight: 0.9,
                color: brand.white,
                opacity: progress,
                transform: `translateY(${(1 - progress) * 48}px)`,
              }}
            >
              {line}
            </div>
          );
        })}
      </div>

      {/* Subline */}
      {beat.subline && (
        <div
          style={{
            marginTop: 36,
            fontFamily: font,
            fontSize: 34,
            fontWeight: 400,
            color: brand.blue400,
            letterSpacing: "-0.01em",
            opacity: sublineOpacity,
            textAlign: "center",
            padding: "0 56px",
          }}
        >
          {beat.subline}
        </div>
      )}

      {/* Audio */}
      <Audio src={staticFile(`audio/beats/${beat.id}.mp3`)} />
    </div>
  );
};
