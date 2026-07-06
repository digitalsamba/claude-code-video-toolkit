import React from "react";
import { Audio, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { interpolate, spring } from "remotion";
import { brand, font } from "../lib/brand";
import type { StatBeat as StatBeatProps } from "../lib/types";

function parseValue(value: string): { prefix: string; num: number; suffix: string } {
  // e.g. "+4.2% EV" → { prefix: "+", num: 4.2, suffix: "% EV" }
  // e.g. "87%" → { prefix: "", num: 87, suffix: "%" }
  const match = value.match(/^([^0-9]*)([0-9]+(?:\.[0-9]+)?)(.*)$/);
  if (!match) return { prefix: "", num: 0, suffix: value };
  return { prefix: match[1], num: parseFloat(match[2]), suffix: match[3] };
}

export const StatBeat: React.FC<{ beat: StatBeatProps }> = ({ beat }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const { prefix, num, suffix } = parseValue(beat.value);

  // Number counts up over the first 60 frames with easing
  const COUNT_DURATION = Math.min(60, durationInFrames - 20);
  const countProgress = interpolate(frame, [8, COUNT_DURATION], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: (t) => 1 - Math.pow(1 - t, 3), // ease-out cubic
  });
  const currentNum = num * countProgress;
  const displayNum = num % 1 === 0 ? Math.round(currentNum).toString() : currentNum.toFixed(1);

  // Pop animation when number reaches its target
  const popProgress = spring({ frame: frame - COUNT_DURATION, fps, config: { damping: 8, stiffness: 180 }, durationInFrames: 20 });
  const scale = 1 + popProgress * 0.06;

  const contextOpacity = interpolate(frame, [COUNT_DURATION - 5, COUNT_DURATION + 15], [0, 1], {
    extrapolateRight: "clamp",
  });

  const fadeOutStart = durationInFrames - 12;
  const globalOpacity = interpolate(frame, [fadeOutStart, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Entry slide-up
  const entryProgress = spring({ frame, fps, config: { damping: 20, stiffness: 100 }, durationInFrames: 20 });

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
        fontFamily: font,
      }}
    >
      {/* Background glow */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(ellipse 60% 60% at 50% 50%, rgba(52,211,153,0.08) 0%, transparent 70%)",
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
          fontSize: 28,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: brand.white,
          opacity: 0.8,
        }}
      >
        Arb<span style={{ color: brand.blue400 }}>Edge</span>
      </div>

      {/* Main stat */}
      <div
        style={{
          transform: `translateY(${(1 - entryProgress) * 60}px) scale(${scale})`,
          textAlign: "center",
        }}
      >
        {/* Prefix (e.g. "+") */}
        {prefix && (
          <div
            style={{
              fontSize: 100,
              fontWeight: 900,
              color: brand.green,
              lineHeight: 1,
              letterSpacing: "-0.04em",
              display: "inline",
            }}
          >
            {prefix}
          </div>
        )}
        {/* Number */}
        <div
          style={{
            fontSize: 200,
            fontWeight: 900,
            color: brand.green,
            lineHeight: 0.85,
            letterSpacing: "-0.06em",
            display: "inline",
            textShadow: `0 0 80px rgba(52,211,153,0.5)`,
          }}
        >
          {displayNum}
        </div>

        {/* Suffix — split so "%" is same size, rest smaller */}
        {suffix && (
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "center", gap: 8, marginTop: 8 }}>
            <span
              style={{
                fontSize: 80,
                fontWeight: 900,
                color: brand.green,
                letterSpacing: "-0.03em",
              }}
            >
              {suffix.trim()}
            </span>
          </div>
        )}
      </div>

      {/* Context line */}
      <div
        style={{
          marginTop: 48,
          fontSize: 34,
          fontWeight: 400,
          color: brand.gray400,
          textAlign: "center",
          padding: "0 64px",
          opacity: contextOpacity,
          letterSpacing: "-0.01em",
        }}
      >
        {beat.context}
      </div>

      <Audio src={staticFile(`audio/beats/${beat.id}.mp3`)} />
    </div>
  );
};
