import React from "react";
import { Audio, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { interpolate, spring } from "remotion";
import { brand, font } from "../lib/brand";
import type { CtaBeat as CtaBeatProps } from "../lib/types";

export const CtaBeat: React.FC<{ beat: CtaBeatProps }> = ({ beat }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const logoProgress = spring({ frame, fps, config: { damping: 18, stiffness: 100 }, durationInFrames: 20 });
  const urlProgress = spring({ frame: frame - 18, fps, config: { damping: 20, stiffness: 110 }, durationInFrames: 20 });
  const priceProgress = spring({ frame: frame - 32, fps, config: { damping: 20, stiffness: 110 }, durationInFrames: 20 });
  const ctaProgress = spring({ frame: frame - 46, fps, config: { damping: 16, stiffness: 90 }, durationInFrames: 25 });
  const finePrintProgress = spring({ frame: frame - 62, fps, config: { damping: 22, stiffness: 120 }, durationInFrames: 20 });

  const fadeOutStart = durationInFrames - 15;
  const globalOpacity = interpolate(frame, [fadeOutStart, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Button glow pulse
  const glowPulse = 0.25 + 0.12 * Math.sin(frame * 0.08);

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
      {/* Radial glow — blue */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(ellipse 80% 60% at 50% 50%, rgba(29,78,216,${glowPulse}) 0%, transparent 70%)`,
          pointerEvents: "none",
        }}
      />

      {/* Logo */}
      <div
        style={{
          fontSize: 96,
          fontWeight: 900,
          letterSpacing: "-0.04em",
          color: brand.white,
          lineHeight: 1,
          opacity: logoProgress,
          transform: `scale(${0.8 + logoProgress * 0.2})`,
          textAlign: "center",
        }}
      >
        Arb<span style={{ color: brand.blue400 }}>Edge</span>
      </div>

      {/* URL */}
      <div
        style={{
          marginTop: 20,
          fontSize: 40,
          fontWeight: 500,
          color: brand.gray400,
          letterSpacing: "-0.01em",
          opacity: urlProgress,
          transform: `translateY(${(1 - urlProgress) * 16}px)`,
        }}
      >
        arbedge.au
      </div>

      {/* Price */}
      <div
        style={{
          marginTop: 72,
          textAlign: "center",
          opacity: priceProgress,
          transform: `translateY(${(1 - priceProgress) * 20}px)`,
        }}
      >
        <div
          style={{
            fontSize: 88,
            fontWeight: 900,
            color: brand.white,
            letterSpacing: "-0.04em",
            lineHeight: 1,
          }}
        >
          {beat.price}
        </div>
        <div
          style={{
            fontSize: 28,
            fontWeight: 400,
            color: brand.gray500,
            marginTop: 8,
          }}
        >
          Australian dollars
        </div>
      </div>

      {/* CTA button */}
      <div
        style={{
          marginTop: 60,
          opacity: ctaProgress,
          transform: `translateY(${(1 - ctaProgress) * 24}px) scale(${0.92 + ctaProgress * 0.08})`,
        }}
      >
        <div
          style={{
            background: brand.blue600,
            color: brand.white,
            fontSize: 40,
            fontWeight: 700,
            letterSpacing: "-0.01em",
            padding: "22px 60px",
            borderRadius: 16,
            boxShadow: `0 0 60px rgba(37,99,235,${0.45 + 0.15 * Math.sin(frame * 0.08)})`,
            textAlign: "center",
          }}
        >
          Start free trial
        </div>
      </div>

      {/* Fine print */}
      <div
        style={{
          marginTop: 24,
          fontSize: 24,
          fontWeight: 400,
          color: brand.gray600,
          textAlign: "center",
          opacity: finePrintProgress,
          letterSpacing: "-0.005em",
        }}
      >
        7-day free trial · Cancel anytime
      </div>

      <Audio src={staticFile(`audio/beats/${beat.id}.mp3`)} />
    </div>
  );
};
