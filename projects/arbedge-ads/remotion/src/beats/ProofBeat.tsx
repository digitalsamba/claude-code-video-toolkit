import React from "react";
import { Audio, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { interpolate, spring } from "remotion";
import { brand, font } from "../lib/brand";
import type { ProofBeat as ProofBeatProps, Opportunity } from "../lib/types";

const SPORT_COLORS: Record<string, string> = {
  AFL: "#f97316",
  NRL: "#10b981",
  NBA: "#8b5cf6",
  NBL: "#06b6d4",
  EPL: "#6366f1",
  Soccer: "#6366f1",
  MMA: "#ef4444",
  Cricket: "#eab308",
  Tennis: "#84cc16",
};

const OpportunityCard: React.FC<{ opp: Opportunity; delay: number }> = ({
  opp,
  delay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - delay,
    fps,
    config: { damping: 20, stiffness: 120 },
    durationInFrames: 25,
  });

  const sportColor = SPORT_COLORS[opp.sport] ?? brand.blue400;

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: `1px solid ${brand.border}`,
        borderRadius: 16,
        padding: "24px 28px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        opacity: progress,
        transform: `translateY(${(1 - progress) * 40}px)`,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Left accent */}
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 4,
          background: brand.green,
          borderRadius: "0 2px 2px 0",
        }}
      />

      {/* Left: sport + event + bookmaker */}
      <div style={{ paddingLeft: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span
            style={{
              background: sportColor,
              color: "#000",
              fontSize: 18,
              fontWeight: 800,
              letterSpacing: "0.05em",
              padding: "2px 10px",
              borderRadius: 4,
            }}
          >
            {opp.sport}
          </span>
          <span style={{ fontSize: 28, fontWeight: 700, color: brand.white, letterSpacing: "-0.01em" }}>
            {opp.event}
          </span>
        </div>
        <div style={{ fontSize: 24, color: brand.gray400, fontWeight: 500 }}>
          {opp.bookmaker} · {opp.odds.toFixed(2)}
        </div>
      </div>

      {/* Right: EV badge */}
      <div
        style={{
          background: brand.green,
          color: "#000",
          fontWeight: 900,
          fontSize: 30,
          letterSpacing: "-0.01em",
          padding: "10px 20px",
          borderRadius: 10,
          whiteSpace: "nowrap",
          boxShadow: "0 0 20px rgba(52,211,153,0.35)",
        }}
      >
        +{opp.ev_percent}% EV
      </div>
    </div>
  );
};

export const ProofBeat: React.FC<{ beat: ProofBeatProps }> = ({ beat }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const HEADER_DELAY = 0;
  const CARD_START = 20;
  const CARD_STAGGER = 22;

  const headerOpacity = interpolate(frame, [HEADER_DELAY, HEADER_DELAY + 15], [0, 1], {
    extrapolateRight: "clamp",
  });

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
        padding: "120px 52px 80px",
        position: "relative",
        overflow: "hidden",
        opacity: globalOpacity,
        fontFamily: font,
      }}
    >
      {/* Top gradient */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(ellipse 80% 30% at 50% 0%, rgba(29,78,216,0.12) 0%, transparent 60%)",
          pointerEvents: "none",
        }}
      />

      {/* Logo */}
      <div
        style={{
          position: "absolute",
          top: 60,
          left: 52,
          fontSize: 26,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: brand.white,
          opacity: 0.85,
        }}
      >
        Arb<span style={{ color: brand.blue400 }}>Edge</span>
      </div>

      {/* Live indicator */}
      <div
        style={{
          position: "absolute",
          top: 60,
          right: 52,
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "rgba(52,211,153,0.1)",
          border: "1px solid rgba(52,211,153,0.2)",
          borderRadius: 20,
          padding: "6px 14px",
          opacity: headerOpacity,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: brand.green,
            // Pulse via opacity oscillation based on frame
            opacity: 0.5 + 0.5 * Math.sin(frame * 0.15),
          }}
        />
        <span style={{ fontSize: 20, fontWeight: 600, color: brand.green }}>LIVE</span>
      </div>

      {/* Header */}
      <div
        style={{
          marginBottom: 36,
          opacity: headerOpacity,
        }}
      >
        <div
          style={{
            fontSize: 52,
            fontWeight: 800,
            color: brand.white,
            letterSpacing: "-0.025em",
            lineHeight: 1.1,
          }}
        >
          Recent Opportunities
        </div>
        <div style={{ fontSize: 28, color: brand.gray500, marginTop: 8 }}>
          Flagged in the last 60 minutes
        </div>
      </div>

      {/* Cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {beat.opportunities.map((opp, i) => (
          <OpportunityCard
            key={i}
            opp={opp}
            delay={CARD_START + i * CARD_STAGGER}
          />
        ))}
      </div>

      <Audio src={staticFile(`audio/beats/${beat.id}.mp3`)} />
    </div>
  );
};
