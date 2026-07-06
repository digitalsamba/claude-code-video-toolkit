import React from "react";
import { Audio, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { interpolate, spring } from "remotion";
import { brand, font } from "../lib/brand";
import type { OddsCompareBeat as OddsCompareBeatProps } from "../lib/types";

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

export const OddsCompareBeat: React.FC<{ beat: OddsCompareBeatProps }> = ({
  beat,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Phase timings
  const HEADER_IN = 0;
  const TABLE_HEADER_IN = 18;
  const ROW_START = 30;
  const ROW_STAGGER = 12;
  const PINNACLE_IN = ROW_START + beat.rows.length * ROW_STAGGER + 8;
  const EV_IN = PINNACLE_IN + 20;

  const headerProgress = spring({ frame: frame - HEADER_IN, fps, config: { damping: 20, stiffness: 120 }, durationInFrames: 20 });
  const tableHeaderOpacity = interpolate(frame, [TABLE_HEADER_IN, TABLE_HEADER_IN + 12], [0, 1], { extrapolateRight: "clamp" });
  const evProgress = spring({ frame: frame - EV_IN, fps, config: { damping: 14, stiffness: 80 }, durationInFrames: 25 });

  const fadeOutStart = durationInFrames - 12;
  const globalOpacity = interpolate(frame, [fadeOutStart, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const sportColor = SPORT_COLORS[beat.sport] ?? brand.blue400;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: brand.bg,
        display: "flex",
        flexDirection: "column",
        alignItems: "stretch",
        justifyContent: "center",
        padding: "0 52px",
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
            "radial-gradient(ellipse 70% 40% at 50% 5%, rgba(29,78,216,0.14) 0%, transparent 60%)",
          pointerEvents: "none",
        }}
      />

      {/* Logo */}
      <div
        style={{
          position: "absolute",
          top: 60,
          left: 52,
          fontFamily: font,
          fontSize: 26,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: brand.white,
          opacity: 0.85,
        }}
      >
        Arb<span style={{ color: brand.blue400 }}>Edge</span>
      </div>

      {/* Sport badge + event */}
      <div
        style={{
          opacity: headerProgress,
          transform: `translateY(${(1 - headerProgress) * -20}px)`,
          marginBottom: 40,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <div
            style={{
              background: sportColor,
              color: "#000",
              fontWeight: 800,
              fontSize: 22,
              letterSpacing: "0.05em",
              padding: "4px 14px",
              borderRadius: 6,
            }}
          >
            {beat.sport}
          </div>
          <div style={{ fontSize: 28, fontWeight: 600, color: brand.gray400 }}>
            {beat.team}
          </div>
        </div>
        <div style={{ fontSize: 44, fontWeight: 800, color: brand.white, letterSpacing: "-0.02em" }}>
          {beat.event}
        </div>
      </div>

      {/* Table */}
      <div style={{ borderRadius: 16, overflow: "hidden", border: `1px solid ${brand.border}` }}>
        {/* Table header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "16px 24px",
            background: "rgba(255,255,255,0.03)",
            borderBottom: `1px solid ${brand.border}`,
            opacity: tableHeaderOpacity,
          }}
        >
          <span style={{ fontSize: 22, fontWeight: 500, color: brand.gray500, letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Bookmaker
          </span>
          <span style={{ fontSize: 22, fontWeight: 500, color: brand.gray500, letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Odds
          </span>
        </div>

        {/* Bookmaker rows */}
        {beat.rows.map((row, i) => {
          const rowDelay = ROW_START + i * ROW_STAGGER;
          const rowProgress = spring({ frame: frame - rowDelay, fps, config: { damping: 22, stiffness: 130 }, durationInFrames: 20 });
          const isOutlier = row.is_outlier === true;

          return (
            <div
              key={row.bookmaker}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "20px 24px",
                borderBottom: `1px solid ${brand.border}`,
                background: isOutlier
                  ? `rgba(52, 211, 153, ${0.08 * rowProgress})`
                  : "transparent",
                opacity: rowProgress,
                transform: `translateX(${(1 - rowProgress) * -30}px)`,
                position: "relative",
              }}
            >
              {/* Outlier left accent bar */}
              {isOutlier && (
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: 4,
                    background: brand.green,
                    opacity: rowProgress,
                    borderRadius: "0 2px 2px 0",
                  }}
                />
              )}
              <div
                style={{
                  fontSize: 32,
                  fontWeight: isOutlier ? 700 : 500,
                  color: isOutlier ? brand.white : brand.gray400,
                }}
              >
                {row.bookmaker}
              </div>
              <div
                style={{
                  fontSize: 36,
                  fontWeight: 800,
                  color: isOutlier ? brand.green : brand.gray400,
                  letterSpacing: "-0.01em",
                }}
              >
                {row.odds.toFixed(2)}
              </div>
            </div>
          );
        })}

        {/* Pinnacle row */}
        {(() => {
          const pinnacleProgress = spring({ frame: frame - PINNACLE_IN, fps, config: { damping: 22, stiffness: 130 }, durationInFrames: 20 });
          return (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "20px 24px",
                background: "rgba(255,255,255,0.03)",
                opacity: pinnacleProgress,
                transform: `translateX(${(1 - pinnacleProgress) * -30}px)`,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ fontSize: 32, fontWeight: 600, color: brand.gray500 }}>
                  Pinnacle
                </div>
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 600,
                    color: brand.blue400,
                    background: "rgba(96,165,250,0.1)",
                    border: `1px solid rgba(96,165,250,0.2)`,
                    borderRadius: 4,
                    padding: "2px 8px",
                    letterSpacing: "0.04em",
                  }}
                >
                  SHARP
                </div>
              </div>
              <div style={{ fontSize: 36, fontWeight: 800, color: brand.gray500, letterSpacing: "-0.01em" }}>
                {beat.pinnacle_odds.toFixed(2)}
              </div>
            </div>
          );
        })()}
      </div>

      {/* EV badge */}
      <div
        style={{
          marginTop: 32,
          display: "flex",
          justifyContent: "flex-end",
          opacity: evProgress,
          transform: `translateX(${(1 - evProgress) * 40}px) scale(${0.85 + evProgress * 0.15})`,
        }}
      >
        <div
          style={{
            background: brand.green,
            color: "#000",
            fontFamily: font,
            fontSize: 42,
            fontWeight: 900,
            letterSpacing: "-0.02em",
            padding: "12px 28px",
            borderRadius: 12,
            boxShadow: "0 0 40px rgba(52,211,153,0.4)",
          }}
        >
          +{beat.ev_percent}% EV
        </div>
      </div>

      <Audio src={staticFile(`audio/beats/${beat.id}.mp3`)} />
    </div>
  );
};
