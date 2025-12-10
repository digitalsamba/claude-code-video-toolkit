import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const EndCredits: React.FC = () => {
  const frame = useCurrentFrame();

  const credits = [
    { category: "Made with", items: ["Claude Code", "JIRA MCP Server"] },
    {
      category: "Video Production",
      items: ["Remotion - Create real MP4 videos with React"],
    },
    {
      category: "Audio Generation",
      items: [
        "ElevenLabs API",
        "Voiceover • Sound Effects • Background Music",
      ],
    },
    {
      category: "Demo Recording Equipment",
      items: ["Joel's iPad", "A grubby finger"],
    },
    {
      category: "Special Thanks",
      items: ["Not enough patience to sync everything"],
    },
  ];

  // Stagger each credit section
  const sectionDuration = 50; // frames per section
  const fadeInDuration = 20;
  const fadeOutDuration = 15;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#1a1a2e",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      {/* Subtle gradient overlay */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(ellipse at center, rgba(255,107,53,0.1) 0%, transparent 70%)",
        }}
      />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 40,
          padding: 60,
        }}
      >
        {/* Title */}
        {(() => {
          const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
            extrapolateRight: "clamp",
          });
          return (
            <h1
              style={{
                fontSize: 48,
                fontWeight: 600,
                color: "#FF6B35",
                opacity: titleOpacity,
                marginBottom: 20,
                letterSpacing: 2,
              }}
            >
              CREDITS
            </h1>
          );
        })()}

        {/* Credit sections */}
        {credits.map((section, sectionIndex) => {
          const sectionStart = 30 + sectionIndex * sectionDuration;
          const opacity = interpolate(
            frame,
            [sectionStart, sectionStart + fadeInDuration],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const translateY = interpolate(
            frame,
            [sectionStart, sectionStart + fadeInDuration],
            [20, 0],
            {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            }
          );

          return (
            <div
              key={sectionIndex}
              style={{
                opacity,
                transform: `translateY(${translateY}px)`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontSize: 24,
                  fontWeight: 500,
                  color: "#FF6B35",
                  marginBottom: 8,
                  textTransform: "uppercase",
                  letterSpacing: 3,
                }}
              >
                {section.category}
              </div>
              {section.items.map((item, itemIndex) => (
                <div
                  key={itemIndex}
                  style={{
                    fontSize: 32,
                    fontWeight: 300,
                    color: "#ffffff",
                    lineHeight: 1.4,
                  }}
                >
                  {item}
                </div>
              ))}
            </div>
          );
        })}
      </div>

      {/* Digital Samba logo/text at bottom */}
      {(() => {
        const logoOpacity = interpolate(frame, [280, 310], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            style={{
              position: "absolute",
              bottom: 60,
              opacity: logoOpacity,
              fontSize: 28,
              fontWeight: 600,
              color: "#FF6B35",
              letterSpacing: 4,
            }}
          >
            DIGITAL SAMBA
          </div>
        );
      })()}
    </AbsoluteFill>
  );
};
