import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from 'remotion';
import { useTheme } from '../../config/theme';
import { sprintConfig } from '../../config/sprint-config';

export const EndCredits: React.FC = () => {
  const frame = useCurrentFrame();
  const theme = useTheme();
  const { credits, info } = sprintConfig;

  // Stagger each credit section
  const sectionDuration = 50; // frames per section
  const fadeInDuration = 20;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: theme.colors.bgDark,
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: theme.fonts.primary,
      }}
    >
      {/* Subtle gradient overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(ellipse at center, ${hexToRgba(theme.colors.primary, 0.1)} 0%, transparent 70%)`,
        }}
      />

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 40,
          padding: 60,
        }}
      >
        {/* Title */}
        {(() => {
          const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
            extrapolateRight: 'clamp',
          });
          return (
            <h1
              style={{
                fontSize: 48,
                fontWeight: 600,
                color: theme.colors.primary,
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
            { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
          );
          const translateY = interpolate(
            frame,
            [sectionStart, sectionStart + fadeInDuration],
            [20, 0],
            {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
              easing: Easing.out(Easing.cubic),
            }
          );

          return (
            <div
              key={sectionIndex}
              style={{
                opacity,
                transform: `translateY(${translateY}px)`,
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  fontSize: 24,
                  fontWeight: 500,
                  color: theme.colors.primary,
                  marginBottom: 8,
                  textTransform: 'uppercase',
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
                    color: '#ffffff',
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

      {/* Product name at bottom */}
      {(() => {
        const logoOpacity = interpolate(frame, [280, 310], [0, 1], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        });
        return (
          <div
            style={{
              position: 'absolute',
              bottom: 60,
              opacity: logoOpacity,
              fontSize: 28,
              fontWeight: 600,
              color: theme.colors.primary,
              letterSpacing: 4,
            }}
          >
            {info.product.toUpperCase()}
          </div>
        );
      })()}
    </AbsoluteFill>
  );
};

// Helper to convert hex to rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
