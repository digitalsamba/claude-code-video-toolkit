import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from 'remotion';
import { useTheme } from '../../config/theme';
import type { SolutionContent } from '../../config/types';

interface SolutionSlideProps {
  content: SolutionContent;
}

export const SolutionSlide: React.FC<SolutionSlideProps> = ({ content }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = useTheme();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 25], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const titleY = interpolate(frame, [0, 30], [30, 0], {
    extrapolateRight: 'clamp',
  });

  // Description animation
  const descOpacity = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Highlights animation
  const highlightAnimations = (content.highlights || []).map((_, i) => {
    const startFrame = 50 + i * 20;
    return {
      opacity: interpolate(frame, [startFrame, startFrame + 15], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }),
      scale: spring({
        frame: Math.max(0, frame - startFrame),
        fps,
        config: { damping: 12, stiffness: 100 },
      }),
    };
  });

  // Pulse effect on question mark
  const pulse = 0.95 + Math.sin(frame * 0.1) * 0.05;

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 50,
          maxWidth: 1200,
          textAlign: 'center',
        }}
      >
        {/* Main headline */}
        <h2
          style={{
            fontSize: theme.typography.h2.size,
            fontWeight: theme.typography.h2.weight,
            fontFamily: theme.fonts.primary,
            color: theme.colors.textDark,
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleY}px) scale(${pulse})`,
          }}
        >
          {content.headline}
        </h2>

        {/* Description */}
        {content.description && (
          <p
            style={{
              fontSize: theme.typography.body.size + 4,
              fontFamily: theme.fonts.primary,
              color: theme.colors.textMedium,
              margin: 0,
              opacity: descOpacity,
            }}
          >
            {content.description}
          </p>
        )}

        {/* Highlights */}
        {content.highlights && content.highlights.length > 0 && (
          <div
            style={{
              display: 'flex',
              gap: 30,
              marginTop: 30,
            }}
          >
            {content.highlights.map((highlight, i) => (
              <div
                key={i}
                style={{
                  opacity: highlightAnimations[i].opacity,
                  transform: `scale(${highlightAnimations[i].scale})`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  background: `linear-gradient(135deg, ${theme.colors.primary}15, ${theme.colors.primary}05)`,
                  border: `1px solid ${theme.colors.primary}40`,
                  borderRadius: 12,
                  padding: '16px 24px',
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: theme.colors.primary,
                  }}
                />
                <span
                  style={{
                    color: theme.colors.textDark,
                    fontSize: 20,
                    fontWeight: 500,
                  }}
                >
                  {highlight}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};
