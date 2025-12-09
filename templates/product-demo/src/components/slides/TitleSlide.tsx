import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Img,
  staticFile,
} from 'remotion';
import { useTheme } from '../../config/theme';
import type { TitleContent } from '../../config/types';

interface TitleSlideProps {
  content: TitleContent;
}

export const TitleSlide: React.FC<TitleSlideProps> = ({ content }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = useTheme();

  // Logo animations
  const logoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const logoOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Title animation (staggered)
  const titleOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const titleY = interpolate(frame, [20, 45], [30, 0], {
    extrapolateRight: 'clamp',
  });

  // Subtitle animation
  const subtitleOpacity = interpolate(frame, [45, 65], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const subtitleY = interpolate(frame, [45, 70], [20, 0], {
    extrapolateRight: 'clamp',
  });

  // Accent line animation
  const lineWidth = interpolate(frame, [35, 60], [0, 400], {
    extrapolateRight: 'clamp',
  });

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
          gap: 40,
        }}
      >
        {/* Logos */}
        {content.logos && content.logos.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 60,
              opacity: logoOpacity,
              transform: `scale(${logoScale})`,
            }}
          >
            {content.logos.map((logo, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 16,
                }}
              >
                <Img
                  src={staticFile(logo.src)}
                  style={{
                    width: 80,
                    height: 80,
                    objectFit: 'contain',
                  }}
                />
                {logo.label && (
                  <span
                    style={{
                      fontSize: 36,
                      fontWeight: 600,
                      color: theme.colors.primary,
                    }}
                  >
                    {logo.label}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Main Title */}
        <h1
          style={{
            fontSize: theme.typography.h1.size,
            fontWeight: theme.typography.h1.weight,
            fontFamily: theme.fonts.primary,
            color: theme.colors.textDark,
            margin: 0,
            textAlign: 'center',
            opacity: titleOpacity,
            transform: `translateY(${titleY}px)`,
            letterSpacing: '-1px',
          }}
        >
          {content.headline}
        </h1>

        {/* Accent Line */}
        <div
          style={{
            width: lineWidth,
            height: 3,
            background: `linear-gradient(90deg, ${theme.colors.primary}, ${theme.colors.accent || theme.colors.primaryLight})`,
            borderRadius: 2,
          }}
        />

        {/* Subtitle */}
        {content.subheadline && (
          <p
            style={{
              fontSize: theme.typography.body.size + 4,
              fontFamily: theme.fonts.primary,
              color: theme.colors.textMedium,
              margin: 0,
              opacity: subtitleOpacity,
              transform: `translateY(${subtitleY}px)`,
            }}
          >
            {content.subheadline}
          </p>
        )}
      </div>
    </AbsoluteFill>
  );
};
