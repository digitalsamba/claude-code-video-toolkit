import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  OffthreadVideo,
  staticFile,
} from 'remotion';
import { useTheme } from '../../config/theme';
import type { DemoContent } from '../../config/types';

interface DemoSlideProps {
  content: DemoContent;
}

export const DemoSlide: React.FC<DemoSlideProps> = ({ content }) => {
  const frame = useCurrentFrame();
  const theme = useTheme();

  // Video container animation
  const containerOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const containerScale = interpolate(frame, [0, 25], [0.95, 1], {
    extrapolateRight: 'clamp',
  });

  // Label animation
  const labelOpacity = interpolate(frame, [15, 35], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 30,
          width: '100%',
        }}
      >
        {/* Label */}
        {content.label && (
          <div
            style={{
              opacity: labelOpacity,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <div
              style={{
                width: 4,
                height: 24,
                borderRadius: 2,
                background: theme.colors.primary,
              }}
            />
            <span
              style={{
                fontSize: theme.typography.h3.size,
                fontWeight: theme.typography.h3.weight,
                fontFamily: theme.fonts.primary,
                color: theme.colors.textDark,
              }}
            >
              {content.label}
            </span>
          </div>
        )}

        {/* Video Container */}
        <div
          style={{
            opacity: containerOpacity,
            transform: `scale(${containerScale})`,
            borderRadius: 16,
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
            border: '1px solid rgba(255,255,255,0.1)',
            maxWidth: 1600,
            width: '100%',
          }}
        >
          {/* Browser/Terminal chrome */}
          {(content.type === 'browser' || content.type === 'terminal') && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '12px 16px',
                background: content.type === 'terminal' ? '#1a1a1a' : '#2d2d2d',
                borderBottom: '1px solid #333',
              }}
            >
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#FF5F56' }} />
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#FFBD2E' }} />
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#27C93F' }} />
              {content.type === 'browser' && (
                <div
                  style={{
                    marginLeft: 20,
                    flex: 1,
                    background: '#1a1a1a',
                    borderRadius: 6,
                    padding: '6px 12px',
                    fontSize: 12,
                    color: '#666',
                    fontFamily: theme.fonts.mono,
                  }}
                >
                  localhost:3000
                </div>
              )}
            </div>
          )}

          {/* Video */}
          {content.videoFile && (
            <OffthreadVideo
              src={staticFile(content.videoFile)}
              style={{
                width: '100%',
                display: 'block',
              }}
            />
          )}
        </div>

        {/* Caption */}
        {content.caption && (
          <p
            style={{
              opacity: labelOpacity,
              fontSize: theme.typography.body.size - 4,
              fontFamily: theme.fonts.primary,
              color: theme.colors.textMedium,
              margin: 0,
            }}
          >
            {content.caption}
          </p>
        )}
      </div>
    </AbsoluteFill>
  );
};
