import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from 'remotion';
import { useTheme } from '../../config/theme';
import type { ProblemContent } from '../../config/types';

interface ProblemSlideProps {
  content: ProblemContent;
}

export const ProblemSlide: React.FC<ProblemSlideProps> = ({ content }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const theme = useTheme();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: 'clamp',
  });

  // Code block reveal
  const codeBlockOpacity = interpolate(frame, [15, 35], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const codeBlockX = interpolate(frame, [15, 45], [-50, 0], {
    extrapolateRight: 'clamp',
  });

  // Problem icons animation (right side)
  const iconAnimations = content.problems.map((_, i) => {
    const startFrame = 60 + i * 30;
    return {
      opacity: interpolate(frame, [startFrame, startFrame + 15], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }),
      scale: spring({
        frame: Math.max(0, frame - startFrame),
        fps,
        config: { damping: 10, stiffness: 100 },
      }),
    };
  });

  return (
    <AbsoluteFill style={{ padding: 80 }}>
      {/* Title */}
      <h2
        style={{
          fontSize: theme.typography.h2.size,
          fontWeight: theme.typography.h2.weight,
          fontFamily: theme.fonts.primary,
          color: theme.colors.textDark,
          marginBottom: 60,
          opacity: titleOpacity,
          textAlign: 'center',
        }}
      >
        {content.headline.split('...')[0]}
        <span style={{ color: '#FF6B6B' }}>...</span>
      </h2>

      <div
        style={{
          display: 'flex',
          gap: 80,
          alignItems: 'flex-start',
          justifyContent: 'center',
        }}
      >
        {/* Code Block */}
        {content.codeExample && (
          <div
            style={{
              opacity: codeBlockOpacity,
              transform: `translateX(${codeBlockX}px)`,
              background: '#1a1a1a',
              borderRadius: 16,
              padding: 30,
              border: '1px solid #333',
              width: 700,
              boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
            }}
          >
            {/* Window controls */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#FF5F56' }} />
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#FFBD2E' }} />
              <div style={{ width: 12, height: 12, borderRadius: '50%', background: '#27C93F' }} />
            </div>

            {/* Code lines */}
            <pre
              style={{
                fontFamily: theme.fonts.mono,
                fontSize: 16,
                lineHeight: 1.6,
                margin: 0,
                color: '#888',
              }}
            >
              {content.codeExample.map((line, i) => {
                const lineOpacity = interpolate(
                  frame,
                  [30 + i * 3, 35 + i * 3],
                  [0, 1],
                  { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
                );
                return (
                  <div key={i} style={{ opacity: lineOpacity }}>
                    {line}
                  </div>
                );
              })}
            </pre>
          </div>
        )}

        {/* Problem Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {content.problems.map((problem, i) => (
            <div
              key={i}
              style={{
                opacity: iconAnimations[i].opacity,
                transform: `scale(${iconAnimations[i].scale})`,
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                background: 'rgba(255, 107, 107, 0.1)',
                border: '1px solid rgba(255, 107, 107, 0.3)',
                borderRadius: 12,
                padding: '16px 24px',
              }}
            >
              <div
                style={{
                  width: 50,
                  height: 50,
                  borderRadius: 10,
                  background: '#FF6B6B',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 18,
                  fontWeight: 700,
                  color: '#fff',
                }}
              >
                {problem.icon}
              </div>
              <span
                style={{
                  color: theme.colors.textDark,
                  fontSize: 20,
                  fontWeight: 500,
                }}
              >
                {problem.text}
              </span>
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};
