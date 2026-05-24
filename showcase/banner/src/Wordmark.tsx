import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from 'remotion';
import { loadFont } from '@remotion/google-fonts/Audiowide';
import type { Theme } from './themes';

const { fontFamily } = loadFont();

export const Wordmark: React.FC<{ theme: Theme }> = ({ theme }) => {
  const frame = useCurrentFrame();

  const drop = interpolate(frame, [20, 48], [-30, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const fade = interpolate(frame, [20, 48], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const flickerA = frame > 48 && frame < 54 ? (frame % 2 === 0 ? 0.4 : 1) : 1;
  const flickerB = frame > 78 && frame < 82 ? (frame % 2 === 0 ? 0.7 : 1) : 1;
  const flicker = flickerA * flickerB;

  const baseStyle: React.CSSProperties = {
    fontFamily,
    fontSize: 52,
    fontWeight: 400,
    letterSpacing: 4,
    whiteSpace: 'nowrap',
  };

  return (
    <AbsoluteFill
      style={{
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingTop: 96,
      }}
    >
      <div
        style={{
          position: 'relative',
          transform: `translateY(${drop}px)`,
          opacity: fade * flicker,
        }}
      >
        {theme.wordmark.aberration.enabled && (
          <>
            <div
              style={{
                ...baseStyle,
                position: 'absolute',
                inset: 0,
                transform: `translate(-${theme.wordmark.aberration.offset}px, 0)`,
                color: theme.wordmark.aberration.left,
                opacity: 0.65,
              }}
              aria-hidden
            >
              CLAUDE CODE VIDEO TOOLKIT
            </div>
            <div
              style={{
                ...baseStyle,
                position: 'absolute',
                inset: 0,
                transform: `translate(${theme.wordmark.aberration.offset}px, 0)`,
                color: theme.wordmark.aberration.right,
                opacity: 0.65,
              }}
              aria-hidden
            >
              CLAUDE CODE VIDEO TOOLKIT
            </div>
          </>
        )}

        <div
          style={{
            ...baseStyle,
            color: theme.wordmark.color,
            textShadow: theme.wordmark.glow.join(', '),
          }}
        >
          CLAUDE CODE VIDEO TOOLKIT
        </div>
      </div>
    </AbsoluteFill>
  );
};
