import { AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';
import { brand } from '../../config/brand';
import type { CaptionChunk } from '../../config/types';

type Props = {
  chunks: CaptionChunk[];
  style?: 'inscribed' | 'modern';
};

const FADE = 6;

export const Captions: React.FC<Props> = ({ chunks, style = 'inscribed' }) => {
  const frame = useCurrentFrame();

  const active = chunks.find(
    (c) => frame >= c.fromFrame && frame < c.fromFrame + c.durationInFrames,
  );

  if (!active) return null;

  const local = frame - active.fromFrame;
  const opacity = interpolate(
    local,
    [0, FADE, active.durationInFrames - FADE, active.durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  const inscribed = style === 'inscribed';

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom: 90,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          opacity,
          maxWidth: '72%',
          padding: '22px 46px',
          background: inscribed
            ? 'linear-gradient(180deg, rgba(10,8,6,0.78) 0%, rgba(10,8,6,0.92) 100%)'
            : 'rgba(0,0,0,0.75)',
          border: inscribed ? `1px solid ${brand.colors.divider}` : 'none',
          borderRadius: inscribed ? brand.borderRadius.sm : brand.borderRadius.md,
          boxShadow: inscribed
            ? `0 2px 0 rgba(255,255,255,0.04) inset, 0 18px 40px ${brand.colors.shadow}`
            : `0 12px 30px ${brand.colors.shadow}`,
          textAlign: 'center',
        }}
      >
        <p
          style={{
            margin: 0,
            color: brand.colors.silver,
            fontFamily: brand.fonts.primary,
            fontSize: 44,
            fontWeight: 500,
            letterSpacing: inscribed ? 1 : 0,
            lineHeight: 1.25,
            textShadow: inscribed ? '0 1px 0 rgba(0,0,0,0.8)' : 'none',
          }}
        >
          {active.text}
        </p>
      </div>
    </AbsoluteFill>
  );
};
