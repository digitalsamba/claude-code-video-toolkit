import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from 'remotion';
import { brand } from '../../config/brand';
import { Captions } from '../core/Captions';
import { Triskelion } from '../core/Triskelion';
import type { CaptionChunk } from '../../config/types';

type Props = {
  team: string;
  sprintName: string;
  subtitle?: string;
  captions?: CaptionChunk[];
  durationInFrames: number;
};

export const TitleCardScene: React.FC<Props> = ({
  team,
  sprintName,
  subtitle,
  captions,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const teamSpring = spring({ frame, fps, config: { damping: 180, stiffness: 90 } });
  const nameSpring = spring({
    frame: frame - 10,
    fps,
    config: { damping: 180, stiffness: 80 },
  });

  const outroFadeStart = durationInFrames - 12;
  const outroOpacity = interpolate(
    frame,
    [outroFadeStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  const subtitleOpacity = interpolate(frame, [28, 48], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        opacity: outroOpacity,
        background: `radial-gradient(ellipse at 50% 50%, ${brand.colors.ironLight} 0%, ${brand.colors.ironDeep} 72%)`,
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <AbsoluteFill
        style={{
          background:
            'repeating-linear-gradient(90deg, rgba(201,179,122,0.025) 0px, rgba(201,179,122,0.025) 1px, transparent 1px, transparent 4px)',
          pointerEvents: 'none',
        }}
      />

      <div style={{ position: 'absolute', top: 140, opacity: 0.4 }}>
        <Triskelion size={160} rotateSpeed={0.05} strokeWidth={1.5} />
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 32,
          textAlign: 'center',
        }}
      >
        <div
          style={{
            opacity: teamSpring,
            color: brand.colors.accent,
            fontFamily: brand.fonts.primary,
            fontSize: 28,
            fontWeight: 600,
            letterSpacing: 12,
            textTransform: 'uppercase',
          }}
        >
          {team}
        </div>

        <h1
          style={{
            margin: 0,
            opacity: nameSpring,
            transform: `translateY(${interpolate(nameSpring, [0, 1], [16, 0])}px)`,
            color: brand.colors.silver,
            fontFamily: brand.fonts.display,
            fontSize: 160,
            fontWeight: 700,
            letterSpacing: 14,
            lineHeight: 1,
            textShadow: '0 2px 0 rgba(0,0,0,0.7), 0 0 50px rgba(201,179,122,0.12)',
          }}
        >
          {sprintName}
        </h1>

        <div
          style={{
            width: 240,
            height: 1,
            background: `linear-gradient(90deg, transparent 0%, ${brand.colors.accent} 50%, transparent 100%)`,
            opacity: subtitleOpacity,
          }}
        />

        {subtitle && (
          <p
            style={{
              margin: 0,
              opacity: subtitleOpacity,
              color: brand.colors.textMedium,
              fontFamily: brand.fonts.primary,
              fontSize: 28,
              fontWeight: 500,
              letterSpacing: 8,
              textTransform: 'uppercase',
            }}
          >
            {subtitle}
          </p>
        )}
      </div>

      {captions && captions.length > 0 && <Captions chunks={captions} />}
    </AbsoluteFill>
  );
};
