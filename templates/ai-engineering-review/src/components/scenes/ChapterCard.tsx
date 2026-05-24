import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from 'remotion';
import { brand } from '../../config/brand';
import { Triskelion } from '../core/Triskelion';

type Props = {
  label: string;
  sprintName: string;
  dateRange: string;
  durationInFrames: number;
};

export const ChapterCard: React.FC<Props> = ({
  label,
  sprintName,
  dateRange,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const labelSpring = spring({ frame, fps, config: { damping: 180, stiffness: 90 } });

  const nameSpring = spring({
    frame: frame - 6,
    fps,
    config: { damping: 200, stiffness: 80 },
  });

  const dateOpacity = interpolate(frame, [18, 36], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const outroFadeStart = durationInFrames - 10;
  const outroOpacity = interpolate(
    frame,
    [outroFadeStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  return (
    <AbsoluteFill
      style={{
        opacity: outroOpacity,
        background: `radial-gradient(ellipse at 50% 50%, ${brand.colors.ironLight} 0%, ${brand.colors.ironDeep} 80%)`,
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <AbsoluteFill
        style={{
          background:
            'repeating-linear-gradient(90deg, rgba(201,179,122,0.02) 0px, rgba(201,179,122,0.02) 1px, transparent 1px, transparent 3px)',
          pointerEvents: 'none',
        }}
      />

      {/* Faded center triskelion */}
      <div style={{ position: 'absolute', opacity: 0.12 }}>
        <Triskelion size={620} rotateSpeed={0.03} strokeWidth={1.2} />
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 26,
          textAlign: 'center',
          position: 'relative',
        }}
      >
        <div
          style={{
            opacity: labelSpring,
            color: brand.colors.accent,
            fontFamily: brand.fonts.primary,
            fontSize: 24,
            fontWeight: 600,
            letterSpacing: 14,
            textTransform: 'uppercase',
          }}
        >
          ▸ {label}
        </div>

        <h1
          style={{
            margin: 0,
            opacity: nameSpring,
            transform: `translateY(${interpolate(nameSpring, [0, 1], [18, 0])}px)`,
            color: brand.colors.silver,
            fontFamily: brand.fonts.display,
            fontSize: 120,
            fontWeight: 700,
            letterSpacing: 14,
            lineHeight: 1,
            textShadow: '0 2px 0 rgba(0,0,0,0.7), 0 0 40px rgba(201,179,122,0.12)',
            textTransform: 'uppercase',
          }}
        >
          {sprintName}
        </h1>

        <div
          style={{
            width: 220,
            height: 1,
            background: `linear-gradient(90deg, transparent 0%, ${brand.colors.accent} 50%, transparent 100%)`,
            opacity: dateOpacity,
          }}
        />

        <p
          style={{
            margin: 0,
            opacity: dateOpacity,
            color: brand.colors.textMedium,
            fontFamily: brand.fonts.primary,
            fontSize: 26,
            fontWeight: 500,
            letterSpacing: 6,
            textTransform: 'uppercase',
          }}
        >
          {dateRange}
        </p>
      </div>
    </AbsoluteFill>
  );
};
