import {
  AbsoluteFill,
  Img,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from 'remotion';
import { brand } from '../../config/brand';
import { Triskelion } from '../core/Triskelion';
import { Captions } from '../core/Captions';
import type { CaptionChunk } from '../../config/types';

type Props = {
  portraitFile: string;
  nameplate: string;
  tagline: string;
  captions?: CaptionChunk[];
  durationInFrames: number;
};

export const LughIntroScene: React.FC<Props> = ({
  portraitFile,
  nameplate,
  tagline,
  captions,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const kenBurnsScale = interpolate(
    frame,
    [0, durationInFrames],
    [1.02, 1.10],
    { extrapolateRight: 'clamp' },
  );

  const portraitOpacity = interpolate(frame, [0, 18], [0, 1], {
    extrapolateRight: 'clamp',
  });

  const nameSpring = spring({
    frame: frame - 12,
    fps,
    config: { damping: 180, stiffness: 90 },
  });
  const nameTranslate = interpolate(nameSpring, [0, 1], [20, 0]);

  const taglineOpacity = interpolate(frame, [36, 54], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const outroFadeStart = durationInFrames - 14;
  const outroOpacity = interpolate(
    frame,
    [outroFadeStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 70% 55%, ${brand.colors.ironLight} 0%, ${brand.colors.ironDeep} 72%)`,
      }}
    >
      {/* Subtle grain / vertical inscription lines */}
      <AbsoluteFill
        style={{
          background:
            'repeating-linear-gradient(90deg, rgba(201,179,122,0.025) 0px, rgba(201,179,122,0.025) 1px, transparent 1px, transparent 4px)',
          pointerEvents: 'none',
        }}
      />

      {/* Portrait — right side, iris-fading into iron */}
      <AbsoluteFill
        style={{
          opacity: portraitOpacity * outroOpacity,
          justifyContent: 'center',
          alignItems: 'flex-end',
          paddingRight: 80,
        }}
      >
        <div
          style={{
            position: 'relative',
            width: 780,
            height: 780,
            transform: `scale(${kenBurnsScale})`,
            transformOrigin: 'center center',
          }}
        >
          <Img
            src={staticFile(portraitFile)}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              borderRadius: '50%',
              filter: 'saturate(0.85) contrast(1.05)',
            }}
          />
          {/* Vignette mask — fade portrait edges into background */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              borderRadius: '50%',
              boxShadow: `inset 0 0 180px 30px ${brand.colors.ironDeep}`,
              pointerEvents: 'none',
            }}
          />
          {/* Thin inscribed ring */}
          <div
            style={{
              position: 'absolute',
              inset: -12,
              borderRadius: '50%',
              border: `1px solid ${brand.colors.divider}`,
              boxShadow: `0 0 0 1px ${brand.colors.ironDeep}, 0 0 60px rgba(201,179,122,0.06)`,
              pointerEvents: 'none',
            }}
          />
        </div>
      </AbsoluteFill>

      {/* Left side: triskelion + nameplate + tagline */}
      <AbsoluteFill
        style={{
          opacity: outroOpacity,
          alignItems: 'flex-start',
          justifyContent: 'center',
          paddingLeft: 140,
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 48 }}>
          <Triskelion size={280} rotateSpeed={0.06} />

          <div
            style={{
              transform: `translateY(${nameTranslate}px)`,
              opacity: nameSpring,
            }}
          >
            <h1
              style={{
                margin: 0,
                color: brand.colors.silver,
                fontFamily: brand.fonts.display,
                fontSize: 168,
                fontWeight: 700,
                letterSpacing: 18,
                lineHeight: 0.95,
                textShadow: `0 2px 0 rgba(0,0,0,0.6), 0 0 40px rgba(201,179,122,0.12)`,
              }}
            >
              {nameplate}
            </h1>
            <div
              style={{
                marginTop: 18,
                width: 180,
                height: 1,
                background: `linear-gradient(90deg, ${brand.colors.accent} 0%, transparent 100%)`,
              }}
            />
          </div>

          <p
            style={{
              margin: 0,
              opacity: taglineOpacity,
              color: brand.colors.textMedium,
              fontFamily: brand.fonts.primary,
              fontSize: 28,
              fontWeight: 500,
              letterSpacing: 8,
              textTransform: 'uppercase',
            }}
          >
            {tagline}
          </p>
        </div>
      </AbsoluteFill>

      {captions && captions.length > 0 && <Captions chunks={captions} />}
    </AbsoluteFill>
  );
};
