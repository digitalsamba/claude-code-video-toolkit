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
import { Captions } from '../core/Captions';
import { Triskelion } from '../core/Triskelion';
import type { TicketScene as TicketSceneData, Subsystem } from '../../config/types';

const SUBSYSTEM_LABEL: Record<Subsystem, string> = {
  sambalive: 'SambaLive',
  scaleway: 'HLS pipeline',
  longarm: 'Longarm',
  mobile: 'Mobile',
  integrations: 'Integrations',
  'account-center': 'Account Center',
  locales: 'Locales',
};

type Props = {
  ticket: TicketSceneData;
  portraitFile: string;
  index: number;
  total: number;
  durationInFrames: number;
};

export const TicketScene: React.FC<Props> = ({
  ticket,
  portraitFile,
  index,
  total,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const keySpring = spring({ frame, fps, config: { damping: 200, stiffness: 80 } });
  const keyTranslate = interpolate(keySpring, [0, 1], [-30, 0]);

  const titleOpacity = interpolate(frame, [12, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const titleTranslate = interpolate(frame, [12, 30], [20, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const brokeOpacity = interpolate(frame, [36, 54], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const shippedOpacity = interpolate(frame, [66, 84], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const outroFadeStart = durationInFrames - 12;
  const outroOpacity = interpolate(
    frame,
    [outroFadeStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' },
  );

  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(135deg, ${brand.colors.iron} 0%, ${brand.colors.ironDeep} 100%)`,
        opacity: outroOpacity,
      }}
    >
      {/* Faint vertical inscription lines */}
      <AbsoluteFill
        style={{
          background:
            'repeating-linear-gradient(90deg, rgba(201,179,122,0.02) 0px, rgba(201,179,122,0.02) 1px, transparent 1px, transparent 3px)',
          pointerEvents: 'none',
        }}
      />

      {/* Corner triskelion */}
      <div style={{ position: 'absolute', top: 48, right: 56, opacity: 0.5 }}>
        <Triskelion size={130} rotateSpeed={0.04} strokeWidth={1.6} />
      </div>

      {/* Sprint progress marker: "N of M" */}
      <div
        style={{
          position: 'absolute',
          top: 70,
          left: 120,
          color: brand.colors.textLight,
          fontFamily: brand.fonts.primary,
          fontSize: 22,
          fontWeight: 600,
          letterSpacing: 6,
          textTransform: 'uppercase',
        }}
      >
        Ticket {index + 1} of {total}
      </div>

      {/* Main content column */}
      <div
        style={{
          position: 'absolute',
          top: 150,
          left: 120,
          right: 360,
          display: 'flex',
          flexDirection: 'column',
          gap: 36,
        }}
      >
        {/* Ticket key — inscribed */}
        <div
          style={{
            opacity: keySpring,
            transform: `translateX(${keyTranslate}px)`,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 20,
              marginBottom: 14,
            }}
          >
            <span
              style={{
                color: brand.colors.accent,
                fontFamily: brand.fonts.primary,
                fontSize: 22,
                fontWeight: 600,
                letterSpacing: 6,
                textTransform: 'uppercase',
              }}
            >
              Closed
            </span>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 10,
                padding: '6px 16px',
                border: `1px solid ${brand.colors.divider}`,
                borderRadius: brand.borderRadius.sm,
                background: 'rgba(201, 179, 122, 0.05)',
                color: brand.colors.silver,
                fontFamily: brand.fonts.primary,
                fontSize: 20,
                fontWeight: 600,
                letterSpacing: 4,
                textTransform: 'uppercase',
              }}
            >
              <span style={{ color: brand.colors.accent }}>▸</span>
              {SUBSYSTEM_LABEL[ticket.subsystem]}
            </span>
          </div>
          <h1
            style={{
              margin: 0,
              color: brand.colors.silver,
              fontFamily: brand.fonts.display,
              fontSize: 108,
              fontWeight: 700,
              letterSpacing: 10,
              lineHeight: 1,
              textShadow: '0 2px 0 rgba(0,0,0,0.6)',
            }}
          >
            {ticket.key}
          </h1>
        </div>

        {/* Divider */}
        <div
          style={{
            height: 1,
            width: 360,
            background: `linear-gradient(90deg, ${brand.colors.accent} 0%, transparent 100%)`,
            opacity: titleOpacity,
          }}
        />

        {/* Title */}
        <h2
          style={{
            margin: 0,
            opacity: titleOpacity,
            transform: `translateY(${titleTranslate}px)`,
            color: brand.colors.silver,
            fontFamily: brand.fonts.primary,
            fontSize: 48,
            fontWeight: 600,
            lineHeight: 1.2,
            letterSpacing: -0.3,
            maxWidth: 1200,
          }}
        >
          {ticket.title}
        </h2>

        {/* Two-column problem → fix */}
        <div style={{ display: 'flex', gap: 80, marginTop: 24 }}>
          <div style={{ flex: 1, opacity: brokeOpacity }}>
            <div
              style={{
                color: '#b87060',
                fontFamily: brand.fonts.primary,
                fontSize: 20,
                fontWeight: 600,
                letterSpacing: 4,
                textTransform: 'uppercase',
                marginBottom: 14,
              }}
            >
              What broke
            </div>
            <p
              style={{
                margin: 0,
                color: brand.colors.textMedium,
                fontFamily: brand.fonts.primary,
                fontSize: 32,
                fontWeight: 400,
                lineHeight: 1.35,
              }}
            >
              {ticket.whatBroke}
            </p>
          </div>

          <div style={{ flex: 1, opacity: shippedOpacity }}>
            <div
              style={{
                color: brand.colors.accent,
                fontFamily: brand.fonts.primary,
                fontSize: 20,
                fontWeight: 600,
                letterSpacing: 4,
                textTransform: 'uppercase',
                marginBottom: 14,
              }}
            >
              What shipped
            </div>
            <p
              style={{
                margin: 0,
                color: brand.colors.textDark,
                fontFamily: brand.fonts.primary,
                fontSize: 32,
                fontWeight: 500,
                lineHeight: 1.35,
              }}
            >
              {ticket.whatShipped}
            </p>
          </div>
        </div>
      </div>

      {/* Lugh PiP bottom-right */}
      <div
        style={{
          position: 'absolute',
          bottom: 60,
          right: 60,
          width: 220,
          height: 220,
        }}
      >
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            overflow: 'hidden',
            border: `1px solid ${brand.colors.divider}`,
            boxShadow: `0 10px 40px ${brand.colors.shadow}, 0 0 0 4px ${brand.colors.ironDeep}`,
          }}
        >
          <Img
            src={staticFile(portraitFile)}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              filter: 'saturate(0.9)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              inset: 0,
              boxShadow: `inset 0 0 40px 4px ${brand.colors.ironDeep}`,
            }}
          />
        </div>
      </div>

      {ticket.captions && ticket.captions.length > 0 && (
        <Captions chunks={ticket.captions} />
      )}
    </AbsoluteFill>
  );
};
