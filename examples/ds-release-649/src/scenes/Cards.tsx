import React from 'react';
import { AbsoluteFill, Img, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { fade } from '../Stage';
import { RFONT } from '../room/theme';

// Restyled to the DS act palette (video-design palettes.md, "brand blue + cool grey + navy"; tokens from the DS
// design system): identity = DS blue with white type, record = cool grey with navy type and a blue accent.
const BLUE = '#3771e0';
const GREY = '#eef2f8';
const NAVY = '#323e66';
const ON_BLUE = '#ffffff';

export const TITLE_LEN = 150;
export const OUTRO_LEN = 120;

const ITEMS = [
  'Hide tiles in recordings by role',
  'Mute or turn off cameras by role',
  'Safer ban duration',
  'Recording bookmarks through the SDK',
  'Unmute by default in breakout rooms',
  'Fixed: admitting from the lobby',
];


// Title and outro on the identity field: flat DS blue, white type, the same collage of video tiles.
const Navy: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill style={{ background: BLUE, fontFamily: RFONT, color: ON_BLUE }}>
    {children}
  </AbsoluteFill>
);

const Frame: React.FC<{ img: string; w: number; h: number; name?: string; speaking?: boolean; style?: React.CSSProperties }> = ({ img, w, h, name, speaking, style }) => (
  <div style={{ width: w, height: h, borderRadius: 14, overflow: 'hidden', position: 'relative', ...style }}>
    <Img src={staticFile(`people/${img}.jpg`)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
    {name && (
      <div style={{ position: 'absolute', left: 12, bottom: 12, display: 'flex', alignItems: 'center', gap: 8, padding: '5px 12px', borderRadius: 14, background: 'rgba(0,0,0,0.6)', fontSize: 17 }}>
        <span style={{ width: 8, height: 8, borderRadius: 4, background: speaking ? '#00ff00' : 'rgba(255,255,255,0.7)' }} />
        {name}
      </div>
    )}
  </div>
);

export const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = (d: number) => spring({ frame: frame - d, fps, config: { damping: 18, mass: 0.9 } });
  const a = s(0);
  const b = s(6);
  const big = s(10);
  const smalls = [s(22), s(28), s(34)];
  const drift = frame * 0.06;
  return (
    <Navy>
      <div style={{ position: 'absolute', left: 140, top: 270, opacity: a, transform: `translateY(${(1 - a) * 20}px)` }}>
        <Img src={staticFile('logo_mono.svg')} style={{ height: 40, filter: 'brightness(0) invert(1)' }} />
      </div>
      <div style={{ position: 'absolute', left: 140, top: 350, opacity: b, transform: `translateY(${(1 - b) * 30}px)` }}>
        <div style={{ fontSize: 136, fontWeight: 800, letterSpacing: -4, lineHeight: 1 }}>Release 649</div>
        <div style={{ fontSize: 36, color: ON_BLUE, opacity: 0.9, fontWeight: 600, marginTop: 22 }}>What's new · October 2026</div>
      </div>
      <div style={{ position: 'absolute', left: 140, top: 590, display: 'grid', gridTemplateColumns: 'auto', rowGap: 14, width: 860 }}>
        {ITEMS.map((t, i) => {
          const o = fade(frame, 30 + i * 6, 42 + i * 6);
          return (
            <div key={t} style={{ fontSize: 28, fontWeight: 600, whiteSpace: 'nowrap', opacity: o, transform: `translateY(${(1 - o) * 12}px)`, display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ width: 10, height: 10, borderRadius: 5, flexShrink: 0, background: t.startsWith('Fixed') ? NAVY : ON_BLUE }} />
              {t}
            </div>
          );
        })}
      </div>

      {/* Hero collage */}
      <div
        style={{
          position: 'absolute',
          left: 1040,
          top: 280,
          padding: 12,
          borderRadius: 22,
          background: '#fff',
          display: 'flex',
          gap: 12,
          opacity: big,
          transform: `translateY(${(1 - big) * 40 - drift}px) scale(${0.94 + 0.06 * big})`,
        }}
      >
        <Frame img="elena" w={520} h={520} name="Elena Moreau" speaking style={{ boxShadow: 'none' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {['matthias', 'sofia', 'julien'].map((p, i) => (
            <Frame key={p} img={p} w={190} h={165} style={{ boxShadow: 'none', opacity: smalls[i], transform: `translateX(${(1 - smalls[i]) * 30}px)` }} />
          ))}
        </div>
      </div>
      <div style={{ position: 'absolute', left: 140, bottom: 30, fontSize: 28, color: ON_BLUE, opacity: 0.85 }}>People shown are AI-generated</div>
    </Navy>
  );
};

export const OutroCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = spring({ frame, fps, config: { damping: 200 } });
  const people = ['elena', 'matthias', 'sofia', 'jonas', 'julien', 'mateo', 'isabelle', 'amara'];
  return (
    <Navy>
      <div style={{ position: 'absolute', left: 140, top: 230, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 30, opacity: a, transform: `translateY(${(1 - a) * 24}px)` }}>
        <Img src={staticFile('logo_mono.svg')} style={{ height: 46, filter: 'brightness(0) invert(1)' }} />
        <div style={{ fontSize: 112, fontWeight: 800, letterSpacing: -4, lineHeight: 1 }}>Release 649 is available now</div>
        <div style={{ fontSize: 36, color: ON_BLUE, opacity: 0.9 * fade(frame, 16, 28), fontWeight: 600 }}>digitalsamba.com/release-notes</div>
      </div>
      <div style={{ position: 'absolute', left: 140, top: 640, display: 'flex', gap: 16 }}>
        {people.map((p, i) => {
          const o = spring({ frame: frame - 10 - i * 3, fps, config: { damping: 16 } });
          return (
            <div key={p} style={{ padding: 6, borderRadius: 16, background: '#fff', opacity: o, transform: `translateY(${(1 - o) * 40}px)` }}>
              <Frame img={p} w={190} h={150} style={{ boxShadow: 'none', borderRadius: 11 }} />
            </div>
          );
        })}
      </div>
      <div style={{ position: 'absolute', left: 140, bottom: 30, fontSize: 28, color: ON_BLUE, opacity: 0.85 }}>People shown are AI-generated</div>
    </Navy>
  );
};

// Section heading over each feature scene, worded as in the release notes
export const SceneLabel: React.FC<{ kind: string; title: string }> = ({ kind, title }) => {
  const frame = useCurrentFrame();
  const o = fade(frame, 6, 16, 84, 96);
  if (o <= 0) return null;
  return (
    <div
      style={{
        position: 'absolute',
        left: 60,
        top: 44,
        opacity: o,
        transform: `translateY(${(1 - o) * -10}px)`,
        background: GREY,
        borderRadius: 0,
        padding: '16px 26px',
        fontFamily: RFONT,
        zIndex: 50,
      }}
    >
      <div style={{ fontSize: 28, fontWeight: 700, color: kind === 'Fixed' ? NAVY : BLUE }}>{kind}</div>
      <div style={{ fontSize: 36, fontWeight: 800, color: NAVY, marginTop: 4 }}>{title}</div>
    </div>
  );
};
