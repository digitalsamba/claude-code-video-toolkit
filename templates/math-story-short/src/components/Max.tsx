import { interpolate, useCurrentFrame } from 'remotion';
import { theme } from '../config/theme';
import type { MaxDirection, MaxMood } from '../config/types';

const SKIN = '#fde6c8';
const SKIN_SHADE = '#f1c9a0';
const HAIR = '#780000';
const HAIR_LIGHT = '#a31d26';
const FRAME = '#062338';

// Curls around the crown (cx, cy, r): fixed so Max always has the same hair
const CURLS: [number, number, number][] = [
  [92, 168, 40], [104, 120, 44], [140, 86, 46], [186, 70, 48], [234, 72, 46],
  [276, 92, 44], [306, 128, 42], [312, 172, 38], [122, 126, 36], [282, 140, 34],
];
const FRINGE: [number, number, number][] = [[142, 132, 30], [184, 120, 32], [226, 122, 31], [264, 138, 28]];

const BROWS: Record<MaxMood, [number, number]> = {
  happy: [-6, 6],
  thinking: [8, -14],
  smug: [-2, -16],
  shocked: [-14, 14],
  confused: [14, -4],
  sad: [16, -16],
};
const TILT: Record<MaxMood, number> = { happy: 0, thinking: -6, smug: 5, shocked: 0, confused: 9, sad: -4 };

const Mouth: React.FC<{ mood: MaxMood; talk: number }> = ({ mood, talk }) => {
  switch (mood) {
    case 'happy':
      return (
        <g>
          <path d={`M168,282 Q200,${318 + talk * 6} 232,282 Z`} fill="#5a0d14" />
          <path d={`M180,298 Q200,${312 + talk * 4} 220,298`} fill="none" stroke="#e5626b" strokeWidth={8} strokeLinecap="round" />
        </g>
      );
    case 'smug':
      return <path d="M176,292 Q206,304 230,280" fill="none" stroke="#5a0d14" strokeWidth={8} strokeLinecap="round" />;
    case 'shocked':
      return <ellipse cx={200} cy={296} rx={18} ry={24 + talk * 3} fill="#5a0d14" />;
    case 'confused':
      return <path d="M172,294 Q186,284 200,294 T228,290" fill="none" stroke="#5a0d14" strokeWidth={8} strokeLinecap="round" />;
    case 'sad':
      return <path d="M176,302 Q200,282 224,302" fill="none" stroke="#5a0d14" strokeWidth={8} strokeLinecap="round" />;
    case 'thinking':
    default:
      return <path d="M184,294 L222,288" fill="none" stroke="#5a0d14" strokeWidth={8} strokeLinecap="round" />;
  }
};

const LOOK: Record<NonNullable<MaxDirection['look']>, [number, number]> = {
  up: [3, -9],
  left: [-10, 0],
  right: [10, -2],
  center: [0, 0],
  down: [0, 8],
};

export interface MaxProps {
  mood: MaxMood;
  look?: MaxDirection['look'];
  /** Frames since the mood was set: drives the little hop on a change */
  since: number;
  /** 0-1 talking amount (mouth opening) */
  talk?: number;
  height?: number;
  /** Raise an arm toward the stage */
  pose?: 'point';
}

/**
 * Max, redrawn as layered SVG: curly maroon hair, round glasses, crimson
 * hoodie with a pi badge. Blinks, breathes, hops when his mood changes.
 * Body is a 400x500 box with the feet on the bottom edge.
 */
export const Max: React.FC<MaxProps> = ({ mood, look = 'up', since, talk = 0, height = 500, pose }) => {
  const frame = useCurrentFrame();
  const breathe = Math.sin(frame / 22);
  const bob = Math.sin(frame / 18) * 4;
  const blinkPhase = (frame + 37) % 113;
  const blink = blinkPhase < 6 ? interpolate(blinkPhase, [0, 3, 6], [1, 0.08, 1]) : 1;
  const hop = since < 14 ? -Math.sin((since / 14) * Math.PI) * 26 : 0;
  const squash = since < 14 ? 1 + Math.sin((since / 14) * Math.PI) * 0.04 : 1;
  const [lx, ly] = LOOK[look];
  const [bl, br] = BROWS[mood];
  const lid = mood === 'smug' ? 0.45 : mood === 'sad' ? 0.3 : 0;

  const eye = (cx: number) => (
    <g transform={`translate(${cx + lx}, ${214 + ly}) scale(1, ${blink})`}>
      <ellipse cx={0} cy={0} rx={mood === 'shocked' ? 12 : 11} ry={mood === 'shocked' ? 15 : 13} fill={FRAME} />
      <circle cx={4} cy={-5} r={4} fill="#ffffff" />
      {lid > 0 && <rect x={-20} y={-22} width={40} height={22 * lid + 8} fill={SKIN} />}
    </g>
  );

  return (
    <svg width={(height * 400) / 500} height={height} viewBox="0 0 400 500" style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id="max-hoodie" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#d61f2c" />
          <stop offset="100%" stopColor="#8f0a14" />
        </linearGradient>
        <radialGradient id="max-skin" cx="0.42" cy="0.38" r="0.75">
          <stop offset="0%" stopColor="#fff3e0" />
          <stop offset="70%" stopColor={SKIN} />
          <stop offset="100%" stopColor={SKIN_SHADE} />
        </radialGradient>
        <radialGradient id="max-hair" cx="0.4" cy="0.3" r="0.8">
          <stop offset="0%" stopColor={HAIR_LIGHT} />
          <stop offset="100%" stopColor={HAIR} />
        </radialGradient>
      </defs>

      <ellipse cx={200} cy={494} rx={130 - hop * 1.2} ry={12} fill="rgba(0, 10, 20, 0.45)" />

      <g transform={`translate(0, ${hop}) translate(200, 500) scale(${1 / squash}, ${squash}) translate(-200, -500)`}>
        {/* body */}
        <g transform={`translate(200, 500) scale(1, ${1 + breathe * 0.012}) translate(-200, -500)`}>
          <path d="M62,500 C62,398 112,338 200,338 C288,338 338,398 338,500 Z" fill="url(#max-hoodie)" />
          <path d="M112,500 C120,440 150,420 200,420 C250,420 280,440 288,500" fill="rgba(0,0,0,0.08)" />
          <ellipse cx={200} cy={346} rx={78} ry={24} fill={theme.primaryDark} />
          <path d="M178,356 L170,414" stroke={theme.cream} strokeWidth={6} strokeLinecap="round" />
          <path d="M222,356 L230,414" stroke={theme.cream} strokeWidth={6} strokeLinecap="round" />
          <circle cx={170} cy={418} r={7} fill={theme.cream} />
          <circle cx={230} cy={418} r={7} fill={theme.cream} />
          <circle cx={262} cy={446} r={24} fill={theme.cream} />
          <text x={262} y={458} textAnchor="middle" fontFamily={theme.fonts.math} fontStyle="italic" fontSize={36} fill={theme.bg}>
            π
          </text>
          {pose === 'point' && mood !== 'thinking' && (
            <g transform={`rotate(${Math.sin(frame / 14) * 3}, 306, 404)`}>
              <path d="M306,404 C340,360 362,316 372,268" fill="none" stroke="#b5111c" strokeWidth={46} strokeLinecap="round" />
              <path d="M306,404 C340,360 362,316 372,268" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={12} strokeLinecap="round" transform="translate(-10,0)" />
              <circle cx={374} cy={252} r={27} fill={SKIN} stroke={SKIN_SHADE} strokeWidth={3} />
              <rect x={381} y={196} width={17} height={50} rx={8.5} fill={SKIN} stroke={SKIN_SHADE} strokeWidth={3} transform="rotate(18, 389, 246)" />
            </g>
          )}
          {mood === 'thinking' && (
            <g>
              <path d="M300,430 C300,380 280,340 246,326" fill="none" stroke="#b5111c" strokeWidth={46} strokeLinecap="round" />
              <circle cx={240} cy={318} r={26} fill={SKIN} stroke={SKIN_SHADE} strokeWidth={3} />
            </g>
          )}
        </g>

        {/* head */}
        <g transform={`translate(0, ${bob}) rotate(${TILT[mood]}, 200, 320)`}>
          <rect x={178} y={300} width={44} height={46} rx={14} fill={SKIN_SHADE} />
          <ellipse cx={200} cy={150} rx={128} ry={86} fill="url(#max-hair)" />
          {CURLS.map(([x, y, r], i) => (
            <circle key={i} cx={x} cy={y} r={r} fill="url(#max-hair)" />
          ))}
          <circle cx={88} cy={226} r={22} fill={SKIN_SHADE} />
          <circle cx={312} cy={226} r={22} fill={SKIN_SHADE} />
          <ellipse cx={200} cy={218} rx={112} ry={110} fill="url(#max-skin)" />
          {FRINGE.map(([x, y, r], i) => (
            <circle key={i} cx={x} cy={y} r={r} fill="url(#max-hair)" />
          ))}
          {CURLS.slice(1, 7).map(([x, y, r], i) => (
            <path key={`h${i}`} d={`M${x - r * 0.4},${y - r * 0.35} q${r * 0.3},-${r * 0.3} ${r * 0.6},-${r * 0.05}`} stroke="#c4414a" strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.7} />
          ))}

          <ellipse cx={128} cy={262} rx={22} ry={13} fill="#f28b8b" opacity={0.45} />
          <ellipse cx={272} cy={262} rx={22} ry={13} fill="#f28b8b" opacity={0.45} />

          {/* brows */}
          <path d="M126,172 Q154,160 182,170" transform={`rotate(${bl}, 154, 168)`} stroke={HAIR} strokeWidth={9} fill="none" strokeLinecap="round" />
          <path d="M218,170 Q246,160 274,172" transform={`rotate(${br}, 246, 168)`} stroke={HAIR} strokeWidth={9} fill="none" strokeLinecap="round" />

          {eye(155)}
          {eye(245)}

          {/* glasses */}
          <circle cx={155} cy={216} r={40} fill="rgba(102,155,188,0.14)" stroke={FRAME} strokeWidth={9} />
          <circle cx={245} cy={216} r={40} fill="rgba(102,155,188,0.14)" stroke={FRAME} strokeWidth={9} />
          <path d="M195,212 Q200,204 205,212" stroke={FRAME} strokeWidth={8} fill="none" />
          <path d="M115,210 L92,204 M285,210 L308,204" stroke={FRAME} strokeWidth={7} strokeLinecap="round" />
          <path d="M132,196 Q140,186 152,184" stroke="#ffffff" strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.75} />
          <path d="M222,196 Q230,186 242,184" stroke="#ffffff" strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.75} />

          <path d="M196,240 Q190,256 202,258" stroke={SKIN_SHADE} strokeWidth={6} fill="none" strokeLinecap="round" />
          <Mouth mood={mood} talk={talk} />

          {(mood === 'shocked' || mood === 'sad') && (
            <path d="M318,128 C318,128 302,152 302,162 A16,16 0 0 0 334,162 C334,152 318,128 318,128 Z" fill="#8fc3e0" opacity={interpolate(since, [4, 12], [0, 0.95], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })} />
          )}
        </g>
      </g>
    </svg>
  );
};
