// The chalkboard itself: slate, the ghosts of erased chalk, a wooden frame and a chalk tray. Static, so the video
// uses a still of it (public/board/bg.png: `npm run board` writes it) when that exists.
import React from 'react';
import { AbsoluteFill, Img, getStaticFiles, staticFile } from 'remotion';

export const W = 1920;
export const H = 1080;
export const WOOD = 26;
export const TRAY = 1032;

/** The board drawn. `w`, `h` and `tray` draw it at another size (a short's 1080x1920, its tray where the
 * presenter's clip puts it); the sticks and the eraser keep their places as shares of the width. */
export const BoardDrawn: React.FC<{ w?: number; h?: number; tray?: number }> = ({ w: W = 1920, h: H = 1080, tray: TRAY = 1032 }) => (
  <AbsoluteFill style={{ background: '#26332f' }}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
      <defs>
        <radialGradient id="slate" cx="42%" cy="38%" r="85%">
          <stop offset="0" stopColor="#33453f" />
          <stop offset="1" stopColor="#222d2a" />
        </radialGradient>
        <filter id="smear" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.0016 0.0034" numOctaves="2" seed="11" />
          <feColorMatrix type="matrix" values="0 0 0 0 0.93  0 0 0 0 0.95  0 0 0 0 0.9  1.5 0 0 0 -0.62" />
        </filter>
        <filter id="grain" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="4" />
          <feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0.5 0 0 0 -0.2" />
        </filter>
        <linearGradient id="wood" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#8a6240" />
          <stop offset="0.5" stopColor="#6f4c30" />
          <stop offset="1" stopColor="#5a3c25" />
        </linearGradient>
        <linearGradient id="woodV" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#7d5839" />
          <stop offset="1" stopColor="#5f4129" />
        </linearGradient>
      </defs>
      <rect width={W} height={H} fill="url(#slate)" />
      <rect width={W} height={H} filter="url(#smear)" opacity="0.13" />
      <rect width={W} height={H} filter="url(#grain)" opacity="0.1" />
      {/* frame */}
      <rect x="0" y="0" width={W} height={WOOD} fill="url(#wood)" />
      <rect x="0" y="0" width={WOOD} height={H} fill="url(#woodV)" />
      <rect x={W - WOOD} y="0" width={WOOD} height={H} fill="url(#woodV)" />
      <rect x={WOOD} y={WOOD} width={W - 2 * WOOD} height="5" fill="#000" opacity="0.28" />
      <rect x={WOOD} y={WOOD} width="5" height={H} fill="#000" opacity="0.22" />
      {/* tray: a ledge with a lip, chalk dust on it, three sticks and a felt eraser */}
      <rect x="0" y={TRAY - 8} width={W} height="8" fill="#000" opacity="0.3" />
      <rect x="0" y={TRAY} width={W} height={H - TRAY} fill="url(#wood)" />
      <rect x="0" y={TRAY} width={W} height="6" fill="#a37a52" />
      <rect x="0" y={TRAY + 6} width={W} height="14" filter="url(#grain)" opacity="0.35" />
      {[[330 * W / 1920, 0, '#f1eee4'], [330 * W / 1920 + 62, -3, '#efd27a'], [1210 * W / 1920, 2, '#f1eee4']].map(([x, r, c], i) => (
        <rect key={i} x={x as number} y={TRAY - 11} width="58" height="13" rx="5" fill={c as string}
          transform={`rotate(${r} ${x} ${TRAY})`} />
      ))}
      <g transform={`translate(${Math.round(560 * W / 1920)} ${TRAY - 30})`}>
        <rect width="118" height="16" rx="3" fill="#6b4a2f" />
        <rect y="16" width="118" height="15" rx="3" fill="#2a2a2c" />
        <rect y="24" width="118" height="7" rx="3" fill="#c9c6bc" opacity="0.55" />
      </g>
    </svg>
  </AbsoluteFill>
);

/** The board: the pre-rendered still when it is there (`npm run board`), otherwise drawn live. */
export const Board: React.FC<{ still?: string; w?: number; h?: number; tray?: number }> = ({ still = 'board/bg.png', w = W, h = H, tray }) => {
  const has = getStaticFiles().some((f) => f.name === still);
  return has ? (
    <AbsoluteFill>
      <Img src={staticFile(still)} style={{ width: w, height: h }} />
    </AbsoluteFill>
  ) : (
    <BoardDrawn w={w} h={h} tray={tray} />
  );
};
