// The board camera. It sits square on the board and only ever moves closer: a move names the board point that
// comes to the middle of the frame and how much closer. Board and presenter are both inside, so they move as one.
import React from 'react';
import { Easing, useCurrentFrame } from 'remotion';

export type Move = { at: number; dur: number; s: number; cx: number; cy: number };
export const HOME = { s: 1, cx: 960, cy: 540 };

export const cameraAt = (moves: Move[], frame: number) => {
  let cur = HOME;
  for (const m of moves) {
    if (frame < m.at) break;
    const t = Easing.inOut(Easing.cubic)(Math.min(1, (frame - m.at) / m.dur));
    cur = { s: cur.s + (m.s - cur.s) * t, cx: cur.cx + (m.cx - cur.cx) * t, cy: cur.cy + (m.cy - cur.cy) * t };
    if (t < 1) break;
  }
  // never past the board's edge
  const hw = 960 / cur.s;
  const hh = 540 / cur.s;
  return { s: cur.s, cx: Math.min(1920 - hw, Math.max(hw, cur.cx)), cy: Math.min(1080 - hh, Math.max(hh, cur.cy)) };
};

export const Camera: React.FC<{ moves: Move[]; children: React.ReactNode }> = ({ moves, children }) => {
  const { s, cx, cy } = cameraAt(moves, useCurrentFrame());
  return (
    <div style={{ position: 'absolute', left: 0, top: 0, width: 1920, height: 1080, transformOrigin: '0 0',
      transform: `translate(${960 - cx * s}px, ${540 - cy * s}px) scale(${s})` }}>
      {children}
    </div>
  );
};
