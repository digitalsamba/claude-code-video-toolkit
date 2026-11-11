import React from 'react';
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { Cursor, Point, cursorAt } from './Cursor';
import { APP_H, APP_W, CHROME_H, RFONT } from './room/theme';

export type Cam = { f: number; z: number; x: number; y: number };

const ease = Easing.bezier(0.65, 0, 0.35, 1);
export const cameraAt = (frame: number, keys: Cam[]): Cam => {
  if (frame <= keys[0].f) return keys[0];
  for (let i = 1; i < keys.length; i++) {
    const a = keys[i - 1];
    const b = keys[i];
    if (frame <= b.f) {
      const t = ease((frame - a.f) / (b.f - a.f));
      return { f: frame, z: a.z + (b.z - a.z) * t, x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }
  }
  return keys[keys.length - 1];
};

// Hold a value from `from` until the next key: keys(f) helpers keep scene timelines terse
export const fade = (frame: number, a: number, b: number, c?: number, d?: number) =>
  c === undefined
    ? interpolate(frame, [a, b], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
    : interpolate(frame, [a, b, c, d!], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

export const typed = (frame: number, text: string, start: number, perChar = 3) =>
  text.slice(0, Math.max(0, Math.min(text.length, Math.floor((frame - start) / perChar) + 1)));

const BASE = 1.15; // 1440-wide window shown 1656px wide at zoom 1
const WIN_H = APP_H + CHROME_H;
const clampCentre = (c: number, half: number, size: number) => (size <= half * 2 ? size / 2 : Math.min(Math.max(c, half), size - half));

const BrowserChrome: React.FC<{ url: string }> = ({ url }) => {
  const [host, ...rest] = url.split('/');
  return (
    <div style={{ height: CHROME_H, background: '#f1f3f6', borderBottom: '1px solid #dfe3ea', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 8 }}>
      {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
        <div key={c} style={{ width: 12, height: 12, borderRadius: 6, background: c }} />
      ))}
      <div
        style={{
          marginLeft: 120,
          flex: 1,
          maxWidth: 760,
          height: 26,
          borderRadius: 13,
          background: '#fff',
          display: 'flex',
          alignItems: 'center',
          padding: '0 14px',
          fontFamily: RFONT,
          fontSize: 13,
          color: '#5f6368',
        }}
      >
        {host}
        <span style={{ color: '#9aa0a6' }}>{rest.length ? '/' + rest.join('/') : ''}</span>
      </div>
    </div>
  );
};

// A browser window holding an app canvas, a camera that pans and zooms over it, and a cursor
export const Stage: React.FC<{
  url: string;
  camera: Cam[];
  path?: Point[];
  clicks?: number[];
  hideCursor?: boolean;
  aiNote?: boolean;
  children: React.ReactNode;
}> = ({ url, camera, path, clicks = [], hideCursor, aiNote, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cam = cameraAt(frame, camera);
  const intro = spring({ frame, fps, config: { damping: 200 }, durationInFrames: 14 });
  const s = BASE * cam.z;
  const cx = clampCentre(cam.x, 960 / s, APP_W);
  const cy = clampCentre(cam.y + CHROME_H, 540 / s, WIN_H);
  const cur = path ? cursorAt(frame, path) : null;

  return (
    <AbsoluteFill style={{ background: '#eef2f8', overflow: 'hidden', opacity: intro }}>
      {/* data-qc: qc.py --layout reports the browser and the product's own UI apart from the video's text */}
      <div
        data-qc="product"
        style={{
          position: 'absolute',
          width: APP_W,
          height: WIN_H,
          transformOrigin: '0 0',
          transform: `translate(${960 - cx * s}px, ${540 - cy * s}px) scale(${s})`,
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: '0 30px 80px rgba(50,63,102,0.25), 0 0 0 1px rgba(50,63,102,0.08)',
        }}
      >
        <BrowserChrome url={url} />
        <div style={{ position: 'relative', width: APP_W, height: APP_H }}>
          {children}
          {cur && !hideCursor && (
            <div style={{ position: 'absolute', inset: 0, zIndex: 100 }}>
              <Cursor x={cur.x} y={cur.y} clicks={clicks} frame={frame} />
            </div>
          )}
        </div>
      </div>
      {aiNote && (
        <div style={{ position: 'absolute', left: 24, bottom: 16, fontFamily: RFONT, fontSize: 18, color: 'rgba(50,63,102,0.55)' }}>
          People shown are AI-generated
        </div>
      )}
    </AbsoluteFill>
  );
};

// Where a point is, used when a scene needs the cursor's position (hover states)
export const cursorPos = (frame: number, path: Point[]) => cursorAt(frame, path);
export const near = (a: { x: number; y: number }, b: { x: number; y: number }, rx = 60, ry = 18) =>
  Math.abs(a.x - b.x) < rx && Math.abs(a.y - b.y) < ry;
