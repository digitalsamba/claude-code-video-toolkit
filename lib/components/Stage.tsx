/**
 * Browser Stage for Synthetic UI Demos
 *
 * A browser window holding a fixed-size app canvas, a camera that zooms and pans over
 * it, and a scripted cursor on top. The screens inside are static React rebuilt from
 * the product's source (see the synthetic-ui-demo skill), laid out absolutely on the
 * canvas so click targets are known coordinates.
 *
 * Camera keyframes are in canvas pixels: `x`/`y` is the point to centre, `z` the zoom
 * (1 = the whole window, at `fit` of the frame width). The camera clamps so it never
 * shows past the window's edges. The cursor renders at zIndex 100 inside the canvas.
 *
 * @example
 * ```tsx
 * import { Stage } from '../../../../lib/components';
 *
 * <Stage
 *   url="app.example.com/settings"
 *   camera={[
 *     { f: 0, z: 1, x: 720, y: 405 },
 *     { f: 30, z: 1, x: 720, y: 405 },
 *     { f: 55, z: 1.6, x: 420, y: 300 },
 *   ]}
 *   path={[{ f: 0, x: 900, y: 600 }, { f: 40, x: 412, y: 318 }]}
 *   clicks={[46]}
 *   aiNote
 * >
 *   <SettingsScreen />
 * </Stage>
 * ```
 */

import React from 'react';
import { AbsoluteFill, Easing, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { Cursor, CursorPoint, cursorAt } from './Cursor';

export type CameraKeyframe = { f: number; z: number; x: number; y: number };

const ease = Easing.bezier(0.65, 0, 0.35, 1);

/** Camera at a frame, eased between keyframes */
export const cameraAt = (frame: number, keys: CameraKeyframe[]): CameraKeyframe => {
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

/** Text revealed one character per `perChar` frames from `start`, for typing into fields */
export const typed = (frame: number, text: string, start: number, perChar = 3) =>
  text.slice(0, Math.max(0, Math.min(text.length, Math.floor((frame - start) / perChar) + 1)));

const clampCentre = (c: number, half: number, size: number) =>
  size <= half * 2 ? size / 2 : Math.min(Math.max(c, half), size - half);

const SYSTEM_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

const BrowserChrome: React.FC<{ url: string; height: number; fontFamily: string }> = ({ url, height, fontFamily }) => {
  const [host, ...rest] = url.split('/');
  return (
    <div
      style={{
        height,
        background: '#f1f3f6',
        borderBottom: '1px solid #dfe3ea',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: 8,
      }}
    >
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
          fontFamily,
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

export interface StageProps {
  /** Address bar text, e.g. "app.example.com/settings" (host shown darker than the path) */
  url: string;
  /** Camera keyframes in canvas pixels */
  camera: CameraKeyframe[];
  /** Cursor keyframes in canvas pixels; omit for no cursor */
  path?: CursorPoint[];
  /** Frames at which the cursor clicks */
  clicks?: number[];
  hideCursor?: boolean;
  /** Bottom-left disclosure. `true` shows "People shown are AI-generated"; a string replaces the text */
  aiNote?: boolean | string;
  /** App canvas size in px (default 1440x810) */
  canvasWidth?: number;
  canvasHeight?: number;
  /** Browser chrome height in px (default 40); 0 hides the chrome */
  chromeHeight?: number;
  /** Window width as a fraction of the frame width at zoom 1 (default 0.8625) */
  fit?: number;
  /** Backdrop behind the window */
  background?: string;
  /** Font for the address bar and AI note */
  fontFamily?: string;
  /** Click ripple colour */
  rippleColor?: string;
  /** Fade the stage in over its first frames (default true) */
  fadeIn?: boolean;
  /**
   * The window's edge (default: a soft drop shadow). video-design has no shadows at rest, so a designed video
   * passes a hairline instead, e.g. `0 0 0 1px rgba(17,17,17,0.12)`.
   */
  windowShadow?: string;
  children: React.ReactNode;
}

export const Stage: React.FC<StageProps> = ({
  url,
  camera,
  path,
  clicks = [],
  hideCursor,
  aiNote,
  canvasWidth = 1440,
  canvasHeight = 810,
  chromeHeight = 40,
  fit = 0.8625,
  background = 'linear-gradient(135deg, #f8fafe 0%, #EBF1FC 100%)',
  fontFamily = SYSTEM_FONT,
  rippleColor,
  fadeIn = true,
  windowShadow = '0 30px 80px rgba(50,63,102,0.25), 0 0 0 1px rgba(50,63,102,0.08)',
  children,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const cam = cameraAt(frame, camera);
  const intro = fadeIn ? spring({ frame, fps, config: { damping: 200 }, durationInFrames: 14 }) : 1;
  const winH = canvasHeight + chromeHeight;
  const s = ((width * fit) / canvasWidth) * cam.z;
  const cx = clampCentre(cam.x, width / 2 / s, canvasWidth);
  const cy = clampCentre(cam.y + chromeHeight, height / 2 / s, winH);
  const cur = path ? cursorAt(frame, path) : null;

  return (
    <AbsoluteFill style={{ background, overflow: 'hidden', opacity: intro }}>
      {/* data-qc: qc.py --layout reports the browser and the product's own UI apart from the video's text */}
      <div
        data-qc="product"
        style={{
          position: 'absolute',
          width: canvasWidth,
          height: winH,
          transformOrigin: '0 0',
          transform: `translate(${width / 2 - cx * s}px, ${height / 2 - cy * s}px) scale(${s})`,
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: windowShadow,
        }}
      >
        {chromeHeight > 0 && <BrowserChrome url={url} height={chromeHeight} fontFamily={fontFamily} />}
        <div style={{ position: 'relative', width: canvasWidth, height: canvasHeight, overflow: 'hidden' }}>
          {children}
          {cur && !hideCursor && (
            <div style={{ position: 'absolute', inset: 0, zIndex: 100 }}>
              <Cursor x={cur.x} y={cur.y} clicks={clicks} frame={frame} rippleColor={rippleColor} />
            </div>
          )}
        </div>
      </div>
      {aiNote && (
        <div style={{ position: 'absolute', left: 24, bottom: 16, fontFamily, fontSize: 18, color: 'rgba(50,63,102,0.8)' }}>
          {typeof aiNote === 'string' ? aiNote : 'People shown are AI-generated'}
        </div>
      )}
    </AbsoluteFill>
  );
};
