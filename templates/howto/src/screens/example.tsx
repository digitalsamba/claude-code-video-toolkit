/**
 * An example `screen` scene and the kit it draws from: a made-up app's notification settings, where the cursor
 * turns on a weekly summary and saves. It is here to copy from. A real project rebuilds its own product's screens
 * from the product's source in src/kit/ (the `synthetic-ui-demo` skill) and keeps its scenes in src/screens/.
 *
 * Studio shows it as `Screen-example` and `Screen-example-short`, timed by EXAMPLE_CUES instead of narration.
 * Delete this file and those two compositions in src/Root.tsx when your own screens are in.
 */
import React from 'react';
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from 'remotion';
import { boxMid, cursorAt, near, typed, visitPath, type CursorPoint } from '../../../../lib/components';
import { SCENES } from '../howto/scenes';
import { FIELDS, layoutFor, SANS } from '../howto/theme';
import type { Cue, ScreenProps } from '../howto/types';

// ---------------------------------------------------------------- the kit: one page, laid out on a 1440 x 810 canvas

const INK = '#1d2230';
const DIM = '#667085';
const LINE = '#e4e7ec';
const BLUE = '#2f6fed';

/** Where the controls are, in canvas pixels: the scene aims the cursor at these */
export const PAGE = {
  email: { x: 300, y: 250, w: 460, h: 44 },
  weekly: { x: 300, y: 380, w: 352, h: 40 },
  toggle: { x: 600, y: 386, w: 52, h: 28 },
  save: { x: 300, y: 500, w: 120, h: 44 },
};

const Toggle: React.FC<{ on: number }> = ({ on }) => (
  <div style={{ position: 'absolute', left: PAGE.toggle.x, top: PAGE.toggle.y, width: PAGE.toggle.w, height: PAGE.toggle.h, borderRadius: 14, background: on > 0.5 ? BLUE : '#cfd4dc' }}>
    <div style={{ position: 'absolute', top: 3, left: 3 + on * 24, width: 22, height: 22, borderRadius: 11, background: '#fff' }} />
  </div>
);

/** The settings page as it looks on a frame: everything that changes comes in as a prop */
export const SettingsPage: React.FC<{ email: string; caret: boolean; weekly: number; hoverSave: boolean; press: number; saved: number }> = ({ email, caret, weekly, hoverSave, press, saved }) => (
  <AbsoluteFill style={{ background: '#fff', color: INK, fontSize: 16, fontFamily: SANS }}>
    <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 220, borderRight: `1px solid ${LINE}`, padding: '32px 24px' }}>
      {['Profile', 'Notifications', 'Billing', 'Team'].map((item) => (
        <div key={item} style={{ padding: '10px 12px', borderRadius: 6, marginBottom: 4, background: item === 'Notifications' ? '#eef3fe' : 'transparent', color: item === 'Notifications' ? BLUE : DIM, fontWeight: item === 'Notifications' ? 600 : 400 }}>
          {item}
        </div>
      ))}
    </div>
    <div style={{ position: 'absolute', left: 300, top: 64, fontSize: 28, fontWeight: 600 }}>Notifications</div>
    <div style={{ position: 'absolute', left: 300, top: 216, color: DIM }}>Send notifications to</div>
    <div style={{ position: 'absolute', left: PAGE.email.x, top: PAGE.email.y, width: PAGE.email.w, height: PAGE.email.h, border: `1px solid ${caret ? BLUE : LINE}`, borderRadius: 6, padding: '0 14px', display: 'flex', alignItems: 'center' }}>
      {email}
      {caret && <span style={{ width: 2, height: 20, background: INK, marginLeft: 1 }} />}
    </div>
    <div style={{ position: 'absolute', left: PAGE.weekly.x, top: PAGE.weekly.y + 8 }}>A weekly summary of what changed</div>
    <Toggle on={weekly} />
    <div style={{ position: 'absolute', left: PAGE.save.x, top: PAGE.save.y, width: PAGE.save.w, height: PAGE.save.h, borderRadius: 6, background: hoverSave ? '#245bd0' : BLUE, color: '#fff', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', transform: `scale(${1 - 0.04 * press})` }}>
      Save
    </div>
    <div style={{ position: 'absolute', left: 440, top: PAGE.save.y + 12, color: '#12805c', opacity: saved }}>Saved</div>
  </AbsoluteFill>
);

// ---------------------------------------------------------------- the scene: every frame from a cue

type T = { email: number; weekly: number; save: number };
const EMAIL = 'team@example.com';

const Settings: React.FC<{ T: T; path: CursorPoint[] }> = ({ T, path }) => {
  const frame = useCurrentFrame();
  const cur = cursorAt(frame, path);
  return (
    <SettingsPage
      email={frame >= T.email + 4 ? typed(frame, EMAIL, T.email + 4, 2) : ''}
      caret={frame >= T.email && frame < T.weekly && Math.floor(frame / 15) % 2 === 0}
      weekly={interpolate(frame, [T.weekly, T.weekly + 6], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })}
      hoverSave={near(cur, boxMid(PAGE.save), 60, 22)}
      press={interpolate(frame, [T.save, T.save + 3, T.save + 9], [0, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })}
      saved={interpolate(frame, [T.save + 8, T.save + 16], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })}
    />
  );
};

/**
 * In src/episode.ts: `{ id: 'settings', kind: 'screen', chapter: 1, props: settings, short: settingsShort }`,
 * narrated in public/audio/scenes/NNa_settings.mp3.
 */
export const settings = (cue: Cue): ScreenProps => {
  const t: T = { email: cue('where they go'), weekly: cue('weekly summary'), save: cue('save') };
  const { path, clicks } = visitPath({ x: 900, y: 620 }, [
    { at: t.email, to: boxMid(PAGE.email, -120), click: true },
    { at: t.weekly, to: boxMid(PAGE.toggle), click: true },
    { at: t.save, to: boxMid(PAGE.save), click: true },
  ]);
  return {
    screen: <Settings T={t} path={path} />,
    url: 'app.example.com/settings/notifications',
    path,
    clicks,
    camera: [
      { f: 0, z: 1, x: 720, y: 405 },
      { f: t.email - 20, z: 1, x: 720, y: 405 },
      { f: t.email + 6, z: 1.4, x: 780, y: 330 },
      { f: t.save + 30, z: 1.4, x: 780, y: 330 },
    ],
  };
};

/** In a Short the episode's camera shows too much: drop it, and the camera follows the cursor close up */
export const settingsShort = (): Partial<ScreenProps> => ({ camera: undefined });

// ---------------------------------------------------------------- a preview with no narration

/** Stand-in for the narration while there is none: the frame each phrase would be said on */
export const EXAMPLE_CUES: Record<string, number> = { 'where they go': 40, 'weekly summary': 110, save: 160 };
const fakeCue: Cue = (phrase) => EXAMPLE_CUES[phrase] ?? 0;
export const EXAMPLE_FRAMES = 220;

/** The scene on the work field, as an episode (16:9) or a Short (9:16) would show it */
export const ScreenExample: React.FC = () => {
  const { width, height } = useVideoConfig();
  const f = FIELDS.work;
  const Screen = SCENES.screen;
  const L = layoutFor(width, height);
  const props = { ...settings(fakeCue), ...(L.vertical ? settingsShort() : {}) };
  return (
    <AbsoluteFill style={{ background: f.ground }}>
      <Screen L={L} f={f} {...props} />
    </AbsoluteFill>
  );
};
