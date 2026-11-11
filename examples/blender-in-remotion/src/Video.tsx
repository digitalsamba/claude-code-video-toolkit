/**
 * Blender in Remotion: two ways a Blender render becomes part of a Remotion video.
 *
 *   screen   lab       the terminal scene, flat (src/Screen.tsx)
 *   reveal   (3D)      the same scene is the laptop's screen: the camera pulls back from it    blender/room.json
 *   face     (3D)      the person at the laptop looks up
 *   act      record    the character on a transparent background, beside the beats it is acting  blender/desk.json
 *   close    identity  the title behind the character, confetti in front of it
 *
 * Nothing here is timed by hand: clip lengths come from public/blender/clips.json, and the beat list
 * reads blender/desk.json, the file Blender keyed the acting from.
 */
import React from 'react';
import { AbsoluteFill, Sequence, spring, useCurrentFrame, useVideoConfig } from 'remotion';
import { BlenderClip, BlenderManifest, blenderClip } from '../../../lib/components';
import desk from '../blender/desk.json';
import manifest from '../public/blender/clips.json';
import { FIELDS, FONT, MONO } from './fields';
import { REVEAL, Screen } from './Screen';

const M = manifest as BlenderManifest;
const A1 = blenderClip(M, 'A1_reveal');
const A2 = blenderClip(M, 'A2_face');
const B1 = blenderClip(M, 'B1_act');
const B2 = blenderClip(M, 'B2_cheer');

const FACE = REVEAL + A1.frames;
const ACT = FACE + A2.frames;
const CLOSE = ACT + B1.frames;
export const TOTAL = CLOSE + B2.frames;

const L = 120;
const rise = (frame: number, at: number, fps: number) => (frame < at ? 0 : Math.min(1, spring({ frame: frame - at, fps, config: { damping: 200 } })));

/** A solid tag over a 3D shot: what the viewer is looking at. */
const Tag: React.FC<{ at: number; children: React.ReactNode }> = ({ at, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = rise(frame, at, fps);
  return (
    <div style={{ position: 'absolute', left: L, bottom: 96, padding: '18px 28px', borderRadius: 8, background: FIELDS.lab.bg, color: '#ffffff', fontFamily: FONT, fontSize: 36, fontWeight: 500, opacity: t, transform: `translateY(${(1 - t) * 16}px)` }}>
      {children}
    </div>
  );
};

// ---------------------------------------------------------------- act: the beats, and the character acting them
type Beat = { at: string; do: string; face?: string; pose?: string; speed?: string };
const SHOWN = ['alert', 'expression', 'pose', 'shake_head', 'work', 'done'];
/** "B1_act+1.7" -> seconds into the shot (the spec's own time format, for this shot only). */
const seconds = (at: string) => (at.startsWith('B1_act') ? Number(at.slice('B1_act'.length) || 0) : NaN);
const BEATS = (desk.beats as Beat[])
  .map((b) => ({ t: seconds(b.at), verb: b.do, arg: b.face ?? b.pose ?? b.speed ?? '' }))
  .filter((b) => !Number.isNaN(b.t) && SHOWN.includes(b.verb));

const ActScene: React.FC = () => {
  const f = FIELDS.record;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const head = rise(frame, 2, fps);
  const now = frame / fps;
  const current = BEATS.reduce((last, b, i) => (now >= b.t ? i : last), -1);
  const row = 62;
  return (
    <AbsoluteFill style={{ background: f.bg, fontFamily: FONT }}>
      <div style={{ position: 'absolute', left: L, top: 84, color: f.ink, opacity: head, transform: `translateY(${(1 - head) * 20}px)` }}>
        <div style={{ fontSize: 72, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1 }}>Acting is a list of verbs</div>
        <div style={{ fontSize: 36, marginTop: 14, color: f.muted }}>Each beat is a verb and a time. Blender keys the motion.</div>
      </div>
      <div style={{ position: 'absolute', left: L, top: 290, width: 760, background: '#ffffff', borderRadius: 12, padding: '36px 44px', boxSizing: 'border-box', opacity: head }}>
        <div style={{ fontFamily: MONO, fontSize: 28, color: f.muted, marginBottom: 18 }}>blender/desk.json</div>
        {BEATS.map((b, i) => {
          const on = i === current;
          const done = i < current;
          return (
            <div key={i} style={{ height: row, display: 'flex', alignItems: 'center', fontFamily: MONO, fontSize: 30, color: on ? f.accent : done ? f.ink : '#737b8c', fontWeight: on ? 700 : 400 }}>
              <span style={{ width: 8, height: 36, marginRight: 20, borderRadius: 4, background: on ? f.accent : 'transparent' }} />
              <span style={{ width: 120 }}>{b.t.toFixed(1)} s</span>
              <span style={{ width: 230 }}>{b.verb}</span>
              <span>{b.arg}</span>
            </div>
          );
        })}
      </div>
      {/* the clip's bottom edge is the frame's, so the desk legs run off screen instead of stopping in mid-air */}
      <BlenderClip clip={B1} volume={1} style={{ left: 560, top: 270, width: 1440 }} />
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- close: type behind the character, confetti in front
const CloseScene: React.FC = () => {
  const f = FIELDS.identity;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = rise(frame, 4, fps);
  const s = rise(frame, 16, fps);
  return (
    <AbsoluteFill style={{ background: f.bg, fontFamily: FONT, color: f.ink }}>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 84, textAlign: 'center', opacity: t, transform: `translateY(${(1 - t) * 20}px)` }}>
        <div style={{ fontSize: 112, fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1 }}>3D that takes direction</div>
      </div>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 232, textAlign: 'center', fontFamily: MONO, fontSize: 30, color: f.muted, opacity: s }}>
        uv run tools/blender_render.py render spec.json --clips --alpha
      </div>
      <BlenderClip clip={B2} volume={1} />
    </AbsoluteFill>
  );
};

export const BlenderInRemotion: React.FC = () => (
  <AbsoluteFill style={{ background: FIELDS.lab.bg }}>
    <Sequence durationInFrames={REVEAL}><Screen sound /></Sequence>
    <Sequence from={REVEAL} durationInFrames={A1.frames}>
      <BlenderClip clip={A1} volume={1} />
      <Tag at={78}>The scene you were watching is on the laptop</Tag>
    </Sequence>
    <Sequence from={FACE} durationInFrames={A2.frames}>
      <BlenderClip clip={A2} volume={1} />
      <Tag at={8}>Rigged, lit and filmed from a JSON spec</Tag>
    </Sequence>
    <Sequence from={ACT} durationInFrames={B1.frames}><ActScene /></Sequence>
    <Sequence from={CLOSE} durationInFrames={B2.frames}><CloseScene /></Sequence>
  </AbsoluteFill>
);
