// The presenter and the room shots: Blender clips from public/blender/, played when they are there.
//
// `uv run tools/lecture.py render <project>` writes the presenter's clips and clips.json. Two kinds, told apart by id:
//   * room shots, ids starting "E" (E1_door, E2_table, ...): opaque, 1920x1080, played back to back from frame 0
//     in the order of their ids (a shot rendered again goes to the end of clips.json); the last holds until the
//     board camera takes over. Optional: a lecture can start at the board.
//   * presenter clips, one a segment (P01, P02, ...): transparent, covering `staging.window` of the board camera's
//     frame, each from `rules.lead_frames` before its segment to the next clip's start, so the clips join.
// Until a clip exists nothing is drawn for it, so the video renders at any stage.
import React, { useEffect, useState } from 'react';
import { AbsoluteFill, Audio, Sequence, continueRender, delayRender, getStaticFiles, interpolate, staticFile, useCurrentFrame } from 'remotion';
import { BlenderClip, blenderClip, type BlenderManifest } from '../components/BlenderClip';
import { Stroke, box } from './chalk';
import defaults from './defaults.json';
import type { Lecture, LectureTimeline, Sound } from './timeline';

export const useManifest = (): BlenderManifest | null => {
  const there = getStaticFiles().some((f) => f.name === 'blender/clips.json');
  const [manifest, setManifest] = useState<BlenderManifest | null>(null);
  const [handle] = useState(() => (there ? delayRender('blender/clips.json') : null));
  useEffect(() => {
    if (handle === null) return;
    fetch(staticFile('blender/clips.json'))
      .then((r) => r.json())
      .then((m) => setManifest(m))
      .catch(() => undefined)
      .finally(() => continueRender(handle));
  }, [handle]);
  return manifest;
};

const has = (file: string) => getStaticFiles().some((f) => f.name === `blender/${file}`);

/** The share of the frame the presenter's clips cover, in board pixels (staging.window is left, bottom, right, top). */
export const presenterBox = (L: LectureTimeline) => {
  const [x0, y0, x1, y1] = L.staging.window;
  return { left: Math.round(x0 * 1920), top: Math.round((1 - y1) * 1080), width: Math.round((x1 - x0) * 1920), height: Math.round((y1 - y0) * 1080) };
};

/** The lecture hall's room shots, frames 0 to the first presenter clip. A dark card stands in until they exist. */
export const Room: React.FC<{ L: LectureTimeline }> = ({ L }) => {
  const manifest = useManifest();
  const frame = useCurrentFrame();
  const shots = (manifest?.clips ?? []).filter((c) => /^E/.test(c.id) && has(c.file)).sort((x, y) => x.id.localeCompare(y.id));
  if (!shots.length) {
    return (
      <AbsoluteFill style={{ background: '#161b1a', color: '#5d6b66', font: '28px sans-serif', alignItems: 'center', justifyContent: 'center' }}>
        the lecture hall (Blender room shots, public/blender E*)
      </AbsoluteFill>
    );
  }
  let at = 0;
  // the opening fades in from black (its sound, the room tone and the latch, has started before it)
  const op = L.tl.opening;
  const seen = op ? interpolate(frame, [op.fadeFrom, op.fadeFrom + op.fadeFrames], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) : 1;
  return (
    <AbsoluteFill style={{ background: '#000', opacity: seen }}>
      {shots.map((c, i) => {
        const from = at;
        at += c.frames;
        const last = i === shots.length - 1;
        return (
          <Sequence key={c.id} from={from} durationInFrames={last ? Math.max(1, L.first - from) : c.frames}>
            <BlenderClip clip={blenderClip(manifest!, c.id)} volume={0.5} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

/** Closer shots (a segment's "closes"): opaque clips from a second camera, ids starting "C", each cut to for its own
 * frames over the board and everything on it. One that is not rendered yet draws nothing: the video stays wide. */
export const Closes: React.FC<{ L: LectureTimeline }> = ({ L }) => {
  const manifest = useManifest();
  const frame = useCurrentFrame();
  const c = (L.tl.closes ?? []).find((x) => frame >= x.from && frame < x.to);
  const entry = c && manifest?.clips.find((x) => x.id === c.id);
  if (!c || !manifest || !entry || !has(entry.file)) return null;
  return (
    <Sequence from={c.from} durationInFrames={c.to - c.from} layout="none">
      <BlenderClip clip={blenderClip(manifest, c.id)} volume={0} style={{ left: 0, top: 0, width: 1920, height: 1080 }} />
    </Sequence>
  );
};

/** The presenter's shadow on the board (staging.shadow), `k` times its size: after an opening it fades in from the
 * cut, since the room shot shows none on the board and one switched on at the cut read as an outline round the body. */
export const shadow = (L: LectureTimeline, frame: number, k = 1) => {
  const sh = L.staging.shadow;
  const on = L.tl.opening
    ? interpolate(frame, [L.first, L.first + sh.fade * L.tl.fps], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
    : 1;
  return `drop-shadow(${sh.x * k}px ${sh.y * k}px ${sh.blur * k}px rgba(0,0,0,${(sh.alpha * on).toFixed(3)}))`;
};

/** The presenter in front of the board (inside the camera, above the chalk). A short passes its own `box` (where the
 * clip goes on its frame, `k` times the board's size) and keeps them in frame through the windows. */
export const Presenter: React.FC<{ L: LectureTimeline; box?: { left: number; top: number; width: number; height: number }; k?: number; offset?: number; hidden?: [number, number][] }> = ({ L, box, k = 1, offset = 0, hidden = [] }) => {
  const manifest = useManifest();
  // `offset`: the lecture frame of this composition's frame 0 (a short). Not a Sequence shifted back by it: under one
  // shifted thousands of frames back the clip drew nothing (BlenderClip's Freeze counts from the outer sequence)
  const frame = useCurrentFrame() + offset;
  if (!manifest) return null;
  // In a window the camera is on the left of the board and the presenter is out of frame; a hand could still reach into
  // its edge, so they are not drawn there (back in over the first frames of the move out).
  const win = box ? undefined : L.tl.windows.find((w) => frame >= w.from && frame < w.to + 8);
  // `hidden`: lecture frames a short gives the whole frame to something else (a pane that takes over), faded over 6
  const away = hidden.reduce((m, [a, b]) => Math.min(m, interpolate(frame, [a - 6, a, b, b + 6], [1, 0, 0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })), 1);
  const seen = (win ? Math.max(0, (frame - win.to) / 8) : 1) * away;
  if (seen <= 0) return null;
  const BOX = box ?? presenterBox(L);
  const filter = shadow(L, frame, k);
  return (
    <>
      {L.clips.map(({ id, from, until }) => {
        const c = manifest.clips.find((x) => x.id === id);
        if (!c || !has(c.file) || frame < from || frame >= until) return null;
        return (
          <Sequence key={id} from={from - offset} durationInFrames={until - from} layout="none">
            <BlenderClip clip={blenderClip(manifest, id)} volume={0.35}
              style={{ ...BOX, opacity: seen, filter }} />
          </Sequence>
        );
      })}
    </>
  );
};

/** Where the inset is on screen and the part of the board (their head and chest) it shows. The head comes from
 * `tools/lecture.py check` (timeline.json presenter.head_px), so it follows the presenter's size. */
export const insetRegion = (L: LectureTimeline) => {
  const inset = defaults.inset;
  const p = L.tl.presenter;
  const BOX = presenterBox(L);
  const head = p ? { u: p.head_px[0], v: p.head_px[1] } : { u: BOX.left + BOX.width * 0.6, v: BOX.top + 160 };
  const top = p ? p.box_px[1] : BOX.top + 60;
  const w = Math.max(200, 2 * (head.v - top) + 30); // (head and shoulders: twice from the head to the crown, and a margin)
  const h = (w * inset.height) / inset.width;
  return { inset, region: { u: Math.round(head.u - w / 2), v: Math.round(top - 14), w: Math.round(w), h: Math.round(h) } };
};

/** While the camera is inside a window the presenter is out of frame: they keep talking, small, in a chalk-framed inset. */
export const Inset: React.FC<{ L: LectureTimeline }> = ({ L }) => {
  const manifest = useManifest();
  const frame = useCurrentFrame();
  const win = L.tl.windows.find((w) => frame >= w.from - 10 && frame < w.to + 10);
  if (!manifest || !win) return null;
  const c = L.clips.find((x) => frame >= x.from && frame < x.until);
  const entry = c && manifest.clips.find((x) => x.id === c.id);
  if (!c || !entry || !has(entry.file)) return null;
  const o = interpolate(frame, [win.from - 10, win.from, win.to, win.to + 10], [0, 1, 1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const clip = blenderClip(manifest, c.id);
  const BOX = presenterBox(L);
  const { inset, region } = insetRegion(L);
  const k = clip.width / BOX.width;
  const x = Math.max(0, (region.u - BOX.left) * k);
  const y = Math.max(0, (region.v - BOX.top) * k);
  const crop = { x, y, w: Math.min(region.w * k, clip.width - x), h: Math.min(region.h * k, clip.height - y) };
  const shrink = inset.width / region.w;
  const bg = getStaticFiles().some((f) => f.name === 'board/bg-presenter.png') ? 'board/bg-presenter.png' : undefined;
  return (
    <div style={{ position: 'absolute', left: 0, top: 0, width: 1920, height: 1080, opacity: o }}>
      {/* their clip, cropped, over the part of the bare board it stands in front of (`tools/lecture.py bg`) */}
      <Sequence from={c.from} durationInFrames={c.until - c.from} layout="none">
        <BlenderClip clip={clip} volume={0} crop={crop} background={bg}
          clipStyle={{ filter: shadow(L, frame, shrink) }}
          style={{ ...inset, borderRadius: 4, background: '#2a3633' }} />
      </Sequence>
      <div style={{ position: 'absolute', left: 0, top: 0, width: 1920, height: 1080, filter: 'url(#chalkFx)', pointerEvents: 'none' }}>
        <Stroke d={box(inset.left - 6, inset.top - 6, inset.width + 12, inset.height + 12, 5)} at={win.from - 10} dur={10} width={5} />
      </div>
    </div>
  );
};

/**
 * Sounds for what the clips' own sound does not carry (the chalk going down on the table, hands dusted, a cup):
 * each is laid on the event frame its clip recorded when it was rendered (clips.json), never on a typed frame, so
 * a clip rendered again moves its sound with it. Which event gets which sound is lecture.json "sounds" (defaults
 * in lib/lecture/defaults.json); check them on the render with tools/sync_check.py (a landing is `impact`).
 */
export const PropSounds: React.FC<{ L: LectureTimeline; lecture: Lecture }> = ({ L, lecture }) => {
  const manifest = useManifest();
  if (!manifest) return null;
  const sounds = { ...defaults.sounds, ...(lecture.sounds ?? {}) } as Record<string, Sound>;
  // the room shots' events (the latch, the door, the footsteps of the walk in) take the opening's sounds
  const opening = typeof lecture.opening === 'object' ? lecture.opening : {};
  const roomSounds = { ...sounds, ...defaults.opening.sounds, ...(opening.sounds ?? {}) } as Record<string, Sound>;
  // a room shot starts where the shots before it (by id) end, as Room plays them; a presenter clip at its span
  const starts: Record<string, number> = {};
  let t = 0;
  for (const c of manifest.clips.filter((x) => /^E/.test(x.id)).sort((x, y) => x.id.localeCompare(y.id))) {
    starts[c.id] = t;
    t += c.frames;
  }
  for (const c of L.clips) starts[c.id] = c.from;
  const laid: { key: string; at: number; s: Sound; file: string }[] = [];
  for (const c of manifest.clips) {
    if (starts[c.id] === undefined) continue;
    const count: Record<string, number> = {};
    for (const e of c.events ?? []) {
      const s = (/^E/.test(c.id) ? roomSounds : sounds)[e.kind];
      if (!s) continue;
      const k = (count[e.kind] = (count[e.kind] ?? -1) + 1);
      const file = s.files ? s.files[k % s.files.length] : s.file;  // (in turn: no two footsteps alike)
      if (file && getStaticFiles().some((f) => f.name === file)) laid.push({ key: `${c.id}-${e.kind}-${e.frame}`, at: starts[c.id] + e.frame, s, file });
    }
  }
  return (
    <>
      {laid.map(({ key, at, s, file }) => (
        <Sequence key={key} from={Math.max(0, at + (s.offset ?? 0))} durationInFrames={s.frames ?? 20} layout="none">
          <Audio src={staticFile(file)} volume={s.volume ?? 1} />
        </Sequence>
      ))}
    </>
  );
};
