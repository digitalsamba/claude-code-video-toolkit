/**
 * Blender renders as elements of a Remotion video.
 *
 * `tools/blender_render.py render spec.json --clips [--alpha] --to public/blender` writes one clip per shot of a
 * shot spec (.claude/skills/blender) and a `clips.json` manifest. `blenderClip()` looks a shot up in the manifest
 * and `<BlenderClip>` plays it: once, then it holds its last frame, with the shot's own sound effects (already
 * on the frames the motion was keyed on). A clip rendered with `--alpha` has a transparent background, so the
 * character sits on the scene's own field, with type behind it and in front of it.
 *
 * The spec's `fps` must be the composition's: a clip plays frame for frame, so its acting stays on the frames
 * the spec put it on, and `clip.event('alert')` is the frame to hang a caption or a cut on.
 *
 * @example
 * ```tsx
 * import manifest from '../public/blender/clips.json';
 * import { BlenderClip, blenderClip } from '../../../lib/components';
 *
 * const act = blenderClip(manifest, 'B1_act');
 * <Sequence from={200} durationInFrames={act.frames}>
 *   <BlenderClip clip={act} style={{ left: 760, top: 80, width: 1160 }} />
 * </Sequence>
 * const SHOCK = 200 + act.event('alert');   // the frame the alert lands, in the composition
 * ```
 *
 * `crop` shows one part of the clip, in the clip's own pixels, scaled to fill the box: a presenter's head and
 * chest as a small inset while the camera is somewhere else. `background` is a picture of the scene the clip
 * was placed on, the size of the clip's frame; it is drawn behind the clip and cropped with it, so a
 * transparent character keeps its own backdrop. Nothing is drawn outside the box.
 *
 * @example
 * ```tsx
 * <BlenderClip clip={act} volume={0} crop={{ x: 420, y: 60, w: 380, h: 338 }} background="board/behind-him.png"
 *   style={{ left: 1620, top: 810, width: 270 }} />
 * ```
 */

import React from 'react';
import { Audio, Freeze, Img, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';

/** One shot in clips.json. */
export type BlenderClipEntry = {
  id: string;
  file: string;
  /** Rendered with a transparent background (--alpha) */
  alpha?: boolean;
  frames: number;
  seconds: number;
  /** The clip's own rate and size; a manifest written before these were per clip has them at the top only */
  fps?: number;
  resolution?: number[];
  /** The spec the clip was rendered from, as "folder/file.json" */
  spec?: string;
  /** The shot's effects as a WAV, if it has any */
  sound?: string;
  /** Sound-worthy moments, in frames from the clip's start */
  events: { frame: number; kind: string }[];
};

/** clips.json, as written by tools/blender_render.py --clips. Several specs can share one, so each clip carries
 * its own fps and resolution; the top-level pair is the last render's, kept for older manifests. */
export type BlenderManifest = { fps: number; resolution: number[]; clips: BlenderClipEntry[] };

export type BlenderClipInfo = {
  id: string;
  /** Path under public/ */
  src: string;
  sound?: string;
  frames: number;
  fps: number;
  width: number;
  height: number;
  transparent: boolean;
  events: { frame: number; kind: string }[];
  /** Clip frame of the nth event of a kind ('alert', 'gulp', 'tap', 'success', 'celebrate'...). Throws if absent. */
  event: (kind: string, nth?: number) => number;
};

/** A shot of the manifest, ready for `<BlenderClip>`. `dir` is the folder under public/ the clips were copied to. */
export const blenderClip = (manifest: BlenderManifest, id: string, dir = 'blender'): BlenderClipInfo => {
  const c = manifest.clips.find((x) => x.id === id);
  if (!c) throw new Error(`blenderClip: no shot "${id}" in clips.json (have: ${manifest.clips.map((x) => x.id).join(', ')})`);
  const [width, height] = c.resolution ?? manifest.resolution;
  return {
    id,
    src: `${dir}/${c.file}`,
    sound: c.sound ? `${dir}/${c.sound}` : undefined,
    frames: c.frames,
    fps: c.fps ?? manifest.fps,
    width,
    height,
    transparent: c.alpha ?? /\.webm$/.test(c.file),
    events: c.events,
    event: (kind, nth = 1) => {
      const hits = c.events.filter((e) => e.kind === kind);
      if (hits.length < nth) {
        throw new Error(`blenderClip: shot "${id}" has no ${kind} event #${nth} (has: ${[...new Set(c.events.map((e) => e.kind))].join(', ') || 'none'})`);
      }
      return hits[nth - 1].frame;
    },
  };
};

export type BlenderClipProps = {
  clip: BlenderClipInfo;
  /** Volume of the clip's own effects (default 0.5); 0 leaves them out */
  volume?: number;
  /** Start this many frames into the clip */
  trim?: number;
  /** Style for the box. With no size given it fills the frame; give `width` and it keeps the clip's aspect
   * (the crop's, when there is one). */
  style?: React.CSSProperties;
  /** Show only this part of the clip, in the clip's own pixels, scaled to fill the box. */
  crop?: BlenderCrop;
  /** A picture under public/ the size of the clip's frame, drawn behind the clip (and cropped with it). */
  background?: string;
  /** Style for the clip's picture inside the box: a `filter: drop-shadow(...)` here follows the character,
   * where one on `style` would follow the box. */
  clipStyle?: React.CSSProperties;
};

export type BlenderCrop = { x: number; y: number; w: number; h: number };

/** Where the whole clip sits inside a box that shows only `crop` of it, as CSS percentages of the box. */
export const cropPlacement = (clip: { id: string; width: number; height: number }, crop: BlenderCrop): React.CSSProperties => {
  const { x, y, w, h } = crop;
  if (!(w > 0 && h > 0) || x < 0 || y < 0 || x + w > clip.width || y + h > clip.height) {
    throw new Error(`BlenderClip "${clip.id}": crop ${JSON.stringify(crop)} is not inside the clip (${clip.width}x${clip.height} pixels)`);
  }
  return {
    position: 'absolute',
    left: `${(-x / w) * 100}%`,
    top: `${(-y / h) * 100}%`,
    width: `${(clip.width / w) * 100}%`,
    height: `${(clip.height / h) * 100}%`,
  };
};

/**
 * Plays a Blender clip once and holds its last frame. Wrap it in a <Sequence> that starts where the shot does.
 * Opaque clips fill the frame; transparent ones are usually given a `left`, `top` and `width`.
 */
export const BlenderClip: React.FC<BlenderClipProps> = ({ clip, volume = 0.5, trim = 0, style, crop, background, clipStyle }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (clip.fps !== fps) {
    throw new Error(`BlenderClip "${clip.id}" was rendered at ${clip.fps} fps and the composition is ${fps} fps: set "fps": ${fps} in the shot spec and render the clips again`);
  }
  const sized = style && (style.width !== undefined || style.height !== undefined);
  const shown = crop ?? { w: clip.width, h: clip.height };
  const box: React.CSSProperties = sized
    ? { position: 'absolute', aspectRatio: `${shown.w} / ${shown.h}`, ...style }
    : { position: 'absolute', inset: 0, ...style };
  if (crop || background) box.overflow = 'hidden';
  // the whole clip, placed so that the crop fills the box
  const whole: React.CSSProperties = crop ? cropPlacement(clip, crop) : { position: 'absolute', inset: 0 };
  const fill: React.CSSProperties = { position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' };
  const at = Math.max(0, Math.min(frame + trim, clip.frames - 1));
  return (
    <div style={box}>
      <div style={whole}>
        {background ? <Img src={/^(https?:|data:)/.test(background) ? background : staticFile(background)} style={fill} /> : null}
        <Freeze frame={at} active>
          <OffthreadVideo src={staticFile(clip.src)} muted transparent={clip.transparent} style={{ ...fill, ...clipStyle }} />
        </Freeze>
      </div>
      {clip.sound && volume > 0 && frame + trim < clip.frames ? (
        <Audio src={staticFile(clip.sound)} volume={volume} trimBefore={trim} />
      ) : null}
    </div>
  );
};
