import { useCurrentFrame } from 'remotion';
import { story } from './config/story-config';
import type { Scene } from './config/types';

export interface TimedScene extends Scene {
  index: number;
  from: number;
  frames: number;
}

/** A run of consecutive scenes that share one mounted visual. */
export interface VisualGroup {
  key: string;
  from: number;
  frames: number;
  /** Scene boundaries relative to the group start: the visual's beats. */
  beats: { from: number; frames: number; scene: TimedScene }[];
}

const fps = story.fps;

export const scenes: TimedScene[] = (() => {
  let from = 0;
  return story.scenes.map((s, index) => {
    const frames = Math.round(s.durationSeconds * fps);
    const t = { ...s, index, from, frames };
    from += frames;
    return t;
  });
})();

export const scenesFrames = scenes.reduce((n, s) => n + s.frames, 0);
export const outroFrames = Math.round(story.outroSeconds * fps);
export const totalFrames = scenesFrames + outroFrames;

export const groups: VisualGroup[] = scenes.reduce<VisualGroup[]>((acc, s) => {
  const last = acc[acc.length - 1];
  if (last && last.key === s.visual) {
    last.beats.push({ from: s.from - last.from, frames: s.frames, scene: s });
    last.frames += s.frames;
  } else {
    acc.push({ key: s.visual, from: s.from, frames: s.frames, beats: [{ from: 0, frames: s.frames, scene: s }] });
  }
  return acc;
}, []);

/** Scene under an absolute frame (the last scene holds through the outro). */
export const sceneAt = (frame: number): TimedScene => {
  for (let i = scenes.length - 1; i >= 0; i--) if (frame >= scenes[i].from) return scenes[i];
  return scenes[0];
};

/** The current scene plus the frame inside it. For layers that live outside the scene Series. */
export const useScene = () => {
  const frame = useCurrentFrame();
  const scene = sceneAt(frame);
  return { frame, scene, local: frame - scene.from, prev: scenes[scene.index - 1] };
};

/** Inside a visual group: which beat is playing and how far into it. */
export const useBeat = (beats: VisualGroup['beats']) => {
  const frame = useCurrentFrame();
  let i = 0;
  for (let b = 0; b < beats.length; b++) if (frame >= beats[b].from) i = b;
  return { beat: i, local: frame - beats[i].from, frames: beats[i].frames, frame, beats };
};
