import { Audio, Easing, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion';
import { theme } from '../config/theme';

export type SfxName = 'whoosh' | 'tick' | 'chip' | 'pop' | 'thud' | 'rewind' | 'drop';

/** One sound effect from public/sfx (scripts/make-sfx.mjs), fired at a frame of the current Sequence. */
export const Sfx: React.FC<{ name: SfxName; at: number; volume?: number }> = ({ name, at, volume = 0.6 }) => (
  <Sequence from={Math.max(0, Math.round(at))} durationInFrames={45} name={`sfx: ${name}`} layout="none">
    <Audio src={staticFile(`sfx/${name}.mp3`)} volume={volume} />
  </Sequence>
);

export interface CameraKey {
  at: number;
  scale: number;
  /** Focus point in stage px: this point moves to the stage centre as scale grows */
  x: number;
  y: number;
}

/**
 * Eased camera over a stage: keyframes of scale + focus point. Use it for
 * push-ins on the number that matters, then ease back out.
 */
export const Camera: React.FC<{ keys: CameraKey[]; width?: number; height?: number; children: React.ReactNode }> = ({
  keys,
  width = 1080 - theme.layout.gutter * 2,
  height = theme.layout.stageBottom - theme.layout.stageTop,
  children,
}) => {
  const frame = useCurrentFrame();
  const all: CameraKey[] = [{ at: 0, scale: 1, x: width / 2, y: height / 2 }, ...keys];
  const frames = all.map((k) => k.at);
  const opts = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.bezier(0.45, 0, 0.2, 1) } as const;
  const pick = (f: (k: CameraKey) => number) => (all.length > 1 ? interpolate(frame, frames, all.map(f), opts) : f(all[0]));
  const s = pick((k) => k.scale);
  const fx = pick((k) => k.x);
  const fy = pick((k) => k.y);
  // Pull the focus point toward the centre as we zoom
  const tx = (width / 2 - fx) * (1 - 1 / s);
  const ty = (height / 2 - fy) * (1 - 1 / s);
  return (
    <div style={{ position: 'absolute', inset: 0, transform: `scale(${s}) translate(${tx}px, ${ty}px)`, transformOrigin: `${fx}px ${fy}px` }}>
      {children}
    </div>
  );
};
