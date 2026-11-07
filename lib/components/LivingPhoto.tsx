import React from 'react';
import { Freeze, OffthreadVideo, interpolate, staticFile, useCurrentFrame } from 'remotion';

export interface LivingPhotoProps {
  /** An image-to-video clip of the person (ltx2.py --input), relative to public/. */
  src: string;
  /** The last good frame of the clip, in clip frames (default 145, a 5 s LTX clip at 30 fps).
   *  Lower it to stop before a late cut to a stranger or a smear. */
  hold?: number;
  /** Skip this many frames at the start (a glitch or garbled text in the first second). */
  trim?: number;
  /** Playback speed. Below 1 stretches the action across a long shot (0.5 for slow-motion b-roll). */
  rate?: number;
  /** After `hold`: 'settle' (default) keeps the face alive by drifting slowly back and forth over the clip's
   *  last `settle` frames with a slight push-in; 'freeze' holds the frame dead still (a freeze-frame gag). */
  after?: 'settle' | 'freeze';
  /** Clip frames the settle drifts over (default 30, about a second). */
  settle?: number;
  /** CSS object-position for the cover crop; frame the close-up here, not in the LTX input. */
  pos?: string;
  /** CSS filter for the genre's grade, e.g. "sepia(.25) saturate(1.1)". */
  filter?: string;
  /** Style for the outer box (position absolute, overflow hidden). */
  style?: React.CSSProperties;
}

/** Which clip frame to show at a given shot frame. Exported so a shot can key a gag to the same moment. */
export const livingFrame = (frame: number, { hold = 145, rate = 1, after = 'settle', settle = 30 }: Pick<LivingPhotoProps, 'hold' | 'rate' | 'after' | 'settle'> = {}) => {
  const t = frame * rate;
  if (t < hold || after === 'freeze') return Math.min(t, hold);
  // half speed, there and back: a gentle idle, never a dead frame
  const w = Math.max(2, Math.min(settle, hold));
  const m = ((t - hold) * 0.5) % (2 * w);
  return hold - (m < w ? m : 2 * w - m);
};

/**
 * A person's animated photo. Plays the clip once, then settles: it drifts over its last second at half speed with
 * a slight push-in, so a shot longer than the clip never stops dead (that read as a pause in the video's flow).
 * Wrap it in a <Sequence> that starts at its shot. Method: .claude/skills/parody-short/people.md
 */
export const LivingPhoto: React.FC<LivingPhotoProps> = ({
  src, hold = 145, trim = 0, rate = 1, after = 'settle', settle = 30, pos = 'center 30%', filter, style,
}) => {
  const frame = useCurrentFrame();
  const clipFrame = Math.round(livingFrame(frame, { hold, rate, after, settle }));
  const past = frame * rate - hold;
  const push = after === 'settle' ? interpolate(past, [0, 300], [1, 1.04], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) : 1;
  return (
    <div style={{ position: 'absolute', overflow: 'hidden', ...style }}>
      <Freeze frame={clipFrame} active>
        <OffthreadVideo
          src={staticFile(src)}
          trimBefore={trim}
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: pos, filter, transform: `scale(${push})` }}
        />
      </Freeze>
    </div>
  );
};
