import React from 'react';
import { Freeze, OffthreadVideo, interpolate, staticFile, useCurrentFrame } from 'remotion';

export interface LivingPhotoProps {
  /** An image-to-video clip of the person (ltx2.py --input), relative to public/. */
  src: string;
  /** The last good frame of the clip, in frames at 30 fps counted after `trim` (default 145, a 5 s LTX clip).
   *  Lower it to stop before a late cut to a stranger or a smear; raise it for a clip lengthened with
   *  tools/ltx_extend.py. */
  hold?: number;
  /** Skip this many frames at the start (a glitch or garbled text in the first second). */
  trim?: number;
  /** Playback speed. Below 1 stretches the action (0.5 is natural slow motion for b-roll; faces look slowed below
   *  ~0.8). Overrides `fit`. */
  rate?: number;
  /** The shot's length in frames: slows the clip just enough to fill it, down to 0.8x. A clip still too short at
   *  0.8x needs extending (tools/ltx_extend.py), not looping: a back-and-forth tail reads as a rewind. */
  fit?: number;
  /** After `hold`: 'drift' (default) holds the frame with a slow push-in; 'freeze' holds it dead still (a
   *  freeze-frame gag). Either way the face stops, so a long one is flagged by qc.py as motion that stops dead. */
  after?: 'drift' | 'freeze';
  /** CSS object-position for the cover crop; frame the close-up here, not in the LTX input. */
  pos?: string;
  /** CSS filter for the genre's grade, e.g. "sepia(.25) saturate(1.1)". */
  filter?: string;
  /** Style for the outer box (position absolute, overflow hidden). */
  style?: React.CSSProperties;
}

const FIT_MIN = 0.8;

/** The playback rate a LivingPhoto uses. */
export const livingRate = ({ hold = 145, rate, fit }: Pick<LivingPhotoProps, 'hold' | 'rate' | 'fit'> = {}) =>
  rate ?? (fit ? Math.max(FIT_MIN, Math.min(1, hold / fit)) : 1);

/** Which clip frame shows at a given shot frame. Exported so a shot can key a gag to the same moment. */
export const livingFrame = (frame: number, props: Pick<LivingPhotoProps, 'hold' | 'rate' | 'fit'> = {}) =>
  Math.round(Math.min(frame * livingRate(props), props.hold ?? 145));

/**
 * A person's animated photo: plays the clip forward once (slowed to `fit` the shot if asked), then holds its last
 * good frame. It never plays backwards: a ping-pong tail measured as frames revisiting earlier ones and read as a
 * loop. For a shot longer than the clip, lengthen the clip with tools/ltx_extend.py.
 * Wrap it in a <Sequence> that starts at its shot. Method: .claude/skills/parody-short/people.md
 */
export const LivingPhoto: React.FC<LivingPhotoProps> = ({
  src, hold = 145, trim = 0, rate, fit, after = 'drift', pos = 'center 30%', filter, style,
}) => {
  const frame = useCurrentFrame();
  const r = livingRate({ hold, rate, fit });
  const clipFrame = livingFrame(frame, { hold, rate, fit });
  // the push-in runs over 10 s of the shot from the hold, whatever the playback rate
  const push = after === 'drift'
    ? interpolate(frame - hold / r, [0, 300], [1, 1.04], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' })
    : 1;
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
