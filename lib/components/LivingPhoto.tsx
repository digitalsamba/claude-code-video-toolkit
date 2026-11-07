import React from 'react';
import { Freeze, OffthreadVideo, staticFile, useCurrentFrame } from 'remotion';

export interface LivingPhotoProps {
  /** An image-to-video clip of the person (ltx2.py --input), relative to public/. */
  src: string;
  /** Hold this frame of the clip for the rest of the shot (default 145, a 5 s LTX clip at 30 fps).
   *  Lower it to stop before a late cut to a stranger. */
  hold?: number;
  /** Skip this many frames at the start (a glitch or garbled text in the first second). */
  trim?: number;
  /** CSS object-position for the cover crop; frame the close-up here, not in the LTX input. */
  pos?: string;
  /** CSS filter for the genre's grade, e.g. "sepia(.25) saturate(1.1)". */
  filter?: string;
  /** Style for the outer box (position absolute, overflow hidden). */
  style?: React.CSSProperties;
}

/**
 * A colleague's animated photo: plays the clip once, then holds its last good frame.
 * Wrap it in a <Sequence> that starts at its shot. Method: .claude/skills/parody-short/people.md
 */
export const LivingPhoto: React.FC<LivingPhotoProps> = ({
  src, hold = 145, trim = 0, pos = 'center 30%', filter, style,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ position: 'absolute', overflow: 'hidden', ...style }}>
      <Freeze frame={hold} active={frame >= hold}>
        <OffthreadVideo
          src={staticFile(src)}
          trimBefore={trim}
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: pos, filter }}
        />
      </Freeze>
    </div>
  );
};
