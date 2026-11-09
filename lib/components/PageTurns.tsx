/**
 * PageTurns: one scene that turns into new pages in place, each on its cue
 *
 * A hard edge sweeps across the frame and the next page is behind it. Use it when the same thing is
 * shown again in a new way and the change is the point: the same chart in three design sources, a
 * screen before and after a fix, a plan becoming the real thing. Each turn lands on the word that names
 * the new page (`at` from `cue()`), and nothing fades: the edge is the change.
 *
 * Between scenes, use `wipe()` from `@remotion/transitions` in a `TransitionSeries` instead; this is for
 * turns inside one scene, timed to the narration, which a TransitionSeries can't do.
 *
 * @example
 * ```tsx
 * import { PageTurns } from '../../../lib/components';
 *
 * <PageTurns
 *   before={<Question />}
 *   pages={[
 *     { at: cue('ours'), content: <ChartPage source="ours" /> },
 *     { at: cue('taste'), content: <ChartPage source="taste" /> },
 *     { at: cue("team's"), content: <ChartPage source="yours" /> },
 *   ]}
 * />
 * ```
 *
 * Worked example: projects that show design sources (the toolkit's design-sources sample).
 */

import React from 'react';
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from 'remotion';

export type PageTurn = {
  /** Frame the turn should land on: the cue of the word that names this page. */
  at: number;
  content: React.ReactNode;
};

export type PageTurnsProps = {
  pages: PageTurn[];
  /** Shown until the first turn (often the question the pages answer). */
  before?: React.ReactNode;
  /** Frames a turn takes (default 16, about half a second at 30fps). */
  turnFrames?: number;
  /**
   * Frames the edge starts ahead of `at`, so the new page is mostly in view as the word lands
   * (default 6). Set 0 to start the turn on the word.
   */
  lead?: number;
  /** The edge's direction of travel (default 'right': the next page comes in from the left). */
  direction?: 'right' | 'left' | 'down' | 'up';
  /** Easing of the edge (default ease-out cubic: fast start, settling on the word). */
  easing?: (t: number) => number;
};

const clip = (direction: NonNullable<PageTurnsProps['direction']>, t: number) => {
  const rest = `${((1 - t) * 100).toFixed(3)}%`;
  switch (direction) {
    case 'left': return `inset(0 0 0 ${rest})`;
    case 'down': return `inset(0 0 ${rest} 0)`;
    case 'up': return `inset(${rest} 0 0 0)`;
    default: return `inset(0 ${rest} 0 0)`;
  }
};

/** How far a turn has got at `frame`, 0-1, for timing other things to it. */
export const turnProgress = (frame: number, at: number, turnFrames = 16, lead = 6, easing = Easing.out(Easing.cubic)) =>
  interpolate(frame, [at - lead, at - lead + turnFrames], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing,
  });

export const PageTurns: React.FC<PageTurnsProps> = ({
  pages, before, turnFrames = 16, lead = 6, direction = 'right', easing = Easing.out(Easing.cubic),
}) => {
  const frame = useCurrentFrame();
  const sorted = [...pages].sort((a, b) => a.at - b.at);
  const progress = sorted.map((p) => turnProgress(frame, p.at, turnFrames, lead, easing));
  // Only the page that is fully in place and the one turning over it need to be drawn.
  const settled = progress.reduce((last, t, i) => (t >= 1 ? i : last), -1);
  return (
    <AbsoluteFill>
      {settled < 0 && before}
      {sorted.map((p, i) => {
        const t = progress[i];
        if (t <= 0 || i < settled) return null;
        return (
          <AbsoluteFill key={i} style={{ clipPath: t >= 1 ? undefined : clip(direction, t) }}>
            {p.content}
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};
