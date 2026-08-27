import React from 'react';
import { Composition } from 'remotion';
import { Baseline, BLOCKS, FRAMES_PER_BLOCK, FPS } from './Baseline';

/**
 * Single composition. Duration is derived in calculateMetadata (exercising that
 * API) from the number of blocks so adding a block never desyncs the timeline.
 */
export const Root: React.FC = () => (
  <Composition
    id="Baseline"
    component={Baseline}
    fps={FPS}
    width={1280}
    height={720}
    durationInFrames={BLOCKS.length * FRAMES_PER_BLOCK}
    defaultProps={{ blocks: BLOCKS.length }}
    calculateMetadata={async ({ props }) => ({
      durationInFrames: props.blocks * FRAMES_PER_BLOCK,
      props,
    })}
  />
);
