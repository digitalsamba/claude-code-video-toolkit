import React from 'react';
import { Composition } from 'remotion';
import { EPISODE } from './episode';
import { EpisodeVideo } from './howto/Episode';
import { duration } from './howto/timeline';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Episode" component={EpisodeVideo} durationInFrames={duration(EPISODE)} fps={30} width={1920} height={1080} />
    {EPISODE.chapters.map((_, i) => (
      <Composition
        key={i}
        id={`Short-${i + 1}`}
        component={EpisodeVideo}
        defaultProps={{ short: i }}
        durationInFrames={duration(EPISODE, i)}
        fps={30}
        width={1080}
        height={1920}
      />
    ))}
  </>
);
