import React from 'react';
import { Composition } from 'remotion';
import { EPISODE } from './episode';
import { duration, EpisodeVideo } from './howto/Episode';

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
