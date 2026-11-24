import React from 'react';
import { Composition } from 'remotion';
import { EPISODES } from './episode';
import { EpisodeVideo } from './howto/Episode';
import { duration } from './howto/timeline';
import { EXAMPLE_FRAMES, ScreenExample } from './screens/example';

/**
 * One episode gives `Episode` and a `Short-N` per chapter. A project with several (each with an `id` in
 * src/episode.ts) gives `Episode-<id>` and `Short-<id>-N` for each. `Screen-example` and `Screen-example-short`
 * preview src/screens/example.tsx, a `screen` scene to copy from: delete them with it.
 */
export const RemotionRoot: React.FC = () => (
  <>
    {EPISODES.map((ep) => {
      const tag = ep.id ? `-${ep.id}` : '';
      const which = ep.id ? { episode: ep.id } : {};
      return (
        <React.Fragment key={ep.id ?? 'episode'}>
          <Composition id={`Episode${tag}`} component={EpisodeVideo} defaultProps={which} durationInFrames={duration(ep)} fps={30} width={1920} height={1080} />
          {ep.chapters.map((_, i) => (
            <Composition
              key={i}
              id={`Short${tag}-${i + 1}`}
              component={EpisodeVideo}
              defaultProps={{ short: i, ...which }}
              durationInFrames={duration(ep, i)}
              fps={30}
              width={1080}
              height={1920}
            />
          ))}
        </React.Fragment>
      );
    })}
    <Composition id="Screen-example" component={ScreenExample} durationInFrames={EXAMPLE_FRAMES} fps={30} width={1920} height={1080} />
    <Composition id="Screen-example-short" component={ScreenExample} durationInFrames={EXAMPLE_FRAMES} fps={30} width={1080} height={1920} />
  </>
);
