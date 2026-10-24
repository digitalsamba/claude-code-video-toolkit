import { Composition } from 'remotion';
import { story } from './config/story-config';
import { MathStory } from './MathStory';
import { totalFrames } from './timeline';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="MathStory"
    component={MathStory}
    durationInFrames={totalFrames}
    fps={story.fps}
    width={story.width}
    height={story.height}
  />
);
