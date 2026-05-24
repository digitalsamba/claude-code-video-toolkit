import { Composition } from 'remotion';
import { SprintReview } from './SprintReview';
import { videoConfig } from './config/sprint-config';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SprintReview"
        component={SprintReview}
        durationInFrames={videoConfig.fps * videoConfig.durationSeconds}
        fps={videoConfig.fps}
        width={videoConfig.width}
        height={videoConfig.height}
      />
    </>
  );
};
