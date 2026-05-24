import { Composition } from 'remotion';
import { AIEngineeringReview, totalDurationInFrames } from './SprintReview';
import { videoConfig } from './config/sprint-config';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="AIEngineeringReview"
        component={AIEngineeringReview}
        durationInFrames={totalDurationInFrames}
        fps={videoConfig.fps}
        width={videoConfig.width}
        height={videoConfig.height}
      />
    </>
  );
};
