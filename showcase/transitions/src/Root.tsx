import { Composition } from 'remotion';
import { TransitionGallery, transitionGalleryConfig } from './TransitionGallery';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id={transitionGalleryConfig.id}
        component={TransitionGallery}
        durationInFrames={transitionGalleryConfig.durationInFrames}
        fps={transitionGalleryConfig.fps}
        width={transitionGalleryConfig.width}
        height={transitionGalleryConfig.height}
      />
    </>
  );
};
