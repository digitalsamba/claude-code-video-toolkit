import React from 'react';
import { Composition } from 'remotion';
import { MadeWithItself, TOTAL } from './Video';
import { Thumbnail } from './Thumbnail';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="MadeWithItself" component={MadeWithItself} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
    <Composition id="Thumbnail" component={Thumbnail} durationInFrames={1} fps={30} width={1280} height={720} />
  </>
);
