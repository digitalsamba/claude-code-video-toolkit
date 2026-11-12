import React from 'react';
import { Composition } from 'remotion';
import { Screen, SCREEN_FRAMES } from './Screen';
import { BlenderInRemotion, TOTAL } from './Video';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="BlenderInRemotion" component={BlenderInRemotion} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
    {/* what the laptop shows: render it first (npm run screen), then the Blender clips */}
    <Composition id="Screen" component={Screen} durationInFrames={SCREEN_FRAMES} fps={30} width={1920} height={1080} />
  </>
);
