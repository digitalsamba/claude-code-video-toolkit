import React from 'react';
import { Composition } from 'remotion';
import { DataChartDemo, TOTAL } from './Demo';

export const RemotionRoot: React.FC = () => (
  <Composition id="DataChartDemo" component={DataChartDemo} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
);
