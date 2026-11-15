import React from 'react';
import { Composition } from 'remotion';
import { SessionScene, TOTAL } from './Session';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="ClaudeCode" component={SessionScene} defaultProps={{ skin: 'claude-code' as const }} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
    <Composition id="Codex" component={SessionScene} defaultProps={{ skin: 'codex' as const }} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
    <Composition id="Shell" component={SessionScene} defaultProps={{ skin: 'shell' as const }} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
    <Composition id="Vertical" component={SessionScene} defaultProps={{ skin: 'claude-code' as const, vertical: true }} durationInFrames={TOTAL} fps={30} width={1080} height={1920} />
  </>
);
