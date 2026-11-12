/** The YouTube thumbnail, 1280x720, from the video's own brand and Terminal (rendered with `remotion still`). */
import React from 'react';
import { AbsoluteFill } from 'remotion';
import { Terminal } from '../../../lib/components';
import { FIELDS, FONT, TERM } from './fields';

export const Thumbnail: React.FC = () => {
  const f = FIELDS.identity;
  return (
    <AbsoluteFill style={{ background: f.bg, fontFamily: FONT }}>
      <div style={{ position: 'absolute', left: 64, top: 110, color: f.ink, fontSize: 150, fontWeight: 800, letterSpacing: '-0.05em', lineHeight: 0.9 }}>
        MADE<br />WITH<br />ITSELF
      </div>
      <Terminal
        title="Claude Code"
        theme={TERM}
        lines={[
          { at: -1, command: 'Make a 90 second video', perChar: 0.001 },
          { at: -1, command: 'about this toolkit,', perChar: 0.001, prompt: ' ' },
          { at: -1, command: 'using this toolkit.', perChar: 0.001, prompt: ' ' },
        ]}
        width={560}
        height={310}
        fontSize={30}
        style={{ position: 'absolute', left: 660, top: 220 }}
      />
    </AbsoluteFill>
  );
};
