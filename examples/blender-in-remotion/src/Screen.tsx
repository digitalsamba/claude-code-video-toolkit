/**
 * The terminal scene. It plays flat for the first five seconds of the video, and the same render
 * (out/screen.mp4, `npm run screen`) is the picture on the laptop in blender/room.json, so the cut at
 * frame REVEAL lands on the frame it left and the camera pulls back from it.
 *
 * Every output line is what tools/blender_render.py printed when this example was built (captured.json).
 */
import React from 'react';
import { AbsoluteFill, Audio, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { Terminal, TerminalLine, typedUntil } from '../../../lib/components';
import captured from './captured.json';
import { FIELDS, FONT, TERM } from './fields';

export const REVEAL = 150;        // the frame the video cuts to the laptop
export const SCREEN_FRAMES = 300; // the flat scene plus the pull-out it keeps playing through

const CHECK: TerminalLine = { at: 12, command: captured.check.command, perChar: 0.5, prompt: '$' };
const checkOut: TerminalLine[] = captured.check.lines.map((text, i) => ({
  at: typedUntil(CHECK) + 8 + i * 4,
  text,
  kind: text.includes('PASS') ? 'ok' : 'dim',
}));
const RENDER: TerminalLine = { at: 84, command: captured.render.command, perChar: 0.5, prompt: '$' };
const renderOut: TerminalLine[] = captured.render.lines.map((text, i) => ({
  at: REVEAL + 14 + i * 22,
  text,
  kind: text.startsWith('rendered') ? 'ok' : 'dim',
}));
export const TYPED = [CHECK, RENDER];
export const POPS = checkOut.filter((l) => 'kind' in l && l.kind === 'ok').map((l) => l.at);

export const Screen: React.FC<{ sound?: boolean }> = ({ sound = false }) => {
  const f = FIELDS.lab;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = Math.min(1, spring({ frame: frame - 2, fps, config: { damping: 200 } }));
  return (
    <AbsoluteFill style={{ background: f.bg, fontFamily: FONT }}>
      <div style={{ position: 'absolute', left: 120, top: 84, color: f.ink, opacity: t, transform: `translateY(${(1 - t) * 20}px)` }}>
        <div style={{ fontSize: 72, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1 }}>A spec in, a 3D scene out</div>
        <div style={{ fontSize: 36, marginTop: 14, color: f.muted }}>tools/blender_render.py, as it ran for this video</div>
      </div>
      <Terminal
        title="claude-code-video-toolkit"
        theme={TERM}
        lines={[CHECK, ...checkOut, RENDER, ...renderOut, { at: REVEAL + 14 + renderOut.length * 22, command: '', prompt: '$' }]}
        width={1680}
        height={680}
        fontSize={28}
        style={{ position: 'absolute', left: 120, top: 290 }}
      />
      {sound && TYPED.map((l) => (
        <Sequence key={l.at} from={l.at} durationInFrames={Math.max(4, typedUntil(l) - l.at)}>
          <Audio src={staticFile('audio/sfx/typing.mp3')} volume={0.5} />
        </Sequence>
      ))}
      {sound && POPS.map((at) => (
        <Sequence key={at} from={at} durationInFrames={20}><Audio src={staticFile('audio/sfx/pop.mp3')} volume={0.4} /></Sequence>
      ))}
    </AbsoluteFill>
  );
};
