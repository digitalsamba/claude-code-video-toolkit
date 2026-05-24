import { Fragment } from 'react';
import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { loadFont } from '@remotion/google-fonts/VT323';
import type { Theme } from './themes';

const { fontFamily } = loadFont();

const STAGES = ['NARRATE', 'SCORE', 'GENERATE', 'COMPOSE', 'RENDER'];
const STAGE_START = 55;
const STAGE_STRIDE = 12;

export const Pipeline: React.FC<{ theme: Theme }> = ({ theme }) => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        alignItems: 'center',
        justifyContent: 'flex-end',
        paddingBottom: 74,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 18,
          fontFamily,
          fontSize: 42,
          letterSpacing: 3,
        }}
      >
        {STAGES.map((stage, i) => {
          const ignite = STAGE_START + i * STAGE_STRIDE;
          const arrowIgnite = ignite + 6;
          return (
            <Fragment key={stage}>
              <Stage label={stage} igniteFrame={ignite} frame={frame} theme={theme} />
              {i < STAGES.length - 1 && (
                <Arrow igniteFrame={arrowIgnite} frame={frame} theme={theme} />
              )}
            </Fragment>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const Stage: React.FC<{
  label: string;
  igniteFrame: number;
  frame: number;
  theme: Theme;
}> = ({ label, igniteFrame, frame, theme }) => {
  const brightness = interpolate(
    frame,
    [igniteFrame - 2, igniteFrame, igniteFrame + 4, igniteFrame + 14],
    [0, 1.35, 1, 0.88],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const hueMix = interpolate(frame, [igniteFrame, igniteFrame + 14], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const color = lerpHex(theme.pipeline.ignite, theme.pipeline.settle, hueMix);

  return (
    <span
      style={{
        color,
        opacity: brightness,
        textShadow: `0 0 8px ${color}, 0 0 20px ${color}bb, 0 0 36px ${color}55`,
      }}
    >
      {label}
    </span>
  );
};

const Arrow: React.FC<{
  igniteFrame: number;
  frame: number;
  theme: Theme;
}> = ({ igniteFrame, frame, theme }) => {
  const opacity = interpolate(
    frame,
    [igniteFrame - 2, igniteFrame, igniteFrame + 4],
    [0, 1.2, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  return (
    <span
      style={{
        color: theme.pipeline.arrow,
        opacity,
        textShadow: `0 0 10px ${theme.pipeline.arrowShadow}, 0 0 22px ${theme.pipeline.arrowShadow}`,
      }}
    >
      ▸
    </span>
  );
};

function lerpHex(a: string, b: string, t: number): string {
  const parse = (h: string) => [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];
  const [ar, ag, ab] = parse(a);
  const [br, bg, bb] = parse(b);
  const mix = (x: number, y: number) => Math.round(x + (y - x) * t);
  const hex = (n: number) => n.toString(16).padStart(2, '0');
  return `#${hex(mix(ar, br))}${hex(mix(ag, bg))}${hex(mix(ab, bb))}`;
}
