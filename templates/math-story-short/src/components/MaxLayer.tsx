import { interpolate, spring, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';
import type { MaxLayout } from '../config/types';
import { scenesFrames, useScene } from '../timeline';
import { Max } from './Max';

interface Placement {
  x: number;
  feet: number;
  height: number;
}

const PLACE: Record<MaxLayout | 'outro', Placement> = {
  hero: { x: 240, feet: 1510, height: 470 },
  side: { x: 200, feet: 1515, height: 390 },
  hidden: { x: 200, feet: 2500, height: 390 },
  outro: { x: 540, feet: 1220, height: 560 },
};

const lerp = (a: Placement, b: Placement, t: number): Placement => ({
  x: a.x + (b.x - a.x) * t,
  feet: a.feet + (b.feet - a.feet) * t,
  height: a.height + (b.height - a.height) * t,
});

/**
 * Max lives outside the scene Series so he never re-mounts: between scenes he
 * glides to the new layout, and a mood change triggers his hop.
 */
export const MaxLayer: React.FC = () => {
  const { fps } = useVideoConfig();
  const { frame, scene, local, prev } = useScene();
  const inOutro = frame >= scenesFrames;
  const since = inOutro ? frame - scenesFrames : local;

  const target = inOutro ? PLACE.outro : PLACE[scene.max.layout];
  const from = inOutro ? PLACE[scene.max.layout] : prev ? PLACE[prev.max.layout] : PLACE.hidden;
  const t = spring({ frame: since, fps, config: { damping: 20, stiffness: 90 } });
  const p = lerp(from, target, t);

  const mood = inOutro ? 'happy' : scene.max.mood;
  const moodChanged = inOutro || !prev || prev.max.mood !== scene.max.mood;
  const talking = !inOutro && local > 6 && local < scene.frames - 24;
  const talk = talking ? (Math.sin(frame * 0.9) + 1) / 2 : 0;

  const thought = inOutro ? undefined : scene.max.thought;
  const bubbleIn = spring({ frame: local - 10, fps, config: { damping: 14, stiffness: 160 } });
  const bubbleOut = interpolate(local, [scene.frames - 10, scene.frames], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const bubble = bubbleIn * bubbleOut;

  const emote = inOutro ? undefined : scene.max.emote;
  const emoteIn = spring({ frame: local - 6, fps, config: { damping: 9, stiffness: 200 } });
  const w = (p.height * 400) / 500;
  const headX = p.x + w * 0.12;
  const headY = p.feet - p.height * 0.62;

  return (
    <>
      <div style={{ position: 'absolute', left: p.x - w / 2, top: p.feet - p.height }}>
        <Max mood={mood} look={inOutro ? 'center' : scene.max.look} since={moodChanged ? since : 99} talk={talk} height={p.height} pose={inOutro ? undefined : scene.max.pose} />
      </div>
      {emote && (
        <div
          style={{
            position: 'absolute',
            left: p.x + w * 0.3,
            top: p.feet - p.height * 1.02,
            width: 76,
            height: 76,
            borderRadius: 38,
            background: emote === '!' ? theme.primary : theme.cream,
            color: emote === '!' ? theme.cream : theme.bg,
            fontFamily: theme.fonts.sans,
            fontWeight: 900,
            fontSize: 54,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: emote === '!' ? '0 0 30px rgba(193,18,31,0.6)' : '0 12px 30px rgba(0,10,20,0.4)',
            transform: `scale(${emoteIn * bubbleOut}) rotate(${interpolate(emoteIn, [0, 1], [-30, 8])}deg)`,
          }}
        >
          {emote}
        </div>
      )}
      {thought && bubble > 0.01 && (
        <div style={{ position: 'absolute', left: headX + w * 0.42, top: headY - 200, transformOrigin: '0% 100%', transform: `scale(${bubble})`, opacity: bubble }}>
          <div
            style={{
              background: theme.cream,
              color: theme.bg,
              fontFamily: theme.fonts.sans,
              fontWeight: 800,
              fontSize: 40,
              lineHeight: 1.18,
              letterSpacing: -0.5,
              padding: '24px 32px',
              borderRadius: 34,
              whiteSpace: 'pre',
              boxShadow: '0 24px 50px rgba(0,10,20,0.45)',
            }}
          >
            {thought}
          </div>
          {[22, 14, 9].map((r, i) => (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: 20 - i * 30,
                top: '100%',
                marginTop: 10 + i * 22,
                width: r * 2,
                height: r * 2,
                borderRadius: r,
                background: theme.cream,
              }}
            />
          ))}
        </div>
      )}
    </>
  );
};
