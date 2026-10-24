import { useMemo } from 'react';
import { AbsoluteFill, Audio, getStaticFiles, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion';
import { FilmGrain } from '../../../lib/components/FilmGrain';
import { story } from './config/story-config';
import { Backdrop } from './components/Backdrop';
import { ChapterHeader } from './components/ChapterHeader';
import { Sfx } from './components/Fx';
import { MaxLayer } from './components/MaxLayer';
import { Outro } from './components/Outro';
import { TypewriterCaptions } from './components/TypewriterCaptions';
import { visuals } from './episode/visuals';
import { groups, outroFrames, scenes, scenesFrames, totalFrames } from './timeline';

const EXIT = 10;

/** Outgoing visual pushes toward the viewer and blurs away; the next one springs in. */
const GroupExit: React.FC<{ frames: number; children: React.ReactNode }> = ({ frames, children }) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [frames - EXIT, frames], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ opacity: 1 - t, transform: `scale(${1 + t * 0.08})`, filter: t > 0 ? `blur(${t * 14}px)` : undefined }}>
      {children}
    </AbsoluteFill>
  );
};

const Music: React.FC<{ src: string }> = ({ src }) => {
  const vol = story.musicVolume ?? 0.028;
  return (
    <Audio
      src={staticFile(src)}
      loop
      volume={(f) =>
        interpolate(f, [0, 3, totalFrames - 45, totalFrames], [0, vol, vol, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) *
        (f >= scenesFrames ? 1.6 : 1)
      }
    />
  );
};

/**
 * Layer order, back to front: backdrop, scene visuals (one Sequence per visual
 * group), header, Max, captions, grain. Header and Max sit outside the scene
 * Sequences so they persist and animate across cuts.
 */
export const MathStory: React.FC = () => {
  const files = useMemo(() => new Set(getStaticFiles().map((f) => f.name)), []);

  return (
    <AbsoluteFill style={{ background: '#003049' }}>
      <Backdrop />

      {groups.map((g) => {
        const Visual = visuals[g.key];
        if (!Visual) throw new Error(`No visual registered for "${g.key}" (src/episode/visuals.tsx)`);
        return (
          <Sequence key={`${g.key}-${g.from}`} from={g.from} durationInFrames={g.frames} name={`visual: ${g.key}`}>
            <GroupExit frames={g.frames}>
              <Visual beats={g.beats} />
            </GroupExit>
            {g.from > 0 && <Sfx name="whoosh" at={0} volume={0.3} />}
          </Sequence>
        );
      })}

      <ChapterHeader />

      <Sequence from={scenesFrames} durationInFrames={outroFrames} name="outro">
        <Outro />
        <Sfx name="pop" at={26} />
      </Sequence>

      <MaxLayer />

      {scenes.map((s) => (
        <Sequence key={s.id} from={s.from} durationInFrames={s.frames} name={s.id}>
          {files.has(s.audioFile) && <Audio src={staticFile(s.audioFile)} />}
          <TypewriterCaptions audioFile={s.audioFile} narration={s.narration} frames={s.frames} />
        </Sequence>
      ))}

      {story.backgroundMusicFile && files.has(story.backgroundMusicFile) && <Music src={story.backgroundMusicFile} />}

      <FilmGrain opacity={0.045} />
    </AbsoluteFill>
  );
};
