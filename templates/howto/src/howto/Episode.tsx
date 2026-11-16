/**
 * The timeline: opening, then for each chapter its card and its scenes, then the close. Scene lengths come from
 * the narration (src/timing.json, written by scripts/timing.py), never from hand-set numbers.
 *
 * `short` renders one chapter as a 9:16 Short instead: its card (already settled on frame 0, the cover), its
 * scenes with burned captions, and a short end card.
 *
 * When the brand has an ident (src/ident.json, written by scripts/ident.py) an episode opens with its open clip
 * and ends on its close clip, and a Short ends on the close clip. The 2D opening then starts at the track, and
 * the 2D close stops at the link. With no ident, or `ident: false` in the episode, the 2D mark does both.
 */
import React from 'react';
import { AbsoluteFill, Audio, OffthreadVideo, Sequence, staticFile, useVideoConfig } from 'remotion';
import { Captions, voice } from '../../../../lib/components';
import { EPISODE } from '../episode';
import timingJson from '../timing.json';
import { IDENT } from './ident';
import { Badge, Rise } from './parts';
import { ChapterCard, Close, Open, SCENES } from './scenes';
import { FIELDS, type FieldName, layoutFor, note, loadFonts, SANS, title } from './theme';
import type { Cue, Episode, SceneSpec } from './types';

loadFonts();

type Phrase = { file: string; at: number; frames: number; words: { text: string; startMs: number; endMs: number }[] };
type Timing = { fps: number; scenes: Record<string, { frames: number; phrases: Phrase[] }> };
const TIMING = timingJson as Timing;

export const CARD = 54; // a chapter card, in frames
const OPEN_MIN = 150;
const END_CARD = 75; // a Short's end card
const OPEN_SKIP = 30; // after an ident the 2D opening starts where its track draws: the ident has shown the mark
const LINK_HOLD = 75; // before an ident close, the link stays this long after the last word
const STAGE = '#05070c'; // behind an ident clip while it loads

const phrasesOf = (id: string) => TIMING.scenes[id]?.phrases ?? [];

const cueFor = (id: string): Cue => {
  const ps = phrasesOf(id);
  if (!ps.length) {
    return (phrase) => {
      throw new Error(`Scene "${id}" has no narration, so cue('${phrase}') has nothing to find. Add public/audio/scenes/NNa_${id}.mp3 and run scripts/timing.py.`);
    };
  }
  return voice(ps.map((p) => ({ file: p.file, at: p.at, captions: p.words }))).cue;
};

type Segment =
  | { kind: 'open'; frames: number }
  | { kind: 'card'; chapter: number; frames: number; instant?: boolean }
  | { kind: 'scene'; spec: SceneSpec; frames: number }
  | { kind: 'close'; frames: number; hold?: boolean }
  /** A clip of the brand's ident; `over` lays the episode's name and link on it (a Short's end) */
  | { kind: 'ident'; file: string; frames: number; over?: boolean }
  | { kind: 'end'; frames: number };

const sceneFrames = (spec: SceneSpec) => Math.max(TIMING.scenes[spec.id]?.frames ?? 0, spec.minFrames ?? 0, 30);

export const timeline = (ep: Episode, short?: number): Segment[] => {
  const ident = ep.ident === false ? {} : IDENT;
  if (short !== undefined) {
    const scenes = ep.scenes.filter((s) => s.chapter === short);
    const end = ident.short?.close;
    return [
      { kind: 'card', chapter: short, frames: CARD, instant: true },
      ...scenes.map((spec) => ({ kind: 'scene' as const, spec, frames: sceneFrames(spec) })),
      end ? { kind: 'ident', file: end.file, frames: end.frames, over: true } : { kind: 'end', frames: END_CARD },
    ];
  }
  const out: Segment[] = [];
  const opens = ident.episode?.open;
  if (opens) out.push({ kind: 'ident', file: opens.file, frames: opens.frames });
  out.push({ kind: 'open', frames: Math.max(TIMING.scenes.open?.frames ?? 0, OPEN_MIN) - (opens ? OPEN_SKIP : 0) });
  let chapter = -1;
  for (const spec of ep.scenes) {
    if (spec.chapter !== chapter) {
      chapter = spec.chapter;
      out.push({ kind: 'card', chapter, frames: CARD });
    }
    out.push({ kind: 'scene', spec, frames: sceneFrames(spec) });
  }
  const closes = ident.episode?.close;
  if (closes) {
    const said = phrasesOf('close');
    const last = said[said.length - 1];
    out.push({ kind: 'close', frames: (last ? last.at + last.frames : 60) + LINK_HOLD, hold: true });
    out.push({ kind: 'ident', file: closes.file, frames: closes.frames });
  } else {
    out.push({ kind: 'close', frames: Math.max(TIMING.scenes.close?.frames ?? 0, 150) });
  }
  return out;
};

export const duration = (ep: Episode, short?: number) => timeline(ep, short).reduce((n, s) => n + s.frames, 0);

/** The narration of a scene, and in a Short its captions. */
const Voice: React.FC<{ id: string; captions?: { color: string; accent: string; ground: string; bottom: number } }> = ({ id, captions }) => (
  <>
    {phrasesOf(id).map((p) => (
      <Sequence key={p.file} from={p.at} durationInFrames={p.frames + 6}>
        <Audio src={staticFile(`audio/scenes/${p.file}.mp3`)} />
        {captions && (
          <Captions
            src={`audio/scenes/${p.file}.captions.json`}
            fontFamily={SANS}
            fontSize={58}
            fontWeight={700}
            color={captions.color}
            highlightColor={captions.accent}
            background={captions.ground}
            bottom={captions.bottom}
            maxChars={26}
            maxWords={5}
          />
        )}
      </Sequence>
    ))}
  </>
);

/** The episode comes from src/episode.ts by import: composition props are JSON, and an episode holds functions. */
export const EpisodeVideo: React.FC<{ short?: number }> = ({ short }) => {
  const ep = EPISODE;
  const { width, height } = useVideoConfig();
  const L = layoutFor(width, height);
  const n = ep.chapters.length;
  const badge = `${ep.series} ${ep.number} · toolkit ${ep.version}`;
  const segs = timeline(ep, short);
  const total = segs.reduce((n, s) => n + s.frames, 0);
  // an ident carries its own sound, so the bed starts after the open clip and is out before the close clip
  const first = segs[0];
  const lastSeg = segs[segs.length - 1];
  const bedFrom = first.kind === 'ident' ? first.frames : 0;
  const bed = total - bedFrom - (lastSeg.kind === 'ident' ? lastSeg.frames : 0);
  const openSkip = first.kind === 'ident' ? OPEN_SKIP : 0;
  let from = 0;

  return (
    <AbsoluteFill style={{ background: FIELDS.work.ground }}>
      {ep.music && (
        <Sequence from={bedFrom} durationInFrames={bed}>
          <Audio
            src={staticFile(ep.music.file)}
            volume={(t) => ep.music!.volume * Math.min(1, t / 20, Math.max(0, (bed - t) / 60))}
          />
        </Sequence>
      )}
      {segs.map((s, i) => {
        const start = from;
        from += s.frames;
        const fieldName: FieldName =
          s.kind === 'open' || s.kind === 'ident'
            ? 'work'
            : s.kind === 'close' || s.kind === 'end'
              ? 'identity'
              : ep.chapters[s.kind === 'card' ? s.chapter : s.spec.chapter].field;
        const f = s.kind === 'ident' ? { ...FIELDS.work, ground: STAGE } : FIELDS[fieldName];
        let body: React.ReactNode = null;

        if (s.kind === 'ident') {
          body = (
            <>
              <OffthreadVideo src={staticFile(s.file)} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />
              {s.over && (
                <div style={{ position: 'absolute', left: L.x, top: L.top, width: L.right - L.x }}>
                  <Rise at={0} style={title(L, f)}>
                    {ep.series} {ep.number}:
                    <br />
                    {ep.title.join(' ')}
                  </Rise>
                  <Rise at={6} style={{ ...note(L, f), color: f.ink, marginTop: 28, whiteSpace: 'nowrap' }}>
                    {ep.link}
                  </Rise>
                </div>
              )}
            </>
          );
        } else if (s.kind === 'open') {
          body = (
            <Sequence from={-openSkip}>
              <Open
                L={L}
                f={f}
                text={ep.title}
                chapters={n}
                titleAt={Math.max(44, (phrasesOf('open')[0]?.at ?? 58) - 14)}
                lineCues={ep.openTitle?.(cueFor('open'))}
                stopCues={ep.openStops?.(cueFor('open'))}
              />
              <Voice id="open" />
            </Sequence>
          );
        } else if (s.kind === 'card') {
          body = <ChapterCard L={L} f={f} index={s.chapter} of={n} text={ep.chapters[s.chapter].title} instant={s.instant} />;
        } else if (s.kind === 'scene') {
          const Scene = SCENES[s.spec.kind] as React.FC<any>;
          const cue = cueFor(s.spec.id);
          const props = { ...s.spec.props(cue), ...(L.vertical && s.spec.short ? (s.spec.short as (c: Cue) => object)(cue) : {}) };
          body = (
            <>
              <Scene L={L} f={f} {...props} />
              {s.spec.kind !== 'session' && short === undefined && <Badge L={L} f={f} text={badge} />}
              <Voice id={s.spec.id} captions={short !== undefined ? { color: f.ink, accent: f.accent, ground: f.ground, bottom: 330 } : undefined} />
            </>
          );
        } else if (s.kind === 'close') {
          const ends = phrasesOf('close');
          const nextAt = ends[0]?.at ?? 20;
          const last = ends[ends.length - 1];
          body = (
            <>
              <Close
                L={L}
                f={f}
                chapters={n}
                next={ep.next}
                link={ep.link}
                nextAt={nextAt}
                linkAt={last ? last.at + last.frames - 6 : 40}
                endAt={last ? last.at + last.frames + 14 : 80}
                runUntil={last ? last.at + last.frames : undefined}
                hold={s.hold}
              />
              <Voice id="close" />
            </>
          );
        } else {
          // a Short's end card: the mark closes back to a prompt, the episode it came from above it
          body = <Close L={L} f={f} chapters={n} next={[`${ep.series} ${ep.number}:`, ep.title.join(' ')]} link={ep.link} nextAt={0} linkAt={6} endAt={14} />;
        }

        return (
          <Sequence key={i} from={start} durationInFrames={s.frames}>
            <AbsoluteFill style={{ background: f.ground }}>{body}</AbsoluteFill>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
