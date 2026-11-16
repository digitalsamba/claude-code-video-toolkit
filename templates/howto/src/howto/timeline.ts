/**
 * The timeline: opening, then for each chapter its card and its scenes, then the close. Scene lengths come from
 * the narration (src/timing.json, written by scripts/timing.py), never from hand-set numbers.
 *
 * This file imports nothing from React or Remotion, so a script can read the same timeline the render plays:
 * scripts/timeline.mjs prints it as JSON, and scripts/chapters.py turns that into YouTube chapters.
 */
import timingJson from '../timing.json';
import { IDENT } from './ident';
import type { Episode, SceneSpec } from './types';

export const ASK_START = 8; // a chapter's question starts typing here, a character a frame
/** How long a chapter card with a question runs: it types, is sent, and the answer opens. */
export const ASK_SENT = 12; // frames after the last character before it is sent
export const ASK_HOLD = 32; // and the question stays this long before the answer opens: time to read it
export const askFrames = (text: string) => ASK_START + text.length + ASK_SENT + ASK_HOLD;

type Phrase = { file: string; at: number; frames: number; words: { text: string; startMs: number; endMs: number }[] };
type SceneTimes = Record<string, { frames: number; phrases: Phrase[] }>;
/** One episode's scenes at the top; a project with several has each under `episodes` by its id */
type Timing = { fps: number; scenes: SceneTimes; episodes?: Record<string, { scenes: SceneTimes }> };
const TIMING = timingJson as Timing;
const scenesOf = (ep: Episode): SceneTimes => (ep.id ? TIMING.episodes?.[ep.id]?.scenes : TIMING.scenes) ?? {};
/** Where an episode's narration is, under public/ */
export const audioOf = (ep: Episode) => (ep.id ? `audio/scenes/${ep.id}/` : 'audio/scenes/');

export const CARD = 54; // a chapter card, in frames
const OPEN_MIN = 150;
const OPEN_TITLE = 105; // with no narration of its own the opening is the title and a beat to read it
const END_CARD = 75; // a Short's end card
export const OPEN_SKIP = 30; // after an ident the 2D opening starts where its track draws: the ident has shown the mark
const LINK_HOLD = 75; // before an ident close, the link stays this long after the last word

export const phrasesOf = (ep: Episode, id: string) => scenesOf(ep)[id]?.phrases ?? [];

export type Segment =
  | { kind: 'open'; frames: number }
  | { kind: 'card'; chapter: number; frames: number; instant?: boolean }
  | { kind: 'scene'; spec: SceneSpec; frames: number; first?: boolean; hook?: boolean }
  /** The hook's question, typed before anything else */
  | { kind: 'ask'; text: string; frames: number }
  | { kind: 'close'; frames: number; hold?: boolean }
  /** A clip of the brand's ident; `over` lays the episode's name and link on it (a Short's end) */
  | { kind: 'ident'; file: string; frames: number; over?: boolean }
  | { kind: 'end'; frames: number };

const sceneFrames = (ep: Episode, spec: SceneSpec) => Math.max(scenesOf(ep)[spec.id]?.frames ?? 0, spec.minFrames ?? 0, 30);

export const timeline = (ep: Episode, short?: number): Segment[] => {
  const ident = ep.ident === false ? {} : IDENT;
  if (short !== undefined) {
    const scenes = ep.scenes.filter((s) => s.chapter === short);
    const end = ident.short?.close;
    return [
      { kind: 'card', chapter: short, frames: CARD, instant: true },
      ...scenes.map((spec, i) => ({ kind: 'scene' as const, spec, frames: sceneFrames(ep, spec), first: i === 0 })),
      end ? { kind: 'ident', file: end.file, frames: end.frames, over: true } : { kind: 'end', frames: END_CARD },
    ];
  }
  const out: Segment[] = [];
  if (ep.hook) {
    if (ep.hook.ask) out.push({ kind: 'ask', text: ep.hook.ask, frames: askFrames(ep.hook.ask) });
    out.push({ kind: 'scene', spec: ep.hook.scene, frames: sceneFrames(ep, ep.hook.scene), first: true, hook: true });
  }
  const opens = ident.episode?.open;
  if (opens) out.push({ kind: 'ident', file: opens.file, frames: opens.frames });
  const openFrames = scenesOf(ep).open?.frames;
  out.push({ kind: 'open', frames: (openFrames ? Math.max(openFrames, OPEN_MIN) : OPEN_TITLE) - (opens ? OPEN_SKIP : 0) });
  let chapter = -1;
  for (const spec of ep.scenes) {
    if (spec.chapter !== chapter) {
      chapter = spec.chapter;
      const ask = ep.chapters[chapter].ask;
      out.push({ kind: 'card', chapter, frames: ask ? Math.max(CARD, askFrames(ask)) : CARD });
      out.push({ kind: 'scene', spec, frames: sceneFrames(ep, spec), first: true });
      continue;
    }
    out.push({ kind: 'scene', spec, frames: sceneFrames(ep, spec) });
  }
  const closes = ident.episode?.close;
  if (closes) {
    const said = phrasesOf(ep, 'close');
    const last = said[said.length - 1];
    out.push({ kind: 'close', frames: (last ? last.at + last.frames : 60) + LINK_HOLD, hold: true });
    out.push({ kind: 'ident', file: closes.file, frames: closes.frames });
  } else {
    out.push({ kind: 'close', frames: Math.max(scenesOf(ep).close?.frames ?? 0, 150) });
  }
  return out;
};

export const duration = (ep: Episode, short?: number) => timeline(ep, short).reduce((n, s) => n + s.frames, 0);

