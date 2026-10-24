export type MaxMood = 'happy' | 'thinking' | 'smug' | 'shocked' | 'confused' | 'sad';

/** Where Max stands. 'hero' is large beside the stage, 'side' is a small corner presence. */
export type MaxLayout = 'hero' | 'side' | 'hidden';

export interface MaxDirection {
  mood: MaxMood;
  layout: MaxLayout;
  /** Pupils follow this. Defaults to 'up' (toward the stage). */
  look?: 'up' | 'left' | 'right' | 'center' | 'down';
  /** Thought bubble text (hero layout only: in side layout use an emote). */
  thought?: string;
  /** Arm raised toward the stage, so Max presents the chart */
  pose?: 'point';
  /** Glyph that pops above his head */
  emote?: '?' | '!';
}

export interface Chapter {
  title: string;
}

export interface Scene {
  /** Stable id, also the audio file stem (01-intro -> audio/scenes/01-intro.mp3) */
  id: string;
  /** Index into story.chapters */
  chapter: number;
  /** Cold-open hook before chapter 1: hides the chapter header */
  hook?: boolean;
  /**
   * Visual group key (see src/episode/visuals.tsx). Consecutive scenes with the
   * same key share ONE mounted visual, which receives the scene boundaries as
   * beats, so a chart keeps drawing across scenes instead of re-entering.
   */
  visual: string;
  /** Spoken text. `npm run scripts` exports it to audio/scenes/<id>.txt for voiceover.py */
  narration: string;
  audioFile: string;
  /** Scene length. sync_timing.py --apply rewrites this from the real audio. */
  durationSeconds: number;
  max: MaxDirection;
}

export interface Story {
  title: string;
  fps: number;
  width: number;
  height: number;
  chapters: Chapter[];
  scenes: Scene[];
  /** Branded end card after the last scene (no narration) */
  outroSeconds: number;
  /** Optional, relative to public/. Ducked under narration. */
  backgroundMusicFile?: string;
  musicVolume?: number;
}
