// Project-local copy of lib/components/cues.ts (this template keeps its own copies, like Stage.tsx and
// Cursor.tsx, so it renders without the toolkit around it). Change both together.

/**
 * Narration-linked cues for synthetic UI demos.
 *
 * A scene's clicks, typing and camera moves belong on the words that name them ("click
 * Save" lands on "save"). Hand-set frame numbers drift every time the voice is regenerated
 * or re-timed, so look the frame up instead: `cue('save')` finds the word in the scene's
 * `.captions.json` (written by tools/captions.py beside each voice phrase) and returns the
 * frame it starts on, relative to the scene.
 *
 *   import line06a from '../../public/audio/scenes/06a_link.captions.json';
 *   import line06b from '../../public/audio/scenes/06b_link.captions.json';
 *
 *   export const LINK_VOICE = voice([
 *     { file: '06a_link', at: 10, captions: line06a },
 *     { file: '06b_link', at: 256, captions: line06b },
 *   ]);
 *   const cue = LINK_VOICE.cue;
 *   const CLICK_SAVE = cue('save');                    // the frame 'save' starts on
 *   const CLICK_FIELD = cue('before', { offset: -3 }); // a beat before the word
 *   const TYPE_AT = cue('change', { after: CLICK_FIELD });
 *
 * `at` is where the phrase starts inside the scene, the same number its <Audio> Sequence
 * uses, so the scene file is the one place phrase placement lives. A phrase that is not in
 * the narration throws at bundle time with the words that are, so a rewritten line fails
 * the render instead of putting a click on nothing.
 */

/** One word of a `.captions.json` file (the @remotion/captions `Caption` shape). */
export type CaptionWord = { text: string; startMs: number; endMs: number };

/** A voice phrase: its audio file stem, its start frame in the scene, and its word timings. */
export type VoicePhrase = { file: string; at: number; captions: readonly CaptionWord[] };

export type CueOptions = {
  /** Frames to add to the result: negative lands the action a beat before the word. */
  offset?: number;
  /** 'start' (default) is the first word's start; 'end' is the last word's end. */
  edge?: 'start' | 'end';
  /** Only matches starting after this frame (e.g. a previous cue), for repeated words. */
  after?: number;
  /** Which match to take, 1-based, after `after` is applied (default 1). */
  nth?: number;
  /** Only search this phrase (its `file`), when the same words occur in more than one. */
  in?: string;
};

export type Voice = {
  /** The phrases as [file, at] pairs, for the scene's <Audio> Sequences. */
  lines: [string, number][];
  /** Scene frame of a word or phrase in the narration. */
  cue: (phrase: string, options?: CueOptions) => number;
};

type Word = { w: string; start: number; end: number; file: string };

const normalise = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9']+/g, ' ')
    .replace(/(^| )'+|'+( |$)/g, '$1$2')
    .trim();

/** Words of a phrase as the matcher compares them: lower case, punctuation dropped. */
export const cueWords = (phrase: string): string[] => normalise(phrase).split(' ').filter(Boolean);

export function voice(phrases: readonly VoicePhrase[], fps = 30): Voice {
  const words: Word[] = [];
  for (const p of phrases) {
    for (const c of p.captions) {
      // a caption entry can hold more than one token ("you've", "Hide tiles"); keep each
      const tokens = cueWords(c.text);
      tokens.forEach((w) =>
        words.push({
          w,
          start: p.at + Math.round((c.startMs * fps) / 1000),
          end: p.at + Math.round((c.endMs * fps) / 1000),
          file: p.file,
        }),
      );
    }
  }

  const cue = (phrase: string, options: CueOptions = {}): number => {
    const target = cueWords(phrase);
    const { offset = 0, edge = 'start', after, nth = 1 } = options;
    if (!target.length) throw new Error(`cue(${JSON.stringify(phrase)}): no words to look for`);
    const matches: number[] = [];
    for (let i = 0; i + target.length <= words.length; i++) {
      if (options.in && words[i].file !== options.in) continue;
      if (after !== undefined && words[i].start <= after) continue;
      if (target.every((t, k) => words[i + k].w === t)) matches.push(i);
    }
    const i = matches[nth - 1];
    if (i === undefined) {
      const scope = options.in ? ` in ${options.in}` : '';
      const said = words
        .filter((w) => !options.in || w.file === options.in)
        .map((w) => `${w.w}@${w.start}`)
        .join(' ');
      const why = matches.length ? `only ${matches.length} match(es)` : 'not found';
      throw new Error(
        `cue(${JSON.stringify(phrase)}${after !== undefined ? `, after ${after}` : ''}): ${why}${scope}. ` +
          `The narration says: ${said}`,
      );
    }
    const frame = edge === 'end' ? words[i + target.length - 1].end : words[i].start;
    return frame + offset;
  };

  return { lines: phrases.map((p) => [p.file, p.at] as [string, number]), cue };
}
