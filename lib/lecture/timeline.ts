// The lecture's data, as Remotion reads it: lecture.json (what is said, chalked and acted) and timeline.json
// (written by tools/lecture.py timeline: when each segment starts, its words by frame, the windows). Every frame
// here comes from those two files; nothing is typed by hand.
import { cueWords } from '../components/cues';
import defaults from './defaults.json';

export type At = string | number | { cue: string; nth?: number; offset?: number; edge?: 'start' | 'end' };

export type BoardItem = {
  at?: At;
  /** Lettering: the text, written left to right from (x, y). */
  chalk?: string;
  x?: number;
  y?: number;
  size?: 'xl' | 'l' | 'm' | 's' | 'xs';
  color?: 'white' | 'yellow';
  weight?: 400 | 700;
  tilt?: number;
  width?: number;
  dur?: number;
  quiet?: boolean;
  /** A stroke: line and arrow [x1, y1, x2, y2], box [x, y, w, h], ring [cx, cy, rx, ry], tick and cross [x, y, size],
   * under [x, y, text size, width] (an underline), or "path" with `d`. */
  stroke?: 'line' | 'arrow' | 'box' | 'ring' | 'tick' | 'cross' | 'under' | 'path';
  points?: number[];
  d?: string;
  dotted?: [number, number];
  /** A tap of the chalk (a full stop, a tick). */
  tap?: boolean;
  /** A pane: a clean slate panel in a plain sans with a title and a source line, for something real (a picture,
   * a video, a code listing). Panes are what the camera goes into in a window. A `video` (public/) plays muted from
   * the pane's `at`, `from` seconds in, and holds its last frame. */
  pane?: { x: number; y: number; w: number; h: number; title: string; source?: string; image?: string; video?: string; from?: number;
    code?: string; out?: At;
    /** In a short: the pane takes the frame (the presenter steps out) and the camera moves inside it, to each
     * `region` [x, y, w, h] of the picture (`size` its pixels) on its word. */
    takeover?: { size: [number, number]; label?: string; keys: { at: At; region: [number, number, number, number] }[] } };
  /** One of the project's own drawings (src/custom.tsx), given `at` and these props. */
  component?: string;
  props?: Record<string, unknown>;
};

export type Acting = [At, 'gesture' | 'show' | 'look' | 'beat', unknown?, unknown?];

export type Segment = {
  id: string;
  heading?: string | false;
  heading_at?: [number, number];
  board?: BoardItem[];
  acting?: Acting[];
  windows?: unknown[];
  /** Cuts to a second camera on the presenter: {in, out, type: 'closeup' | 'medium', az, dist, h, lens, push, dof}. */
  closes?: unknown[];
  gap_after?: number;
  carry?: boolean;
};

export type Lecture = {
  title: string;
  fps?: number;
  presenter: { cast: string };
  room_seconds?: number;
  /** The establishing shot (tools/lecture.py writes its room spec): {} for the defaults, keys as in defaults.json. */
  opening?: Partial<Omit<typeof defaults.opening, 'sounds'>> & { sounds?: Record<string, Sound> } | boolean;
  rules?: Partial<typeof defaults.rules>;
  staging?: Partial<typeof defaults.staging>;
  sounds?: Record<string, Sound>;
  segments: Segment[];
  /** Vertical shorts made from some consecutive segments (lib/lecture/defaults.json "short" has the layout). */
  shorts?: Short[];
};

export type Short = { id: string; segments: string[]; cover?: [string, number]; board?: Record<string, BoardItem[]>; captions?: Partial<typeof defaults.short.captions> } & Partial<Omit<typeof defaults.short, 'captions' | 'note'>>;

/** A sound laid on a clip's event: one file, or several taken in turn (footsteps). */
export type Sound = { file?: string; files?: string[]; volume?: number; offset?: number; frames?: number };
export type Word = { w: string; f: number; e: number };
export type Move = { at: number; dur: number; s: number; cx: number; cy: number };
export type Timeline = {
  fps: number;
  totalFrames: number;
  segments: { id: string; start: number; startFrame: number; dur: number; frames: number; words: Word[] }[];
  windows: { segment: string; from: number; to: number }[];
  moves: Move[];
  presenter?: { head_px: number[]; box_px: number[]; share: number; scale?: number | null };
  roomFrames?: number;
  opening?: { fadeFrom: number; fadeFrames: number; voice?: { id: string; at: number; startFrame: number; dur: number; frames: number } };
  /** Closer shots, in master frames (tools/lecture.py timeline, from each segment's "closes"). */
  closes?: { id: string; segment: string; from: number; to: number; type: string }[];
  /** Each pane video's length in frames (tools/lecture.py timeline reads it from the file). */
  videos?: Record<string, number>;
};

/** What a lecture and its timeline give the board: frames of words, segment edges, the clips' spans. */
export const makeTimeline = (lecture: Lecture, tl: Timeline) => {
  if (tl.segments.map((s) => s.id).join() !== lecture.segments.map((s) => s.id).join()) {
    throw new Error('timeline.json is for other segments than lecture.json: run `uv run tools/lecture.py timeline` again');
  }
  const rules = { ...defaults.rules, ...(lecture.rules ?? {}) };
  const staging = { ...defaults.staging, ...(lecture.staging ?? {}) };
  const seg = (n: number) => tl.segments[n];
  /** Frame a phrase is said in segment n (0-based), as tools/lecture.py finds it. */
  const cue = (n: number, phrase: string, nth = 1, edge: 'start' | 'end' = 'start') => {
    const want = cueWords(phrase);
    const words = seg(n).words;
    let seen = 0;
    for (let i = 0; i + want.length <= words.length; i++) {
      if (want.every((w, k) => words[i + k].w === w)) {
        seen += 1;
        if (seen === nth) return edge === 'start' ? words[i].f : words[i + want.length - 1].e;
      }
    }
    throw new Error(`${seg(n).id}: "${phrase}" is not in the narration; it says: "${words.map((w) => w.w).join(' ').slice(0, 300)}"`);
  };
  /** A board item's frame: words, {cue, nth, offset, edge}, "start", "end", or frames after the first word. */
  const at = (n: number, a: At | undefined): number => {
    if (a === undefined || a === 'start') return seg(n).startFrame;
    if (a === 'end') return seg(n).startFrame + seg(n).frames;
    if (typeof a === 'number') return seg(n).startFrame + a;
    if (typeof a === 'string') return cue(n, a);
    return cue(n, a.cue, a.nth ?? 1, a.edge ?? 'start') + (a.offset ?? 0);
  };
  const S = (n: number) => seg(n).startFrame;
  const E = (n: number) => seg(n).startFrame + seg(n).frames;
  const N = (n: number) => (n + 1 < tl.segments.length ? S(n + 1) : tl.totalFrames);
  const first = tl.roomFrames ?? Math.round((lecture.room_seconds ?? 0) * tl.fps);
  /** Each presenter clip: its id (as tools/lecture.py names it) and the master frames it covers. */
  const clips = tl.segments.map((s, i) => ({
    id: 'P' + s.id.split('_')[0].replace(/[^A-Za-z0-9]/g, '').slice(0, 4),
    from: i === 0 ? first : s.startFrame - rules.lead_frames,
    until: i + 1 < tl.segments.length ? tl.segments[i + 1].startFrame - rules.lead_frames : tl.totalFrames,
  }));
  return { tl, rules, staging, cue, at, S, E, N, clips, first };
};

export type LectureTimeline = ReturnType<typeof makeTimeline>;
