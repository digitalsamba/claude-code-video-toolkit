/**
 * An episode is config: what each scene shows, and the narration words that make it happen. A scene's props are a
 * function of `cue`, so every frame number comes from the voice: `at: cue('three hundred')`.
 */
import type { SessionFocus, SessionHighlight, SessionLine, SessionSkin } from '../../../../lib/components';
import type { FieldName } from './theme';

export type Cue = (phrase: string, options?: { offset?: number; edge?: 'start' | 'end'; after?: number; nth?: number; in?: string }) => number;

/** Text on screen, broken by hand: one string per line, two lines at most for a title. */
export type Lines = string[];

export type LedgerRow = {
  label: string;
  /** A smaller line under the label: the tool, the unit price */
  detail?: string;
  /** A number counts up to itself when the row lands; a string just appears */
  value: number | string;
  format?: (n: number) => string;
  /** Frame the row lands on (a cue); it holds the accent until the next row lands */
  at: number;
  /** The detail line lands later, on its own words */
  detailAt?: number;
  /** The value lands later, on its own words (a number counts up from here) */
  valueAt?: number;
  /**
   * What the row is about, shown in a window beside the rows while the narration is on it: a clip, a few images
   * flicked through, or a moving waveform for something heard. Files are in public/.
   */
  show?: { video?: string; images?: string[]; wave?: boolean };
};

/** A line of text, and optionally more of it that lands on later words */
export type Said = { text: string; at: number; then?: { text: string; at: number; newLine?: boolean } };

export type LedgerProps = {
  title?: Said;
  /**
   * The number the ledger is about, large beside its rows. After a `figure` scene it starts where that scene left
   * its number and moves into place, so the cut carries it across; `carry: false` lands it on `at` instead.
   */
  hero?: { text: string; carry?: boolean; at?: number };
  rows: LedgerRow[];
  /**
   * Up to three rows fill the frame (the detail under the label, a number at figure size) and more are set
   * compact. `big: true` keeps the large setting for a fourth row, which fits when the ledger has no title.
   */
  big?: boolean;
  /** Every value landed so far takes the accent ("all free") until the next row lands */
  allAt?: number;
  note?: string;
  after?: Said;
};

export type FigureProps = {
  title?: Said;
  value: number;
  prefix?: string;
  suffix?: string;
  format?: (n: number) => string;
  at: number;
  /** Lines under the figure; each replaces the ones before on its cue, or with `add` joins them below */
  lines: { text: Lines; at: number; add?: boolean }[];
  note?: string;
};

export type StatementProps = {
  lines: { text: Lines; at: number; size?: 'display' | 'figure' | 'title' }[];
  note?: string;
};

export type BarsProps = {
  title?: { text: Lines; at: number };
  /** A bar grows on `at` (the month's word) and shows its value on `valueAt` (the amount's word) */
  bars: { label: string; value: number; at: number; valueAt?: number; sub?: { text: string; at: number } }[];
  /** A line across the chart (a limit, a free allowance) */
  limit?: { value: number; label: string; at: number };
  format: (n: number) => string;
  note?: string;
  /** Last line, after the bars: what they add up to */
  after?: { text: string; at: number };
};

export type SessionProps = {
  lines: SessionLine[];
  skin?: 'claude-code' | 'codex' | 'shell' | SessionSkin;
  status?: string[];
  focus?: SessionFocus[];
  highlight?: SessionHighlight[];
};

export type SplitRow = {
  text: string;
  /** Frame the row lands on (a cue); it holds the accent until the next row lands, in either column */
  at: number;
  /** How many things the row stands for in its column's count (default 1): "checks Node, uv and FFmpeg" is 3 */
  count?: number;
  /** A smaller line under the row, landing on its own words (a pointer, a condition) */
  detail?: string;
  detailAt?: number;
};

export type SplitSide = {
  /** Who this column is: "You", "Claude Code" */
  head: string;
  /** Six rows at most beside a title in 16:9 (eight without one); in a Short the two columns stack */
  rows: SplitRow[];
  /** Beside the count: what is being counted ("things you do") */
  unit?: string;
  /** The frame the count is said: it takes the accent and the pen circles it */
  countAt?: number;
};

/** Who does what: two columns, rows landing on their words, and a running count under each. */
export type SplitProps = {
  title?: Said;
  left: SplitSide;
  right: SplitSide;
  note?: string;
};

export type CommandProps = { command: string; at: number; lines: { text: string; at: number }[]; note?: string };

export type Kinds = {
  ledger: LedgerProps;
  figure: FigureProps;
  statement: StatementProps;
  bars: BarsProps;
  session: SessionProps;
  command: CommandProps;
  split: SplitProps;
};

export type SceneSpec = {
  [K in keyof Kinds]: {
    /** Matches the narration files: public/audio/scenes/<NN><letter>_<id>.mp3 */
    id: string;
    kind: K;
    /** 0-based chapter; a chapter card plays before its first scene */
    chapter: number;
    props: (cue: Cue) => Kinds[K];
    /** In a 9:16 Short, these replace props (titles broken by hand for the narrow frame) */
    short?: (cue: Cue) => Partial<Kinds[K]>;
    /** At least this many frames, if the narration is shorter */
    minFrames?: number;
  };
}[keyof Kinds];

export type Chapter = {
  title: Lines;
  field: FieldName;
  /**
   * The chapter as a question, typed at a prompt on its card in place of the title ("what does $30 buy?"). The
   * chapter's first scene then opens out of the prompt, as the answer. Keep it under about 28 characters.
   */
  ask?: string;
};

export type Episode = {
  series: string;
  number: number;
  /** Shown in the badge, so a stale episode is easy to spot */
  version: string;
  title: Lines;
  chapters: Chapter[];
  scenes: SceneSpec[];
  next: Lines;
  link: string;
  /**
   * Before the opening: a question typed at a prompt and one scene that answers it, the most interesting thing
   * the episode has to say. The brand's ident and the title come after it. Its narration is scene id `hook`.
   */
  hook?: { ask?: string; scene: SceneSpec };
  /** Frames in the opening's narration that bring in each line of the title */
  openTitle?: (cue: Cue) => number[];
  /** Frames in the opening's narration that light the chapter stops ("what's free, what isn't...") */
  openStops?: (cue: Cue) => number[];
  /** `false` keeps the template's 2D opening and close even when the brand has an ident (src/ident.json) */
  ident?: boolean;
  /** `false` turns off the sounds things land on (public/audio/sfx, made by scripts/sounds.py) */
  sounds?: boolean;
  /** Optional bed under the whole episode, from public/ */
  music?: { file: string; volume: number };
};
