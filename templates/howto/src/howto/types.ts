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
};

/** A line of text, and optionally more of it that lands on later words */
export type Said = { text: string; at: number; then?: { text: string; at: number; newLine?: boolean } };

export type LedgerProps = {
  title?: Said;
  rows: LedgerRow[];
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

export type CommandProps = { command: string; at: number; lines: { text: string; at: number }[]; note?: string };

export type Kinds = {
  ledger: LedgerProps;
  figure: FigureProps;
  statement: StatementProps;
  bars: BarsProps;
  session: SessionProps;
  command: CommandProps;
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

export type Chapter = { title: Lines; field: FieldName };

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
  /** Frames in the opening's narration that bring in each line of the title */
  openTitle?: (cue: Cue) => number[];
  /** Frames in the opening's narration that light the chapter stops ("what's free, what isn't...") */
  openStops?: (cue: Cue) => number[];
  /** Optional bed under the whole episode, from public/ */
  music?: { file: string; volume: number };
};
