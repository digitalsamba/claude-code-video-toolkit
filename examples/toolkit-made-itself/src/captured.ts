/**
 * Lines from capture/ (real output of the commands run while this video was built), via
 * scripts/captured.py. pick() throws at bundle time when a line isn't there, so nothing on screen
 * is typed in by hand.
 */
import data from './captured.json';
import type { TerminalKind } from '../../../lib/components';

export type CapturedLine = { text: string; kind: TerminalKind };
type Run = { command: string; lines: CapturedLine[] };
const runs = data as unknown as Record<string, Run>;

export const run = (name: string): Run => {
  const r = runs[name];
  if (!r) throw new Error(`captured run "${name}" not found (have: ${Object.keys(runs).join(', ')})`);
  return r;
};

/** The first line of a run containing `contains` (after the line containing `after`, if given). */
export const pick = (name: string, contains: string, after?: string): CapturedLine => {
  const { lines } = run(name);
  let from = 0;
  if (after) {
    from = lines.findIndex((l) => l.text.includes(after));
    if (from < 0) throw new Error(`capture ${name}: no line with "${after}"`);
  }
  const hit = lines.slice(from).find((l) => l.text.includes(contains));
  if (!hit) throw new Error(`capture ${name}: no line with "${contains}"${after ? ` after "${after}"` : ''}`);
  return hit;
};

export const has = (name: string): boolean => name in runs;

/** The last line of a run containing `contains` (a progress counter's final value). */
export const pickLast = (name: string, contains: string): CapturedLine => {
  const hit = [...run(name).lines].reverse().find((l) => l.text.includes(contains));
  if (!hit) throw new Error(`capture ${name}: no line with "${contains}"`);
  return hit;
};

/** The line of a run that is exactly `text` (a section header such as "render:"). */
export const pickExact = (name: string, text: string): CapturedLine => {
  const hit = run(name).lines.find((l) => l.text === text);
  if (!hit) throw new Error(`capture ${name}: no line that is exactly "${text}"`);
  return hit;
};
