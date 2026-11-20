/**
 * The brand's mark and ident, as scripts/ident.py found them (src/ident.json). Both are optional: with no mark the
 * track is headed by the template's prompt chevron, and with no ident the 2D opening and close play.
 */
import identJson from '../ident.json';

type Clip = { file: string; frames: number; title_frame?: number };

export type Mark = {
  /** x, y, width, height of the mark in its own SVG's units */
  box: number[];
  /** A fill for each kind of ground; null is the field's ink */
  parts: { id: string; d: string; light: string | null; dark: string | null; mono: string | null }[];
};

export type Ident = {
  brand?: string;
  mark?: Mark;
  /** Opens and closes an episode (16:9) */
  episode?: { name: string; open?: Clip; close?: Clip };
  /** Closes a Short (9:16) */
  short?: { name: string; close?: Clip };
};

export const IDENT = identJson as Ident;
export const MARK = IDENT.mark;
