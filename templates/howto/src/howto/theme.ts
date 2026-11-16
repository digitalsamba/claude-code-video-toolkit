/**
 * design.md as code: fields per act, the five type steps, the layout edges. A scene asks for its field and the
 * frame's layout; it never picks a colour or a size of its own.
 */
import type React from 'react';
import { staticFile } from 'remotion';
import { loadMonoFont } from '../../../../lib/components';
import { BRAND } from '../brand';

export type FieldName = 'explain' | 'work' | 'identity';

export type Field = {
  ground: string;
  ink: string;
  /** The one data accent: whatever the narration is talking about */
  accent: string;
  /** Data that isn't being talked about */
  other: string;
  /** Secondary text (passes AA on the ground at label size) */
  ink2: string;
  /** Ground hairlines */
  hair: string;
  /** Which of the brand mark's fills suits this ground */
  mark: 'light' | 'dark' | 'mono';
};

const c = BRAND.colors as Record<string, string>;

export const FIELDS: Record<FieldName, Field> = {
  explain: { ground: c.bgAlt, ink: c.textDark, accent: c.primary, other: '#c9cfdc', ink2: '#4a5160', hair: 'rgba(18, 20, 24, 0.08)', mark: 'light' },
  work: { ground: c.navyDark, ink: '#ffffff', accent: c.primaryLight, other: '#3d4a70', ink2: c.textOnDarkMuted, hair: 'rgba(255, 255, 255, 0.07)', mark: 'dark' },
  identity: { ground: c.primary, ink: '#ffffff', accent: '#ffffff', other: '#6d97e8', ink2: 'rgba(255, 255, 255, 0.85)', hair: 'rgba(255, 255, 255, 0.12)', mark: 'mono' },
};

export const SANS = `Inter, ${BRAND.fonts.primary}`;
export const MONO = `'JetBrains Mono', ${BRAND.fonts.mono}`;

/** Load the bundled fonts once, so a render looks the same on macOS, Linux and Windows. */
export const loadFonts = () => {
  loadMonoFont('Inter', staticFile('fonts/inter/Inter-Regular.woff2'), staticFile('fonts/inter/Inter-Bold.woff2'));
  loadMonoFont(
    'JetBrains Mono',
    staticFile('fonts/jetbrains-mono/JetBrainsMono-Regular.woff2'),
    staticFile('fonts/jetbrains-mono/JetBrainsMono-Bold.woff2'),
  );
};

export type Layout = {
  W: number;
  H: number;
  vertical: boolean;
  /** Title edge */
  x: number;
  /** Right margin's x */
  right: number;
  /** Titles start here */
  top: number;
  /** The track's line */
  trackY: number;
  /** Notes sit on this baseline */
  foot: number;
  size: { display: number; figure: number; title: number; label: number; note: number };
  /** Ground row pitch */
  row: number;
};

export const layoutFor = (W: number, H: number): Layout => {
  const vertical = H > W;
  const x = vertical ? 80 : 120;
  return {
    W,
    H,
    vertical,
    x,
    right: W - x,
    top: vertical ? 300 : 120,
    trackY: vertical ? 1330 : 790,
    foot: H - (vertical ? 120 : 72),
    size: vertical
      ? { display: 150, figure: 120, title: 64, label: 38, note: 30 }
      : { display: 200, figure: 132, title: 72, label: 36, note: 28 },
    row: 54,
  };
};

export const display = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 700,
  fontSize: L.size.display,
  lineHeight: 0.95,
  letterSpacing: '-0.045em',
  color: f.ink,
});

export const figure = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 700,
  fontSize: L.size.figure,
  lineHeight: 0.95,
  letterSpacing: '-0.035em',
  color: f.ink,
  fontVariantNumeric: 'tabular-nums',
});

export const title = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 700,
  fontSize: L.size.title,
  lineHeight: 1.05,
  letterSpacing: '-0.02em',
  color: f.ink,
});

export const label = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 400,
  fontSize: L.size.label,
  lineHeight: 1.3,
  color: f.ink2,
});

export const note = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: 400,
  fontSize: L.size.note,
  lineHeight: 1.3,
  color: f.ink2,
});
