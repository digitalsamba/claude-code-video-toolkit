/**
 * design.md as code: fields per act, the five type steps, the layout edges. A scene asks for its field and the
 * frame's layout; it never picks a colour or a size of its own.
 */
import type React from 'react';
import { continueRender, delayRender, staticFile } from 'remotion';
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

/**
 * A brand can set any value of a field in brand.json `fields` ({ "work": { "ground": "#323E66" } }): its own dark
 * ground, or `"hair": "transparent"` for a flat ground with no rows.
 */
const own = (BRAND as unknown as { fields?: Partial<Record<FieldName, Partial<Field>>> }).fields ?? {};

export const FIELDS: Record<FieldName, Field> = {
  explain: { ground: c.bgAlt, ink: c.textDark, accent: c.primary, other: '#c9cfdc', ink2: '#4a5160', hair: 'rgba(18, 20, 24, 0.08)', mark: 'light', ...own.explain },
  work: { ground: c.navyDark, ink: '#ffffff', accent: c.primaryLight, other: '#3d4a70', ink2: c.textOnDarkMuted, hair: 'rgba(255, 255, 255, 0.07)', mark: 'dark', ...own.work },
  identity: { ground: c.primary, ink: '#ffffff', accent: '#ffffff', other: '#6d97e8', ink2: 'rgba(255, 255, 255, 0.85)', hair: 'rgba(255, 255, 255, 0.12)', mark: 'mono', ...own.identity },
};

export const SANS = `Inter, ${BRAND.fonts.primary}`;

/**
 * The brand's type: heading weight (the toolkit sets 700; a brand whose site sets headings at 600 says so in
 * brand.json `fonts.weights.heading`) and tracking (`fonts.tracking.heading`, `fonts.tracking.display`).
 */
const fonts = BRAND.fonts as { weights?: { heading?: number; body?: number }; tracking?: { heading?: string; display?: string } };
export const HEAD = fonts.weights?.heading ?? 700;
export const BODY = fonts.weights?.body ?? 400;
const TRACK_TITLE = fonts.tracking?.heading ?? '-0.02em';
const TRACK_DISPLAY = fonts.tracking?.display ?? '-0.045em';
export const MONO = `'JetBrains Mono', ${BRAND.fonts.mono}`;

/** Register each face at its own weight, holding the render until they load. */
const loadFaces = (family: string, faces: [number, string][]) => {
  if (typeof document === 'undefined' || typeof FontFace === 'undefined') return;
  const handle = delayRender(`Loading ${family}`);
  Promise.all(faces.map(([w, url]) => new FontFace(family, `url(${url})`, { weight: String(w) }).load().then((f) => document.fonts.add(f))))
    .catch((err) => console.warn(`${family} did not load, using the fallback`, err))
    .finally(() => continueRender(handle));
};

/** Load the bundled fonts once, so a render looks the same on macOS, Linux and Windows. */
export const loadFonts = () => {
  loadFaces('Inter', [
    [400, staticFile('fonts/inter/Inter-Regular.woff2')],
    [600, staticFile('fonts/inter/Inter-SemiBold.woff2')],
    [700, staticFile('fonts/inter/Inter-Bold.woff2')],
  ]);
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
  fontWeight: HEAD,
  fontSize: L.size.display,
  lineHeight: 0.95,
  letterSpacing: TRACK_DISPLAY,
  color: f.ink,
});

export const figure = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: HEAD,
  fontSize: L.size.figure,
  lineHeight: 0.95,
  letterSpacing: '-0.035em',
  color: f.ink,
  fontVariantNumeric: 'tabular-nums',
});

export const title = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: HEAD,
  fontSize: L.size.title,
  lineHeight: 1.05,
  letterSpacing: TRACK_TITLE,
  color: f.ink,
});

export const label = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: BODY,
  fontSize: L.size.label,
  lineHeight: 1.3,
  color: f.ink2,
});

export const note = (L: Layout, f: Field): React.CSSProperties => ({
  fontFamily: SANS,
  fontWeight: BODY,
  fontSize: L.size.note,
  lineHeight: 1.3,
  color: f.ink2,
});
