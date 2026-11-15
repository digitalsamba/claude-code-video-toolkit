/**
 * Three fields by act, from brands/video-toolkit: navy for the terminal (and the laptop it turns out to be on),
 * light grey for the spec and the character acting it, DS blue for the close.
 */
import brand from '../../../brands/video-toolkit/brand.json';
import { terminalTheme } from '../../../lib/components';

const c = brand.colors;
export const FONT = brand.fonts.primary;
export const MONO = brand.fonts.mono;
export type Field = { bg: string; ink: string; muted: string; accent: string };

export const FIELDS: Record<'lab' | 'record' | 'identity', Field> = {
  lab: { bg: c.navy, ink: '#ffffff', muted: c.textOnDarkMuted, accent: c.primaryLight },
  record: { bg: c.bgAlt, ink: c.textDark, muted: c.textLight, accent: c.primary },
  identity: { bg: c.primary, ink: '#ffffff', muted: '#dfe7fb', accent: '#ffffff' },
};
export const TERM = terminalTheme(brand);
