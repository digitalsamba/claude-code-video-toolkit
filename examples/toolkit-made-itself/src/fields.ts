/**
 * Three fields by act (design.md), from brands/video-toolkit, the site's tokens: DS blue for the prompt
 * and the close, light grey for the work, navy for the check. DS blue is the one accent on grey.
 */
import brand from '../../../brands/video-toolkit/brand.json';
import { ChartTheme, terminalTheme } from '../../../lib/components';

const c = brand.colors;
export const FONT = brand.fonts.primary;
export const MONO = brand.fonts.mono;
export type Field = { bg: string; ink: string; muted: string; accent: string; chart: ChartTheme };

export const FIELDS = {
  identity: {
    bg: c.primary, ink: '#ffffff', muted: '#dfe7fb', accent: '#ffffff',
    chart: { primary: '#ffffff', accent: '#ffffff', text: '#ffffff', muted: '#dfe7fb', grid: 'rgba(255,255,255,0.2)', bg: c.primary, font: FONT, other: '#7fa3ea' },
  } as Field,
  record: {
    bg: c.bgAlt, ink: c.textDark, muted: c.textLight, accent: c.primary,
    chart: { primary: c.primary, accent: c.textDark, text: c.textDark, muted: c.textLight, grid: 'rgba(18,20,24,0.10)', bg: c.bgAlt, font: FONT, other: '#c9ced8' },
  } as Field,
  lab: {
    bg: c.navy, ink: '#ffffff', muted: c.textOnDarkMuted, accent: c.primaryLight,
    chart: { primary: c.primaryLight, accent: '#ffffff', text: '#ffffff', muted: c.textOnDarkMuted, grid: 'rgba(255,255,255,0.14)', bg: c.navy, font: FONT, other: '#4a5680' },
  } as Field,
};

export const TERM = terminalTheme(brand);
export const COLORS = c;
