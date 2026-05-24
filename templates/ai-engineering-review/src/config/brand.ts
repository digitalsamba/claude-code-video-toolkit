import type { Theme } from '../../../../lib/theme';

export const brandName = 'digital-samba-ai-engineering';

export const brand = {
  name: 'Digital Samba · AI Engineering',
  colors: {
    primary: '#ea580c',
    primaryLight: '#fb923c',
    primaryDark: '#c2410c',
    accent: '#c9b37a',
    silver: '#d9d5c5',
    iron: '#1a1816',
    ironLight: '#2a2621',
    ironDeep: '#0f0d0b',
    textDark: '#d9d5c5',
    textMedium: '#a8a292',
    textLight: '#7c7668',
    bgLight: '#1a1816',
    bgAlt: '#2a2621',
    bgDark: '#0f0d0b',
    bgOverlay: 'rgba(15, 13, 11, 0.92)',
    divider: '#3d3830',
    shadow: 'rgba(0, 0, 0, 0.5)',
  },
  fonts: {
    primary: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
    display: "'Cinzel', 'Trajan Pro', Georgia, serif",
    mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace',
  },
  spacing: {
    xs: 8,
    sm: 16,
    md: 24,
    lg: 48,
    xl: 80,
    xxl: 120,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 14,
  },
  typography: {
    h1: { size: 96, weight: 700 },
    h2: { size: 72, weight: 700 },
    h3: { size: 48, weight: 600, letterSpacing: 2 },
    body: { size: 40, weight: 400 },
    label: { size: 28, weight: 600, letterSpacing: 6 },
  },
  assets: {
    logo: undefined as string | undefined,
    logoLight: undefined as string | undefined,
  },
};

export const brandTheme: Theme = {
  colors: brand.colors,
  fonts: brand.fonts,
  spacing: brand.spacing,
  borderRadius: brand.borderRadius,
  typography: brand.typography,
};
