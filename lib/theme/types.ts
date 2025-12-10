/**
 * Shared theme type definitions
 *
 * These types define the theme structure used across all templates and components.
 * They align with the Brand interface in lib/brand.ts.
 */

export interface ThemeColors {
  // Primary palette
  primary: string;
  primaryLight: string;
  primaryDark?: string;
  accent?: string;

  // Text colors
  textDark: string;
  textMedium: string;
  textLight: string;

  // Backgrounds
  bgLight: string;
  bgAlt?: string;
  bgDark: string;
  bgOverlay: string;

  // UI elements
  divider: string;
  shadow: string;
}

export interface ThemeFonts {
  primary: string;
  mono: string;
}

export interface ThemeSpacing {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
  xxl: number;
}

export interface ThemeBorderRadius {
  sm: number;
  md: number;
  lg: number;
}

export interface ThemeTypography {
  h1: { size: number; weight: number };
  h2: { size: number; weight: number };
  h3: { size: number; weight: number; letterSpacing?: number };
  body: { size: number; weight: number };
  label: { size: number; weight: number; letterSpacing?: number };
}

export interface Theme {
  colors: ThemeColors;
  fonts: ThemeFonts;
  spacing: ThemeSpacing;
  borderRadius: ThemeBorderRadius;
  typography: ThemeTypography;
}

/**
 * Default theme values - used as fallback when theme properties are missing
 */
export const defaultThemeValues: Theme = {
  colors: {
    primary: '#3b82f6',
    primaryLight: '#60a5fa',
    textDark: '#1e293b',
    textMedium: '#475569',
    textLight: '#94a3b8',
    bgLight: '#ffffff',
    bgDark: '#0f172a',
    bgOverlay: 'rgba(255, 255, 255, 0.95)',
    divider: '#e2e8f0',
    shadow: 'rgba(0, 0, 0, 0.1)',
  },
  fonts: {
    primary: 'Inter, system-ui, sans-serif',
    mono: 'JetBrains Mono, monospace',
  },
  spacing: {
    xs: 8,
    sm: 16,
    md: 24,
    lg: 32,
    xl: 48,
    xxl: 64,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 16,
  },
  typography: {
    h1: { size: 88, weight: 700 },
    h2: { size: 64, weight: 700 },
    h3: { size: 48, weight: 600 },
    body: { size: 32, weight: 400 },
    label: { size: 24, weight: 500, letterSpacing: 2 },
  },
};
