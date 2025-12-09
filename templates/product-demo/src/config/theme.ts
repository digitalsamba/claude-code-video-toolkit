import React, { createContext, useContext } from 'react';
import type { Theme } from './types';

// Default theme - dark tech aesthetic
export const defaultTheme: Theme = {
  colors: {
    primary: '#0066FF',
    primaryLight: '#3388FF',
    accent: '#00D4AA',
    textDark: '#ffffff',
    textMedium: '#888888',
    textLight: '#555555',
    bgLight: '#0a0a0a',
    bgDark: '#000000',
    bgOverlay: 'rgba(255, 255, 255, 0.05)',
    divider: '#222222',
    shadow: 'rgba(0, 0, 0, 0.5)',
  },
  fonts: {
    primary: 'Inter, system-ui, -apple-system, sans-serif',
    mono: 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace',
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
    sm: 6,
    md: 12,
    lg: 20,
  },
  typography: {
    h1: { size: 72, weight: 700 },
    h2: { size: 56, weight: 600 },
    h3: { size: 40, weight: 600 },
    body: { size: 24, weight: 400 },
    label: { size: 16, weight: 500, letterSpacing: 1 },
  },
};

// Light theme variant
export const lightTheme: Theme = {
  ...defaultTheme,
  colors: {
    ...defaultTheme.colors,
    textDark: '#1e293b',
    textMedium: '#475569',
    textLight: '#94a3b8',
    bgLight: '#ffffff',
    bgDark: '#f1f5f9',
    bgOverlay: 'rgba(0, 0, 0, 0.05)',
    divider: '#e2e8f0',
    shadow: 'rgba(0, 0, 0, 0.12)',
  },
};

// Theme context
const ThemeContext = createContext<Theme>(defaultTheme);

export const ThemeProvider: React.FC<{
  theme?: Theme;
  children: React.ReactNode;
}> = ({ theme = defaultTheme, children }) => {
  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);

// Helper to create theme from brand.json colors
export function createThemeFromBrand(brand: {
  colors: Record<string, string>;
  fonts?: { primary?: string; mono?: string };
  typography?: Record<string, { size: number; weight: number; letterSpacing?: number }>;
}): Theme {
  return {
    ...defaultTheme,
    colors: {
      ...defaultTheme.colors,
      ...brand.colors,
    },
    fonts: {
      ...defaultTheme.fonts,
      ...brand.fonts,
    },
    typography: {
      ...defaultTheme.typography,
      ...brand.typography,
    },
  };
}
