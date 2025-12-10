/**
 * Shared ThemeProvider for video components
 *
 * Provides theme context to all child components. Templates wrap their
 * composition in this provider with their brand-specific theme.
 */

import React, { createContext, useContext } from 'react';
import type { Theme } from './types';
import { defaultThemeValues } from './types';

const ThemeContext = createContext<Theme>(defaultThemeValues);

export interface ThemeProviderProps {
  theme?: Theme;
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  theme = defaultThemeValues,
  children,
}) => {
  return (
    <ThemeContext.Provider value={theme}>{children}</ThemeContext.Provider>
  );
};

/**
 * Hook to access the current theme
 *
 * @example
 * const theme = useTheme();
 * <div style={{ color: theme.colors.primary }}>Hello</div>
 */
export const useTheme = (): Theme => {
  const theme = useContext(ThemeContext);
  return theme;
};

/**
 * Helper to create common styles from theme
 */
export const createStyles = (theme: Theme) => ({
  // Typography
  h1: {
    fontFamily: theme.fonts.primary,
    fontSize: theme.typography.h1.size,
    fontWeight: theme.typography.h1.weight,
    color: theme.colors.textDark,
  },
  h2: {
    fontFamily: theme.fonts.primary,
    fontSize: theme.typography.h2.size,
    fontWeight: theme.typography.h2.weight,
    color: theme.colors.textDark,
  },
  h3: {
    fontFamily: theme.fonts.primary,
    fontSize: theme.typography.h3.size,
    fontWeight: theme.typography.h3.weight,
    color: theme.colors.textDark,
  },
  body: {
    fontFamily: theme.fonts.primary,
    fontSize: theme.typography.body.size,
    fontWeight: theme.typography.body.weight,
    color: theme.colors.textMedium,
  },
  label: {
    fontFamily: theme.fonts.primary,
    fontSize: theme.typography.label.size,
    fontWeight: theme.typography.label.weight,
    color: theme.colors.primary,
    textTransform: 'uppercase' as const,
    letterSpacing: theme.typography.label.letterSpacing || 2,
  },

  // Common UI elements
  card: {
    backgroundColor: theme.colors.bgOverlay,
    borderRadius: theme.borderRadius.md,
    padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
    boxShadow: `0 4px 12px ${theme.colors.shadow}`,
  },

  bullet: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: theme.colors.primary,
  },
});
