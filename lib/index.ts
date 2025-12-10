/**
 * claude-code-video-toolkit shared library
 *
 * This library provides reusable components and utilities for video templates.
 *
 * Usage in templates:
 *   import { ThemeProvider, useTheme } from '../../../lib';
 *   import { AnimatedBackground, Label } from '../../../lib/components';
 */

// Theme system
export * from './theme';

// Components
export * from './components';

// Brand utilities
export {
  loadBrand,
  loadProjectBrand,
  loadBrandAsTheme,
  loadProjectTheme,
  listBrands,
  brandToTheme,
  getBrandAssetPath,
  loadProjectConfig,
} from './brand';

export type { Brand, Theme as BrandTheme, ProjectConfig } from './brand';
