/**
 * Brand Loader Utility
 *
 * Loads brand profiles from the brands/ directory and converts them to theme objects.
 * Used by video templates to apply consistent branding.
 */

import * as fs from 'fs';
import * as path from 'path';

// Brand JSON structure (matches brands/*/brand.json)
export interface Brand {
  name: string;
  description?: string;
  version?: string;
  website?: string;

  colors: {
    primary: string;
    primaryLight: string;
    primaryDark?: string;
    accent?: string;
    textDark: string;
    textMedium: string;
    textLight: string;
    bgLight: string;
    bgAlt?: string;
    bgDark: string;
    bgOverlay: string;
    divider: string;
    shadow: string;
  };

  fonts: {
    primary: string;
    mono: string;
  };

  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };

  borderRadius: {
    sm: number;
    md: number;
    lg: number;
  };

  typography: {
    h1: { size: number; weight: number };
    h2: { size: number; weight: number };
    h3: { size: number; weight: number; letterSpacing?: number };
    body: { size: number; weight: number };
    label: { size: number; weight: number; letterSpacing?: number };
  };

  assets?: {
    logo?: string;
    logoLight?: string;
    logoDark?: string;
    background?: string;
  };
}

// Theme structure used by templates
export interface Theme {
  colors: Brand['colors'];
  fonts: Brand['fonts'];
  spacing: Brand['spacing'];
  borderRadius: Brand['borderRadius'];
  typography: Brand['typography'];
}

// Project configuration (project.json)
export interface ProjectConfig {
  name: string;
  template: string;
  brand: string;
  created?: string;
  description?: string;
}

/**
 * Find the toolkit root directory by looking for the brands/ folder
 */
function findToolkitRoot(startDir: string): string | null {
  let current = startDir;
  const root = path.parse(current).root;

  while (current !== root) {
    const brandsPath = path.join(current, 'brands');
    if (fs.existsSync(brandsPath) && fs.statSync(brandsPath).isDirectory()) {
      return current;
    }
    current = path.dirname(current);
  }

  return null;
}

/**
 * Load a brand by name from the brands/ directory
 *
 * @param brandName - Name of the brand folder (e.g., 'digital-samba', 'default')
 * @param projectDir - Optional project directory to search from (defaults to cwd)
 * @returns Brand object
 * @throws Error if brand not found
 */
export function loadBrand(brandName: string, projectDir?: string): Brand {
  const searchDir = projectDir || process.cwd();
  const toolkitRoot = findToolkitRoot(searchDir);

  if (!toolkitRoot) {
    throw new Error(
      `Could not find toolkit root (brands/ directory) from ${searchDir}. ` +
      `Make sure you're running from within the video toolkit.`
    );
  }

  const brandPath = path.join(toolkitRoot, 'brands', brandName, 'brand.json');

  if (!fs.existsSync(brandPath)) {
    const availableBrands = listBrands(toolkitRoot);
    throw new Error(
      `Brand "${brandName}" not found at ${brandPath}. ` +
      `Available brands: ${availableBrands.join(', ')}`
    );
  }

  const brandJson = fs.readFileSync(brandPath, 'utf-8');
  return JSON.parse(brandJson) as Brand;
}

/**
 * List available brands in the toolkit
 */
export function listBrands(toolkitRoot?: string): string[] {
  const root = toolkitRoot || findToolkitRoot(process.cwd());
  if (!root) return [];

  const brandsDir = path.join(root, 'brands');
  if (!fs.existsSync(brandsDir)) return [];

  return fs.readdirSync(brandsDir).filter((name) => {
    const brandPath = path.join(brandsDir, name, 'brand.json');
    return fs.existsSync(brandPath);
  });
}

/**
 * Load project configuration from project.json
 */
export function loadProjectConfig(projectDir?: string): ProjectConfig | null {
  const dir = projectDir || process.cwd();
  const configPath = path.join(dir, 'project.json');

  if (!fs.existsSync(configPath)) {
    return null;
  }

  const configJson = fs.readFileSync(configPath, 'utf-8');
  return JSON.parse(configJson) as ProjectConfig;
}

/**
 * Load brand for the current project
 *
 * Reads project.json to get brand name, then loads that brand.
 * Falls back to 'default' brand if no project config exists.
 */
export function loadProjectBrand(projectDir?: string): Brand {
  const config = loadProjectConfig(projectDir);
  const brandName = config?.brand || 'default';
  return loadBrand(brandName, projectDir);
}

/**
 * Convert a Brand to a Theme object
 */
export function brandToTheme(brand: Brand): Theme {
  return {
    colors: brand.colors,
    fonts: brand.fonts,
    spacing: brand.spacing,
    borderRadius: brand.borderRadius,
    typography: brand.typography,
  };
}

/**
 * Load brand and convert to theme in one call
 */
export function loadBrandAsTheme(brandName: string, projectDir?: string): Theme {
  const brand = loadBrand(brandName, projectDir);
  return brandToTheme(brand);
}

/**
 * Load project brand as theme
 */
export function loadProjectTheme(projectDir?: string): Theme {
  const brand = loadProjectBrand(projectDir);
  return brandToTheme(brand);
}

/**
 * Get the path to a brand asset
 */
export function getBrandAssetPath(
  brandName: string,
  assetKey: keyof NonNullable<Brand['assets']>,
  projectDir?: string
): string | null {
  const brand = loadBrand(brandName, projectDir);
  const assetRelPath = brand.assets?.[assetKey];

  if (!assetRelPath) return null;

  const searchDir = projectDir || process.cwd();
  const toolkitRoot = findToolkitRoot(searchDir);
  if (!toolkitRoot) return null;

  return path.join(toolkitRoot, 'brands', brandName, assetRelPath);
}
