/**
 * Shared utility functions for video components
 */

/**
 * Convert hex color to rgba
 *
 * @param hex - Hex color string (e.g., '#ff6600')
 * @param alpha - Alpha value 0-1
 * @returns rgba string
 */
export function hexToRgba(hex: string, alpha: number): string {
  // Handle shorthand hex
  const fullHex = hex.length === 4
    ? `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`
    : hex;

  const r = parseInt(fullHex.slice(1, 3), 16);
  const g = parseInt(fullHex.slice(3, 5), 16);
  const b = parseInt(fullHex.slice(5, 7), 16);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Size configurations for PiP and other scaled elements
 */
export const SIZE_PRESETS = {
  sm: { width: 240, height: 135 },
  md: { width: 320, height: 180 },
  lg: { width: 400, height: 225 },
} as const;

/**
 * Position configurations for corner-anchored elements
 */
export const POSITION_PRESETS = {
  'bottom-right': { bottom: 40, right: 40 },
  'bottom-left': { bottom: 40, left: 40 },
  'top-right': { top: 40, right: 40 },
  'top-left': { top: 40, left: 40 },
} as const;

export type SizePreset = keyof typeof SIZE_PRESETS;
export type PositionPreset = keyof typeof POSITION_PRESETS;
