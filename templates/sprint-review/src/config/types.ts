/**
 * Type definitions for sprint-review template
 *
 * Theme types are imported from the shared library.
 * Sprint-specific types are defined here.
 */

// Re-export theme types from lib
export type {
  Theme,
  ThemeColors,
  ThemeFonts,
  ThemeSpacing,
  ThemeBorderRadius,
  ThemeTypography,
} from '../../../../lib/theme';

// Sprint configuration types
export interface SprintInfo {
  name: string;           // Sprint name (e.g., "Cho Oyu")
  dateRange: string;      // Date range (e.g., "24th Nov – 8th Dec")
  product: string;        // Product name (e.g., "Digital Samba Mobile")
  platform: string;       // Platform (e.g., "iOS Embedded App Update")
  version: string;        // Version string (e.g., "4.0.2")
  build?: string;         // Build number (e.g., "233")
}

export interface OverviewItem {
  text: string;           // Main text
  highlight: string;      // Highlighted portion
}

export interface StatItem {
  value: number;
  label: string;
}

export interface DemoConfig {
  type: 'single' | 'split' | 'timelapse';
  videoFile?: string;
  leftVideo?: string;
  rightVideo?: string;
  label: string;
  jiraRef?: string;
  durationSeconds: number;
  playbackRate?: number;
  startFrom?: number;
  leftStartFrom?: number;
  rightStartFrom?: number;
  leftLabel?: string;
  rightLabel?: string;
}

export interface CreditSection {
  category: string;
  items: string[];
}

export interface NarratorConfig {
  enabled: boolean;
  videoFile?: string;           // Default: 'narrator.mp4'
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  size?: 'sm' | 'md' | 'lg';
  startFrame?: number;          // When to show narrator (default: same as voiceover)
}

export interface SprintConfig {
  info: SprintInfo;
  overview: {
    title: string;        // e.g., "What's New in v4.0.2"
    items: OverviewItem[];
  };
  demos: DemoConfig[];
  summary: {
    stats: StatItem[];
    screenshotFile?: string;
  };
  credits: CreditSection[];
  audio: {
    voiceoverFile?: string;
    voiceoverStartFrame?: number;
    backgroundMusicFile?: string;
    backgroundMusicVolume?: number;
    chimeFile?: string;
    chimeFrame?: number;
  };
  narrator?: NarratorConfig;
}

// Video configuration
export interface VideoConfig {
  fps: number;
  width: number;
  height: number;
  durationSeconds: number;
}
