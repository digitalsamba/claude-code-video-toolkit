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
  dateRange: string;      // Date range (e.g., "24th Nov - 8th Dec")
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
  /** Per-scene audio file (e.g., 'scenes/03-demo.mp3'). Renders Audio within the scene's Sequence. */
  audioFile?: string;
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

export interface MazeDecorationConfig {
  enabled: boolean;
  corner?: 'top-right' | 'top-left';
  opacity?: number;             // Default: 0.18
  scale?: number;               // Default: 1
  primaryColor?: string;        // Default: theme primary color
  secondaryColor?: string;      // Default: theme background dark
}

export interface SprintConfig {
  info: SprintInfo & {
    /** Per-scene audio file for title slide (e.g., 'scenes/01-title.mp3') */
    audioFile?: string;
  };
  overview: {
    title: string;        // e.g., "What's New in v4.0.2"
    items: OverviewItem[];
    /** Per-scene audio file for overview slide (e.g., 'scenes/02-overview.mp3') */
    audioFile?: string;
  };
  demos: DemoConfig[];
  summary: {
    stats: StatItem[];
    screenshotFile?: string;
    /** Per-scene audio file for summary slide (e.g., 'scenes/summary.mp3') */
    audioFile?: string;
  };
  credits: CreditSection[];
  audio: {
    /** Single voiceover file (legacy mode). Use per-scene audioFile instead for new projects. */
    voiceoverFile?: string;
    voiceoverStartFrame?: number;
    backgroundMusicFile?: string;
    backgroundMusicVolume?: number;
    chimeFile?: string;
    chimeFrame?: number;
  };
  narrator?: NarratorConfig;
  mazeDecoration?: MazeDecorationConfig;
}

// Video configuration
export interface VideoConfig {
  fps: number;
  width: number;
  height: number;
  durationSeconds: number;
}
