/**
 * Video Project Schema
 *
 * Defines the structure of project.json for multi-session video production.
 * This schema is the source of intent - filesystem is source of truth for assets.
 */

// ============================================================================
// Project Lifecycle
// ============================================================================

/**
 * Project phases in order of progression
 */
export type ProjectPhase =
  | 'planning'   // Scenes being defined, script being written
  | 'assets'     // Recording demos, gathering materials
  | 'audio'      // Generating voiceover, music
  | 'editing'    // Adjusting timing, config, previewing
  | 'rendering'  // Final output in progress
  | 'complete';  // Done

/**
 * Phase descriptions for display
 */
export const PHASE_DESCRIPTIONS: Record<ProjectPhase, string> = {
  planning: 'Planning scenes and writing script',
  assets: 'Recording demos and gathering assets',
  audio: 'Generating voiceover and music',
  editing: 'Adjusting timing and previewing',
  rendering: 'Rendering final video',
  complete: 'Video complete',
};

// ============================================================================
// Scene Configuration
// ============================================================================

/**
 * Visual types for scenes
 */
export type VisualType =
  | 'slide'       // Template-generated slide (no asset needed)
  | 'playwright'  // Browser recording via Playwright
  | 'external'    // User-provided video file
  | 'screenshot'; // Static image capture

/**
 * Scene types available across templates
 */
export type SceneType =
  // Common
  | 'title'
  | 'overview'
  | 'demo'
  | 'split-demo'
  | 'summary'
  | 'credits'
  // Product demo specific
  | 'problem'
  | 'solution'
  | 'feature'
  | 'stats'
  | 'cta';

/**
 * Asset status for tracking progress
 */
export type AssetStatus =
  | 'ready'         // No asset needed (slides) or asset verified
  | 'asset-needed'  // Asset required but not yet created
  | 'asset-present' // File exists but not verified in preview
  | 'asset-missing';// Was present but now missing (error state)

/**
 * Scene visual configuration
 */
export interface SceneVisual {
  type: VisualType;
  /** Relative path from public/, e.g., "demos/dark-mode.mp4" */
  asset?: string;
  /** Recording instructions for Playwright demos */
  instructions?: string;
}

/**
 * Individual scene in the video
 */
export interface Scene {
  /** Unique identifier, e.g., "dark-mode-demo" */
  id: string;
  /** Scene type determines component used */
  type: SceneType;
  /** Estimated duration in seconds */
  durationSeconds: number;
  /** Visual configuration */
  visual: SceneVisual;
  /** Voiceover text for this scene, null if no narration */
  narration: string | null;
  /** Current status */
  status: AssetStatus;
}

// ============================================================================
// Audio Configuration
// ============================================================================

export type AudioStatus = 'needed' | 'present' | 'verified' | 'optional';

export interface AudioAsset {
  /** Relative path from public/, e.g., "audio/voiceover.mp3" */
  file: string;
  /** Current status */
  status: AudioStatus;
}

export interface VoiceoverAsset extends AudioAsset {
  /** Word count from script for timing estimates */
  scriptWordCount?: number;
}

export interface AudioConfig {
  voiceover: VoiceoverAsset;
  music?: AudioAsset;
  sfx?: AudioAsset[];
}

// ============================================================================
// Session History
// ============================================================================

/**
 * Session entry for tracking work history
 */
export interface SessionEntry {
  /** ISO date string (YYYY-MM-DD) */
  date: string;
  /** Brief summary of what was done */
  summary: string;
}

// ============================================================================
// Project Root
// ============================================================================

/**
 * Complete project configuration stored in project.json
 */
export interface VideoProject {
  /** Project name (also folder name) */
  name: string;
  /** Template used: sprint-review, product-demo, etc. */
  template: string;
  /** Brand profile name from brands/ */
  brand: string;
  /** ISO timestamp of creation */
  created: string;
  /** ISO timestamp of last update */
  updated: string;
  /** Current phase */
  phase: ProjectPhase;

  /** Scene list with status */
  scenes: Scene[];

  /** Audio configuration */
  audio: AudioConfig;

  /** Timing estimates */
  estimates: {
    totalDurationSeconds: number;
    /** Estimated voiceover duration in minutes */
    voiceoverMinutes?: number;
  };

  /** Session history for context */
  sessions: SessionEntry[];
}

// ============================================================================
// Scanner Types (for Claude's project scanning logic)
// ============================================================================

/**
 * Project health status after scanning
 */
export type ProjectHealth =
  | 'ready'     // Can proceed to next phase
  | 'blocked'   // Missing required assets
  | 'stale'     // No updates in 7+ days
  | 'complete'; // Finished

/**
 * Result of scanning a single project
 */
export interface ProjectScanResult {
  name: string;
  path: string;
  phase: ProjectPhase;
  health: ProjectHealth;

  /** Scene statistics */
  scenes: {
    total: number;
    ready: number;
    needsAsset: number;
    hasAsset: number;
  };

  /** Audio status */
  audio: {
    voiceoverStatus: AudioStatus;
    musicStatus: AudioStatus;
  };

  /** What's blocking progress */
  blockers: string[];
  /** Suggested next actions */
  nextActions: string[];

  /** Time context */
  lastModified: Date;
  daysSinceUpdate: number;
}

// ============================================================================
// Utilities
// ============================================================================

/**
 * Create a new project.json structure
 */
export function createProject(
  name: string,
  template: string,
  brand: string,
): VideoProject {
  const now = new Date().toISOString();
  return {
    name,
    template,
    brand,
    created: now,
    updated: now,
    phase: 'planning',
    scenes: [],
    audio: {
      voiceover: {
        file: 'audio/voiceover.mp3',
        status: 'needed',
      },
    },
    estimates: {
      totalDurationSeconds: 0,
    },
    sessions: [
      {
        date: now.split('T')[0],
        summary: 'Project created',
      },
    ],
  };
}

/**
 * Add a session entry to project
 */
export function addSession(
  project: VideoProject,
  summary: string,
): VideoProject {
  const today = new Date().toISOString().split('T')[0];
  const lastSession = project.sessions[project.sessions.length - 1];

  // Update existing entry if same day, otherwise add new
  if (lastSession && lastSession.date === today) {
    lastSession.summary = summary;
  } else {
    project.sessions.push({ date: today, summary });
  }

  project.updated = new Date().toISOString();
  return project;
}
