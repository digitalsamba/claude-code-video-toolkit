/**
 * Video Project Schema
 *
 * Defines the structure of project.json for multi-session video production.
 * Runtime validation is defined by schema.json and tools/project_manifest.py.
 * Optional workflow fields allow partial legacy projects and early planning.
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
  | 'review'     // Scene-by-scene visual approval
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
  review: 'Reviewing scenes before generating audio',
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
export type VisualType = string;

/**
 * Scene types available across templates
 */
export type SceneType = string;

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
  assets?: string[];
  /** Recording instructions for Playwright demos */
  instructions?: string;
  [extension: string]: unknown;
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
  durationSeconds?: number;
  startSeconds?: number;
  title?: string | null;
  /** Visual configuration */
  visual?: SceneVisual;
  /** Voiceover text for this scene, null if no narration */
  narration?: string | null;
  /** Script path relative to the project directory. */
  scriptFile?: string;
  /** Audio and captions paths relative to public/, matching existing templates. */
  audioFile?: string;
  captionsFile?: string;
  audio?: VoiceoverAsset;
  /** Current status */
  status?: AssetStatus;
  reviewStatus?: string;
  [extension: string]: unknown;
}

// ============================================================================
// Audio Configuration
// ============================================================================

export type AudioStatus = 'needed' | 'present' | 'verified' | 'optional';

export interface AudioAsset {
  /** Relative path from public/, e.g., "audio/voiceover.mp3" */
  file?: string;
  /** Current status */
  status?: AudioStatus;
  durationSeconds?: number;
  captions?: string;
  provider?: string;
  voiceId?: string;
  [extension: string]: unknown;
}

export interface VoiceoverAsset extends AudioAsset {
  mode?: 'per_scene' | 'single';
  /** Word count from script for timing estimates */
  scriptWordCount?: number;
}

export interface AudioConfig {
  voiceover?: VoiceoverAsset;
  music?: AudioAsset;
  sfx?: AudioAsset[];
  [extension: string]: unknown;
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
  /** Absent in legacy input; the runtime loader returns 1. */
  schemaVersion?: 1;
  /** Project name (also folder name) */
  name: string;
  /** Template used: sprint-review, product-demo, etc. */
  template: string;
  /** Brand profile name from brands/ */
  brand?: string;
  /** ISO timestamp of creation */
  created?: string;
  /** ISO timestamp of last update */
  updated?: string;
  /** Current phase */
  phase?: ProjectPhase;
  engine?: string;
  composition?: string;
  output?: string;
  description?: string;
  format?: string | VideoFormat;
  render?: RenderConfig;
  publish?: PublishConfig;
  production?: ProductionConfig;

  /** Scene list with status */
  scenes: Scene[];

  /** Audio configuration */
  audio?: AudioConfig;

  /** Timing estimates */
  estimates?: {
    totalDurationSeconds?: number;
    /** Estimated voiceover duration in minutes */
    voiceoverMinutes?: number;
  };

  /** Session history for context */
  sessions?: SessionEntry[];
  [extension: string]: unknown;
}

export interface VideoFormat {
  width?: number;
  height?: number;
  fps?: number;
  aspectRatio?: string;
  [extension: string]: unknown;
}

export interface RenderConfig extends VideoFormat {
  engine?: string;
  entry?: string;
  composition?: string;
  file?: string;
  durationSeconds?: number;
}

export interface PublishConfig {
  platform?: string;
  title?: string;
  description?: string;
  tags?: string[];
  category?: string;
  thumbnail?: string;
  captions?: string;
  privacy?: 'private' | 'unlisted' | 'public';
  privacyStatus?: 'private' | 'unlisted' | 'public';
  publishAt?: string | null;
  uploadedAt?: string;
  videoId?: string;
  url?: string;
  playlist?: string;
  [extension: string]: unknown;
}

export interface ProductionTask {
  id: string;
  command: string[];
  outputs: string[];
  dependsOn?: string[];
  inputs?: string[];
  toolkitInputs?: string[];
  environmentInputs?: string[];
  cwd?: string;
  stdoutFile?: string;
  timeoutSeconds?: number;
  estimatedCostUsd?: number;
  label?: string;
  [extension: string]: unknown;
}

export interface ProductionConfig {
  version: 1;
  tasks: ProductionTask[];
  budgetUsd?: number;
  [extension: string]: unknown;
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
    schemaVersion: 1,
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
  const sessions = project.sessions ?? [];
  const lastSession = sessions[sessions.length - 1];

  // Update existing entry if same day, otherwise add new
  if (lastSession && lastSession.date === today) {
    lastSession.summary = summary;
  } else {
    sessions.push({ date: today, summary });
  }

  project.sessions = sessions;
  project.updated = new Date().toISOString();
  return project;
}
