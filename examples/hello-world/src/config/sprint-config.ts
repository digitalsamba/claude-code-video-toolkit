import type { SprintConfig, VideoConfig } from './types';

// ============================================================
// Hello World — minimal example config
// Edit this file to customize your video content
// ============================================================

export const sprintConfig: SprintConfig = {
  info: {
    name: 'Example',
    dateRange: 'Your First Video',
    product: 'Hello World',
    platform: 'Your First AI Video',
    version: 'v1.0',
  },

  overview: {
    title: "What's Inside",
    items: [
      { text: 'React-based ', highlight: 'video creation' },
      { text: 'Config-driven — ', highlight: 'edit text, not code' },
      { text: 'Zero API keys ', highlight: 'to get started' },
    ],
  },

  demos: [],

  summary: {
    stats: [
      { value: 25, label: 'Seconds' },
      { value: 0, label: 'API Keys Required' },
      { value: 100, label: '% Customizable' },
    ],
  },

  credits: [
    { category: 'Built With', items: ['Claude Code', 'Remotion'] },
    { category: 'Source', items: ['claude-code-video-toolkit'] },
  ],

  audio: {},
};

// Video output configuration
export const videoConfig: VideoConfig = {
  fps: 30,
  width: 1920,
  height: 1080,
  durationSeconds: 25,
};

// Helper to calculate frames
export const seconds = (s: number) => s * videoConfig.fps;
