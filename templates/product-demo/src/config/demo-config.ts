import type { ProductDemoConfig, VideoConfig } from './types';

// Example configuration - customize for your product demo
export const demoConfig: ProductDemoConfig = {
  product: {
    name: 'Your Product',
    tagline: 'Build amazing things faster',
    // logo: 'images/logo.png',  // Uncomment and add your logo
    website: 'yourproduct.com',
    github: 'github.com/yourorg/yourproduct',
  },

  scenes: [
    // Title Scene
    {
      type: 'title',
      durationSeconds: 8,
      content: {
        headline: 'Build Video Features in Minutes',
        subheadline: 'From zero to production with AI',
        // logos: [
        //   { src: 'images/logo.png', label: 'Your Product' },
        // ],
      },
    },

    // Problem Scene
    {
      type: 'problem',
      durationSeconds: 10,
      content: {
        headline: 'The old way is complex...',
        problems: [
          { icon: 'API', text: 'Complex API Integration' },
          { icon: 'AUTH', text: 'Authentication Setup' },
          { icon: 'UI', text: 'Building UI Components' },
          { icon: 'ERR', text: 'Error Handling' },
        ],
        codeExample: [
          'const jwt = await generateToken({',
          '  teamId: process.env.TEAM_ID,',
          '  apiKey: process.env.API_KEY,',
          '  // ... 50 more lines',
          '});',
        ],
      },
    },

    // Solution Scene
    {
      type: 'solution',
      durationSeconds: 7,
      content: {
        headline: 'What if there was a better way?',
        description: 'Let AI handle the complexity',
        highlights: [
          'One command to get started',
          'AI understands your API',
          'Production-ready code',
        ],
      },
    },

    // Demo Scene(s) - uncomment when you have a demo video
    // {
    //   type: 'demo',
    //   durationSeconds: 30,
    //   content: {
    //     type: 'video',
    //     videoFile: 'demos/main-demo.mp4',
    //     label: 'Installation & Setup',
    //     caption: 'Watch how easy it is to get started',
    //   },
    // },

    // Stats Scene
    {
      type: 'stats',
      durationSeconds: 7,
      content: {
        headline: 'What You Get',
        stats: [
          { value: '5', unit: 'min', label: 'To Working App', icon: '⚡' },
          { value: '0', label: 'Config Required', icon: '☁️' },
          { value: '100%', label: 'Customizable', icon: '🎛️' },
        ],
      },
    },

    // CTA Scene
    {
      type: 'cta',
      durationSeconds: 8,
      content: {
        headline: 'Start Building Today',
        tagline: 'From zero to production in minutes',
        links: [
          { type: 'github', label: 'Get the code', url: 'github.com/yourorg/yourproduct' },
          { type: 'website', label: 'Learn more', url: 'yourproduct.com' },
        ],
      },
    },
  ],

  audio: {
    voiceoverFile: 'audio/voiceover.mp3',
    voiceoverStartFrame: 120, // 4 seconds in
    backgroundMusicFile: 'audio/background-music.mp3',
    backgroundMusicVolume: 0.12,
  },

  narrator: {
    enabled: false,
    videoFile: 'narrator.mp4',
    position: 'bottom-right',
    size: 'md',
    startFrame: 120,
  },
};

// Video settings
export const videoConfig: VideoConfig = {
  fps: 30,
  width: 1920,
  height: 1080,
};

// Calculate total duration from scenes
export function calculateTotalFrames(config: ProductDemoConfig, fps: number): number {
  return config.scenes.reduce((total, scene) => {
    return total + scene.durationSeconds * fps;
  }, 0);
}
