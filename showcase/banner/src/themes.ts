export type Theme = {
  name: string;
  background: string;
  sun: {
    enabled: boolean;
    inner: string;
    mid: string;
    outer: string;
    bandColor: string;
  };
  grid: {
    vertical: string;
    horizontal: string;
    thickness: number;
  };
  wordmark: {
    color: string;
    glow: string[];
    aberration: {
      enabled: boolean;
      offset: number;
      left: string;
      right: string;
    };
  };
  pipeline: {
    ignite: string;
    settle: string;
    arrow: string;
    arrowShadow: string;
  };
  crt: {
    scanlineOpacity: number;
    vignetteStrength: number;
    sweepColor: string;
  };
};

// V1 — Outrun: classic synthwave, dialed back from the opening prototype
export const outrun: Theme = {
  name: 'outrun',
  background: '#120823',
  sun: {
    enabled: true,
    inner: '#ffd2e8',
    mid: '#d53b8a',
    outer: '#5b1982',
    bandColor: '#120823',
  },
  grid: {
    vertical: 'rgba(213, 59, 138, 0.55)',
    horizontal: 'rgba(74, 160, 204, 0.35)',
    thickness: 1.5,
  },
  wordmark: {
    color: '#fff8ff',
    glow: ['0 0 3px #fff', '0 0 10px #ff9fc8', '0 0 24px #d53b8a'],
    aberration: {
      enabled: true,
      offset: 2,
      left: '#7adef5',
      right: '#ff9fc8',
    },
  },
  pipeline: {
    ignite: '#7adef5',
    settle: '#ff9fc8',
    arrow: '#fff8ff',
    arrowShadow: '#d53b8a',
  },
  crt: {
    scanlineOpacity: 0.9,
    vignetteStrength: 0.55,
    sweepColor: 'rgba(122, 222, 245, 0.75)',
  },
};

// V2 — Dusk: muted sunset, one main accent, no chromatic aberration
export const dusk: Theme = {
  name: 'dusk',
  background: '#1a1530',
  sun: {
    enabled: true,
    inner: '#ffcfbd',
    mid: '#c47892',
    outer: '#2a2a4a',
    bandColor: '#1a1530',
  },
  grid: {
    vertical: 'rgba(196, 120, 146, 0.28)',
    horizontal: 'rgba(147, 162, 220, 0.22)',
    thickness: 1,
  },
  wordmark: {
    color: '#f8e8ff',
    glow: ['0 0 2px #fff', '0 0 14px #d4b3ff'],
    aberration: { enabled: false, offset: 0, left: '', right: '' },
  },
  pipeline: {
    ignite: '#f8e8ff',
    settle: '#d4b3ff',
    arrow: '#a192d0',
    arrowShadow: 'rgba(196, 120, 146, 0.6)',
  },
  crt: {
    scanlineOpacity: 0.7,
    vignetteStrength: 0.5,
    sweepColor: 'rgba(212, 179, 255, 0.6)',
  },
};

// V3 — Amber: mono phosphor CRT terminal, no sun, no synthwave
export const amber: Theme = {
  name: 'amber',
  background: '#0a0603',
  sun: {
    enabled: false,
    inner: '', mid: '', outer: '', bandColor: '',
  },
  grid: {
    vertical: 'rgba(255, 165, 82, 0.22)',
    horizontal: 'rgba(255, 165, 82, 0.18)',
    thickness: 1,
  },
  wordmark: {
    color: '#ffa552',
    glow: ['0 0 2px #ffd8a8', '0 0 10px #ffa552', '0 0 26px #ff7a1a'],
    aberration: { enabled: false, offset: 0, left: '', right: '' },
  },
  pipeline: {
    ignite: '#ffd8a8',
    settle: '#ffa552',
    arrow: '#ffa552',
    arrowShadow: 'rgba(255, 122, 26, 0.7)',
  },
  crt: {
    scanlineOpacity: 1.4,
    vignetteStrength: 0.7,
    sweepColor: 'rgba(255, 216, 168, 0.8)',
  },
};

// V4 — Midnight: minimal, no sun, cool cyan only
export const midnight: Theme = {
  name: 'midnight',
  background: '#0a0e1f',
  sun: {
    enabled: false,
    inner: '', mid: '', outer: '', bandColor: '',
  },
  grid: {
    vertical: 'rgba(110, 231, 255, 0.3)',
    horizontal: 'rgba(110, 231, 255, 0.22)',
    thickness: 1,
  },
  wordmark: {
    color: '#fff',
    glow: ['0 0 2px #fff', '0 0 12px #6ee7ff', '0 0 26px #3a9dbb'],
    aberration: { enabled: false, offset: 0, left: '', right: '' },
  },
  pipeline: {
    ignite: '#e0fbff',
    settle: '#6ee7ff',
    arrow: '#6ee7ff',
    arrowShadow: 'rgba(110, 231, 255, 0.7)',
  },
  crt: {
    scanlineOpacity: 0.6,
    vignetteStrength: 0.45,
    sweepColor: 'rgba(110, 231, 255, 0.7)',
  },
};

export const themes = { outrun, dusk, amber, midnight };
