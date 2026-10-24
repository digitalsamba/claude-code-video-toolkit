import { loadFont as loadInter } from '@remotion/google-fonts/Inter';
import { loadFont as loadMono } from '@remotion/google-fonts/JetBrainsMono';
import { loadFont as loadMath } from '@remotion/google-fonts/STIXTwoText';
import { brand } from './brand';

const inter = loadInter('normal', { weights: ['400', '500', '600', '700', '800', '900'], subsets: ['latin'] });
const mono = loadMono('normal', { weights: ['400', '500', '700'], subsets: ['latin'] });
const math = loadMath('normal', { weights: ['400', '500', '600'], subsets: ['latin'] });
loadMath('italic', { weights: ['400'], subsets: ['latin'] });

const c = brand.colors;

/** Semantic tokens. Components read these, never brand.colors directly. */
export const theme = {
  bg: c.bgDark,
  bgDeep: c.bgDarkAlt,
  panel: c.panel,
  text: c.textOnDark,
  muted: c.accent,
  faint: 'rgba(102, 155, 188, 0.35)',
  hairline: 'rgba(102, 155, 188, 0.22)',
  primary: c.primary,
  primaryLight: c.primaryLight,
  primaryDark: c.primaryDark,
  accent: c.accent,
  green: c.accentGreen,
  cream: c.cream,
  ink: c.ink,
  fonts: {
    sans: `${inter.fontFamily}, system-ui, sans-serif`,
    mono: `${mono.fontFamily}, ui-monospace, monospace`,
    math: `${math.fontFamily}, 'Times New Roman', serif`,
  },
  /** 1080x1920 layout zones (px). Reels/Shorts UI covers the bottom ~260px. */
  layout: {
    gutter: 60,
    headerTop: 92,
    stageTop: 300,
    stageBottom: 1090,
    /** Right of Max (side layout): stats and readouts */
    asideLeft: 410,
    asideTop: 1130,
    captionsTop: 1540,
  },
};

export type Theme = typeof theme;
