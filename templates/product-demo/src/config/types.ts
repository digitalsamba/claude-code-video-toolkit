// Theme configuration (matches brand.json structure)
export interface ThemeColors {
  primary: string;
  primaryLight: string;
  accent?: string;
  textDark: string;
  textMedium: string;
  textLight: string;
  bgLight: string;
  bgDark: string;
  bgOverlay: string;
  divider: string;
  shadow: string;
}

export interface ThemeFonts {
  primary: string;
  mono: string;
}

export interface ThemeSpacing {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
  xxl: number;
}

export interface ThemeTypography {
  h1: { size: number; weight: number };
  h2: { size: number; weight: number };
  h3: { size: number; weight: number };
  body: { size: number; weight: number };
  label: { size: number; weight: number; letterSpacing?: number };
}

export interface Theme {
  colors: ThemeColors;
  fonts: ThemeFonts;
  spacing: ThemeSpacing;
  borderRadius: { sm: number; md: number; lg: number };
  typography: ThemeTypography;
}

// Product Demo Configuration
export interface ProductInfo {
  name: string;
  tagline: string;
  description?: string;
  logo?: string;
  logoLight?: string;
  website?: string;
  github?: string;
}

export interface SceneConfig {
  type: 'title' | 'problem' | 'solution' | 'demo' | 'feature' | 'stats' | 'cta';
  durationSeconds: number;
  // Scene-specific content
  content: TitleContent | ProblemContent | SolutionContent | DemoContent | FeatureContent | StatsContent | CTAContent;
}

export interface TitleContent {
  headline: string;
  subheadline?: string;
  logos?: Array<{ src: string; label?: string }>;
}

export interface ProblemContent {
  headline: string;
  problems: Array<{ icon: string; text: string }>;
  codeExample?: string[];
}

export interface SolutionContent {
  headline: string;
  description?: string;
  highlights?: string[];
}

export interface DemoContent {
  type: 'video' | 'browser' | 'terminal';
  videoFile?: string;
  label?: string;
  caption?: string;
}

export interface FeatureContent {
  headline: string;
  features: Array<{
    icon?: string;
    title: string;
    description: string;
  }>;
}

export interface StatsContent {
  headline?: string;
  stats: Array<{
    value: string;
    unit?: string;
    label: string;
    icon?: string;
    color?: string;
  }>;
}

export interface CTAContent {
  headline: string;
  tagline?: string;
  links?: Array<{
    type: 'github' | 'website' | 'docs' | 'custom';
    label: string;
    url: string;
    icon?: string;
  }>;
}

export interface AudioConfig {
  voiceoverFile?: string;
  voiceoverStartFrame?: number;
  backgroundMusicFile?: string;
  backgroundMusicVolume?: number;
}

export interface NarratorConfig {
  enabled: boolean;
  videoFile?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  size?: 'sm' | 'md' | 'lg';
  startFrame?: number;
}

export interface ProductDemoConfig {
  product: ProductInfo;
  scenes: SceneConfig[];
  audio?: AudioConfig;
  narrator?: NarratorConfig;
}

export interface VideoConfig {
  fps: number;
  width: number;
  height: number;
}
