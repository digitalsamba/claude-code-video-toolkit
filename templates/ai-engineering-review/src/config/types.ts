export type {
  Theme,
  ThemeColors,
  ThemeFonts,
  ThemeSpacing,
  ThemeBorderRadius,
  ThemeTypography,
} from '../../../../lib/theme';

export interface CaptionChunk {
  text: string;
  fromFrame: number;
  durationInFrames: number;
}

export interface SprintInfo {
  team: string;
  sprintName: string;
  dateRange: string;
  tagline?: string;
}

export interface LughConfig {
  portraitFile: string;
  talkingHeadFile?: string;
  circleCropFile?: string;
  introDurationSeconds: number;
  outroDurationSeconds: number;
  introCaptions?: CaptionChunk[];
  outroCaptions?: CaptionChunk[];
  introAudioFile?: string;
  outroAudioFile?: string;
}

export type Subsystem =
  | 'sambalive'
  | 'scaleway'
  | 'longarm'
  | 'mobile'
  | 'integrations'
  | 'account-center'
  | 'locales';

export interface TicketScene {
  key: string;
  title: string;
  subsystem: Subsystem;
  whatBroke: string;
  whatShipped: string;
  durationSeconds: number;
  demoFile?: string;
  audioFile?: string;
  captions?: CaptionChunk[];
}

export interface TitleCardConfig {
  durationSeconds: number;
  /** Optional override for the main heading. If not provided, uses info.sprintName. */
  title?: string;
  subtitle?: string;
  audioFile?: string;
  captions?: CaptionChunk[];
}

export interface ChapterCardConfig {
  label: string;
  sprintName: string;
  dateRange: string;
  durationSeconds: number;
  /** Indicates which ticket index starts this chapter (0-based). */
  startTicketIndex: number;
}

export interface SprintConfig {
  info: SprintInfo;
  lugh: LughConfig;
  titleCard: TitleCardConfig;
  chapters?: ChapterCardConfig[];
  tickets: TicketScene[];
  audio: {
    backgroundMusicFile?: string;
    backgroundMusicVolume?: number;
  };
  captions: {
    enabled: boolean;
    style?: 'inscribed' | 'modern';
  };
}

export interface VideoConfig {
  fps: number;
  width: number;
  height: number;
}
