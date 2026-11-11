/**
 * Shared video components
 *
 * These components are building blocks for video templates.
 * Import into templates via: import { ComponentName } from '../../../lib/components';
 */

// Backgrounds
export { AnimatedBackground } from './AnimatedBackground';
export type { AnimatedBackgroundProps, BackgroundVariant } from './AnimatedBackground';

// Decorations
export { MazeDecoration } from './MazeDecoration';
export type { MazeDecorationProps } from './MazeDecoration';

// Overlays
export { Vignette } from './Vignette';
export type { VignetteProps } from './Vignette';

export { FilmGrain } from './FilmGrain';
export type { FilmGrainProps } from './FilmGrain';

export { LogoWatermark } from './LogoWatermark';
export type { LogoWatermarkProps } from './LogoWatermark';

export { Label } from './Label';
export type { LabelProps, LabelPosition, LabelSize } from './Label';

// Transitions
export { SlideTransition } from './SlideTransition';
export type { SlideTransitionProps, TransitionStyle } from './SlideTransition';

// Layouts
export { SplitScreen } from './SplitScreen';
export type { SplitScreenProps } from './SplitScreen';

// Captions (word-level JSON from tools/captions.py)
export { Captions, captionsFor } from './Captions';
export type { CaptionsProps, CaptionsConfig, CaptionWord } from './Captions';

// Narrator (needs refinement - see component docs)
export { NarratorPiP } from './NarratorPiP';
export type { NarratorPiPProps } from './NarratorPiP';

// Animations
export { Envelope } from './Envelope';
export type { EnvelopeProps } from './Envelope';

export { PointingHand } from './PointingHand';
export type { PointingHandProps } from './PointingHand';

// Synthetic UI demos (see .claude/skills/synthetic-ui-demo/)
export { Stage, cameraAt, typed } from './Stage';
export type { StageProps, CameraKeyframe } from './Stage';

export { Cursor, cursorAt, press, near } from './Cursor';
export type { CursorProps, CursorPoint } from './Cursor';

export { Terminal, terminalTheme, typedUntil } from './Terminal';
export type { TerminalProps, TerminalLine, TerminalTheme, TerminalKind } from './Terminal';

export { voice, cueWords } from './cues';
export type { Voice, VoicePhrase, CaptionWord, CueOptions } from './cues';

// Data charts (SVG, brand-themed, cue-able; see examples/data-chart-stars)
export { LineChart, BarChart, Counter, SplitBar, chartTheme, lineHead, headFrameAt } from './DataChart';
export type {
  LineChartProps, LineSeries, BarChartProps, CounterProps, SplitBarProps, SplitSegment,
  ChartTheme, LinePoint, HeadKey, ChartBand, ChartBar,
} from './DataChart';

// Turning one scene into new pages in place, on narration cues (between scenes: @remotion/transitions wipe())
export { PageTurns, turnProgress } from './PageTurns';
export type { PageTurnsProps, PageTurn } from './PageTurns';

// A person's animated photo (ltx2.py --input): plays once, holds its last good frame (parody-short skill)
export { LivingPhoto, livingFrame, livingRate } from './LivingPhoto';
export type { LivingPhotoProps } from './LivingPhoto';

// Blender renders as elements: one clip per shot, transparent if asked (tools/blender_render.py --clips)
export { BlenderClip, blenderClip } from './BlenderClip';
export type { BlenderClipProps, BlenderClipInfo, BlenderClipEntry, BlenderManifest } from './BlenderClip';

// Utilities
export { hexToRgba, SIZE_PRESETS, POSITION_PRESETS } from './utils';
export type { SizePreset, PositionPreset } from './utils';
