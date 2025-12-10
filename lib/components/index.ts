/**
 * Shared video components
 *
 * These components are building blocks for video templates.
 * Import into templates via: import { ComponentName } from '../../../lib/components';
 */

// Backgrounds
export { AnimatedBackground } from './AnimatedBackground';
export type { AnimatedBackgroundProps, BackgroundVariant } from './AnimatedBackground';

// Overlays
export { Vignette } from './Vignette';
export type { VignetteProps } from './Vignette';

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

// Narrator (needs refinement - see component docs)
export { NarratorPiP } from './NarratorPiP';
export type { NarratorPiPProps } from './NarratorPiP';

// Utilities
export { hexToRgba, SIZE_PRESETS, POSITION_PRESETS } from './utils';
export type { SizePreset, PositionPreset } from './utils';
