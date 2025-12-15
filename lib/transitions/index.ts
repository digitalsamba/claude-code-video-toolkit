/**
 * Transitions Library
 *
 * Unified API for scene transitions in Remotion videos.
 * Combines official @remotion/transitions with custom presentations.
 *
 * Usage with TransitionSeries:
 * ```tsx
 * import { TransitionSeries, linearTiming } from '@remotion/transitions';
 * import { glitch, rgbSplit, zoomBlur, lightLeak } from '../../../lib/transitions';
 *
 * <TransitionSeries>
 *   <TransitionSeries.Sequence durationInFrames={90}>
 *     <SceneA />
 *   </TransitionSeries.Sequence>
 *   <TransitionSeries.Transition
 *     presentation={glitch()}
 *     timing={linearTiming({ durationInFrames: 20 })}
 *   />
 *   <TransitionSeries.Sequence durationInFrames={90}>
 *     <SceneB />
 *   </TransitionSeries.Sequence>
 * </TransitionSeries>
 * ```
 */

// Custom transitions
export { glitch } from './presentations/glitch';
export type { GlitchProps } from './presentations/glitch';

export { rgbSplit } from './presentations/rgb-split';
export type { RgbSplitProps } from './presentations/rgb-split';

export { zoomBlur } from './presentations/zoom-blur';
export type { ZoomBlurProps } from './presentations/zoom-blur';

export { lightLeak } from './presentations/light-leak';
export type { LightLeakProps } from './presentations/light-leak';

export { clockWipe } from './presentations/clock-wipe';
export type { ClockWipeProps } from './presentations/clock-wipe';

export { pixelate } from './presentations/pixelate';
export type { PixelateProps } from './presentations/pixelate';

// Re-export official transitions for convenience
export { slide } from '@remotion/transitions/slide';
export { fade } from '@remotion/transitions/fade';
export { wipe } from '@remotion/transitions/wipe';
export { flip } from '@remotion/transitions/flip';

// Re-export timing functions
export { linearTiming, springTiming, TransitionSeries } from '@remotion/transitions';

// Gallery/showcase components
export { TransitionGallery, transitionGalleryConfig, SingleTransitionPreview, transitionMap } from './TransitionGallery';
export type { TransitionName } from './TransitionGallery';
