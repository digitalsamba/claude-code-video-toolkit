/**
 * Core components for product-demo template
 *
 * Most components are re-exported from the shared library.
 * NarratorPiP remains local as it uses a config-based API that needs refinement.
 */

export {
  AnimatedBackground,
  Vignette,
  LogoWatermark,
} from '../../../../../lib/components';

export type {
  BackgroundVariant,
  VignetteProps,
  LogoWatermarkProps,
} from '../../../../../lib/components';

// NarratorPiP stays local - uses different API (config object vs props)
// TODO: Unify NarratorPiP API in lib component
export { NarratorPiP } from './NarratorPiP';
