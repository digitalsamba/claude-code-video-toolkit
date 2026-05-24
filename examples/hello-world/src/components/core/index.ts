/**
 * Core components - re-exported from shared library
 *
 * These components are now maintained in lib/components/ for reuse across templates.
 * This file re-exports them to maintain existing import paths within the template.
 */

export {
  AnimatedBackground,
  SlideTransition,
  Label,
  NarratorPiP,
} from '../../../../../lib/components';

// Re-export types if needed by template code
export type {
  BackgroundVariant,
  TransitionStyle,
  LabelPosition,
  LabelSize,
} from '../../../../../lib/components';
