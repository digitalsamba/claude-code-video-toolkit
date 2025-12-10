/**
 * NarratorPiP - Picture-in-picture presenter overlay
 *
 * Displays a video of a presenter/narrator in a corner of the frame.
 * Automatically fades in and out.
 *
 * @status needs-refinement
 * The API for this component needs review. Currently supports two patterns:
 * 1. Prop-based (videoFile, position, size as direct props)
 * 2. Config-based (config object with all settings)
 *
 * Future work should unify these approaches and add:
 * - Better timing control (startFrame, endFrame)
 * - Green screen / background removal
 * - Multiple narrator support
 * - Talking head detection / auto-framing
 */

import {
  OffthreadVideo,
  staticFile,
  getStaticFiles,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import { useTheme } from '../theme';
import { SIZE_PRESETS, POSITION_PRESETS, type SizePreset, type PositionPreset } from './utils';

export interface NarratorPiPProps {
  /** Video file name (relative to public/) */
  videoFile?: string;
  /** Corner position */
  position?: PositionPreset;
  /** Size preset */
  size?: SizePreset;
  /** Fade in duration in frames */
  fadeInFrames?: number;
  /** Fade out duration in frames */
  fadeOutFrames?: number;
  /** Frame to start showing narrator (default: 0) */
  startFrame?: number;
  /** Frame to stop showing narrator (default: end of video) */
  endFrame?: number;
  /** Whether narrator is enabled (default: true) */
  enabled?: boolean;
}

export const NarratorPiP: React.FC<NarratorPiPProps> = ({
  videoFile = 'narrator.mp4',
  position = 'bottom-right',
  size = 'md',
  fadeInFrames = 15,
  fadeOutFrames = 30,
  startFrame = 0,
  endFrame,
  enabled = true,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const theme = useTheme();

  // Check if enabled
  if (!enabled) {
    return null;
  }

  // Check if narrator video exists
  const staticFiles = getStaticFiles();
  const hasVideo = staticFiles.some((f) => f.name === videoFile);

  if (!hasVideo) {
    return null;
  }

  const effectiveEndFrame = endFrame ?? durationInFrames;
  const relativeFrame = frame - startFrame;

  // Don't render before start or after end
  if (frame < startFrame || frame > effectiveEndFrame) {
    return null;
  }

  const visibleDuration = effectiveEndFrame - startFrame;

  // Fade in/out animation
  const opacity = interpolate(
    relativeFrame,
    [0, fadeInFrames, visibleDuration - fadeOutFrames, visibleDuration],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const { width, height } = SIZE_PRESETS[size];
  const posStyle = POSITION_PRESETS[position];

  return (
    <div
      style={{
        position: 'absolute',
        ...posStyle,
        width,
        height,
        borderRadius: theme.borderRadius.lg,
        overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        border: '2px solid rgba(255,255,255,0.1)',
        opacity,
      }}
    >
      <OffthreadVideo
        src={staticFile(videoFile)}
        startFrom={startFrame}
        style={{
          width: '100%',
          height: '130%', // Slightly overflow to crop bottom (hands)
          objectFit: 'cover',
          objectPosition: 'center top',
        }}
        muted
      />
      {/* Gradient fade at bottom to soften any remaining visibility */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 40,
          background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.8))',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
};
