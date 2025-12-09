import { useCurrentFrame, interpolate, OffthreadVideo, staticFile } from 'remotion';
import type { NarratorConfig } from '../../config/types';

interface NarratorPiPProps {
  config: NarratorConfig;
  totalFrames: number;
}

const SIZE_MAP = {
  sm: { width: 240, height: 135 },
  md: { width: 320, height: 180 },
  lg: { width: 400, height: 225 },
};

const POSITION_MAP = {
  'bottom-right': { bottom: 40, right: 40 },
  'bottom-left': { bottom: 40, left: 40 },
  'top-right': { top: 40, right: 40 },
  'top-left': { top: 40, left: 40 },
};

export const NarratorPiP: React.FC<NarratorPiPProps> = ({ config, totalFrames }) => {
  const frame = useCurrentFrame();

  if (!config.enabled) return null;

  const size = SIZE_MAP[config.size || 'md'];
  const position = POSITION_MAP[config.position || 'bottom-right'];

  // Fade in over first 15 frames, fade out over last 15 frames
  const fadeOutStart = totalFrames - 60;
  const opacity = interpolate(
    frame,
    [0, 15, fadeOutStart, fadeOutStart + 30],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <div
      style={{
        position: 'absolute',
        ...position,
        width: size.width,
        height: size.height,
        borderRadius: 16,
        overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        border: '2px solid rgba(255,255,255,0.1)',
        opacity,
      }}
    >
      <OffthreadVideo
        src={staticFile(config.videoFile || 'narrator.mp4')}
        style={{
          width: '100%',
          height: '130%',
          objectFit: 'cover',
          objectPosition: 'center top',
        }}
        muted
      />
      {/* Gradient fade at bottom */}
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
