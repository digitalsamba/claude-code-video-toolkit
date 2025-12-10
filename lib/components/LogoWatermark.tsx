/**
 * LogoWatermark - Corner logo branding overlay
 *
 * Displays a small logo with optional label in the corner of the video.
 * Fades in after a specified frame for subtle branding.
 */

import { useCurrentFrame, interpolate, Img, staticFile } from 'remotion';
import { useTheme } from '../theme';

export interface LogoWatermarkProps {
  /** Path to logo file (relative to public/) */
  logoSrc: string;
  /** Optional text label next to logo */
  label?: string;
  /** Frame at which to start fading in. Default: 240 (8s at 30fps) */
  fadeInFrame?: number;
  /** Duration of fade in frames. Default: 30 */
  fadeInDuration?: number;
  /** Final opacity. Default: 0.6 */
  maxOpacity?: number;
  /** Position in frame. Default: top-left */
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export const LogoWatermark: React.FC<LogoWatermarkProps> = ({
  logoSrc,
  label,
  fadeInFrame = 240,
  fadeInDuration = 30,
  maxOpacity = 0.6,
  position = 'top-left',
}) => {
  const frame = useCurrentFrame();
  const theme = useTheme();

  const opacity = interpolate(
    frame,
    [fadeInFrame, fadeInFrame + fadeInDuration],
    [0, maxOpacity],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const positionStyles: Record<string, React.CSSProperties> = {
    'top-left': { top: 24, left: 24 },
    'top-right': { top: 24, right: 24 },
    'bottom-left': { bottom: 24, left: 24 },
    'bottom-right': { bottom: 24, right: 24 },
  };

  return (
    <div
      style={{
        position: 'absolute',
        ...positionStyles[position],
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        opacity,
      }}
    >
      <Img
        src={staticFile(logoSrc)}
        style={{
          width: 32,
          height: 32,
          objectFit: 'contain',
        }}
      />
      {label && (
        <span
          style={{
            fontSize: 14,
            fontWeight: 500,
            color: theme.colors.textLight,
            letterSpacing: '0.5px',
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
};
