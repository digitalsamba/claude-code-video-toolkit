import { useCurrentFrame, interpolate, Img, staticFile } from 'remotion';
import { useTheme } from '../../config/theme';

interface LogoWatermarkProps {
  logoSrc: string;
  label?: string;
  fadeInFrame?: number;
}

export const LogoWatermark: React.FC<LogoWatermarkProps> = ({
  logoSrc,
  label,
  fadeInFrame = 240,
}) => {
  const frame = useCurrentFrame();
  const theme = useTheme();

  const opacity = interpolate(
    frame,
    [fadeInFrame, fadeInFrame + 30],
    [0, 0.6],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <div
      style={{
        position: 'absolute',
        top: 24,
        left: 24,
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
