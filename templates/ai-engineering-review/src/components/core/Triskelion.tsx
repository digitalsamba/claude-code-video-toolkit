import { useCurrentFrame, interpolate } from 'remotion';
import { brand } from '../../config/brand';

type Props = {
  size?: number;
  rotateSpeed?: number;
  strokeWidth?: number;
  color?: string;
};

export const Triskelion: React.FC<Props> = ({
  size = 420,
  rotateSpeed = 0.08,
  strokeWidth = 2.25,
  color = brand.colors.accent,
}) => {
  const frame = useCurrentFrame();
  const rotation = frame * rotateSpeed;

  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // One spiral arm — traced inward-curling via cubic Beziers, then mirrored 3x at 120° offsets.
  const spiralArm = 'M 100 100 C 100 140, 130 170, 160 160 C 182 152, 190 132, 180 118 C 172 107, 158 108, 154 120 C 151 129, 158 136, 166 133';

  return (
    <svg
      viewBox="0 0 200 200"
      width={size}
      height={size}
      style={{ opacity, display: 'block' }}
    >
      <defs>
        <radialGradient id="ring-glow" cx="0.5" cy="0.5" r="0.55">
          <stop offset="0%" stopColor="rgba(201, 179, 122, 0.08)" />
          <stop offset="70%" stopColor="rgba(201, 179, 122, 0)" />
        </radialGradient>
      </defs>

      <circle cx={100} cy={100} r={95} fill="url(#ring-glow)" />

      {/* Outer interlace ring — simplified as a pair of concentric knotwork circles */}
      <circle
        cx={100}
        cy={100}
        r={92}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth * 0.6}
        opacity={0.55}
      />
      <circle
        cx={100}
        cy={100}
        r={86}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth * 0.35}
        opacity={0.35}
      />

      {/* Small triangle dots at 12/4/8 o'clock */}
      {[0, 120, 240].map((angle) => {
        const rad = ((angle - 90) * Math.PI) / 180;
        const x = 100 + Math.cos(rad) * 75;
        const y = 100 + Math.sin(rad) * 75;
        return (
          <circle
            key={angle}
            cx={x}
            cy={y}
            r={2.2}
            fill={color}
            opacity={0.6}
          />
        );
      })}

      {/* Triskelion arms — rotate group as a whole */}
      <g transform={`rotate(${rotation} 100 100)`}>
        {[0, 120, 240].map((angle) => (
          <path
            key={angle}
            d={spiralArm}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            transform={`rotate(${angle} 100 100)`}
          />
        ))}
        {/* Central dot where the three arms meet */}
        <circle cx={100} cy={100} r={3.5} fill={color} />
      </g>
    </svg>
  );
};
