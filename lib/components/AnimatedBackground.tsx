/**
 * AnimatedBackground - Animated background with floating shapes and gradients
 *
 * Provides visual interest behind video content. Supports multiple variants
 * for different aesthetic styles.
 *
 * Variants:
 * - subtle: Light, minimal movement (default for sprint-review)
 * - tech: Grid overlay with geometric shapes
 * - warm: Warm-toned gradients
 * - dark: Dark theme with floating outlined shapes (default for product-demo)
 */

import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion';
import { useTheme } from '../theme';
import { hexToRgba } from './utils';

export type BackgroundVariant = 'subtle' | 'tech' | 'warm' | 'dark';

export interface AnimatedBackgroundProps {
  variant?: BackgroundVariant;
  /** Whether to show grid lines overlay */
  showGrid?: boolean;
  /** Number of floating shapes (for dark variant) */
  shapeCount?: number;
}

export const AnimatedBackground: React.FC<AnimatedBackgroundProps> = ({
  variant = 'subtle',
  showGrid,
  shapeCount = 8,
}) => {
  const frame = useCurrentFrame();
  const theme = useTheme();

  // Dark variant uses floating outlined shapes
  if (variant === 'dark') {
    return <DarkVariant frame={frame} theme={theme} shapeCount={shapeCount} showGrid={showGrid} />;
  }

  // Light variants use gradient blobs
  return <LightVariant frame={frame} theme={theme} variant={variant} showGrid={showGrid} />;
};

// Light variant (subtle, tech, warm)
interface LightVariantProps {
  frame: number;
  theme: ReturnType<typeof useTheme>;
  variant: 'subtle' | 'tech' | 'warm';
  showGrid?: boolean;
}

const LightVariant: React.FC<LightVariantProps> = ({ frame, theme, variant, showGrid }) => {
  const getColors = () => {
    const primary = theme.colors.primary;
    const secondary = theme.colors.textLight;

    switch (variant) {
      case 'tech':
        return {
          bg: theme.colors.bgLight,
          shape1: hexToRgba(primary, 0.04),
          shape2: hexToRgba(secondary, 0.03),
          shape3: hexToRgba(theme.colors.primaryLight, 0.02),
        };
      case 'warm':
        return {
          bg: '#fffbf7',
          shape1: hexToRgba(primary, 0.05),
          shape2: hexToRgba(primary, 0.03),
          shape3: hexToRgba(theme.colors.primaryLight, 0.03),
        };
      case 'subtle':
      default:
        return {
          bg: theme.colors.bgLight,
          shape1: hexToRgba(primary, 0.03),
          shape2: hexToRgba(theme.colors.textDark, 0.02),
          shape3: hexToRgba(primary, 0.02),
        };
    }
  };

  const c = getColors();

  // Slow, organic movements
  const rotation1 = interpolate(frame, [0, 900], [0, 360], {
    extrapolateRight: 'extend',
  });
  const rotation2 = interpolate(frame, [0, 1200], [360, 0], {
    extrapolateRight: 'extend',
  });

  const float1Y = Math.sin(frame * 0.008) * 30;
  const float2Y = Math.cos(frame * 0.006) * 40;
  const float3X = Math.sin(frame * 0.005) * 50;

  const scale1 = 1 + Math.sin(frame * 0.004) * 0.1;
  const scale2 = 1 + Math.cos(frame * 0.003) * 0.08;

  const shouldShowGrid = showGrid ?? variant === 'tech';

  return (
    <AbsoluteFill style={{ backgroundColor: c.bg, overflow: 'hidden' }}>
      {/* Large slow-moving circle top-right */}
      <div
        style={{
          position: 'absolute',
          top: -200 + float1Y,
          right: -150,
          width: 600,
          height: 600,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${c.shape1} 0%, transparent 70%)`,
          transform: `rotate(${rotation1}deg) scale(${scale1})`,
        }}
      />

      {/* Medium shape bottom-left */}
      <div
        style={{
          position: 'absolute',
          bottom: -100 + float2Y,
          left: -100 + float3X,
          width: 500,
          height: 500,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${c.shape2} 0%, transparent 70%)`,
          transform: `rotate(${rotation2}deg) scale(${scale2})`,
        }}
      />

      {/* Subtle accent shape center-left */}
      <div
        style={{
          position: 'absolute',
          top: '40%',
          left: -200 + float3X * 0.5,
          width: 400,
          height: 400,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${c.shape3} 0%, transparent 60%)`,
          transform: `scale(${scale1 * 0.9})`,
        }}
      />

      {/* Grid pattern overlay */}
      {shouldShowGrid && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            backgroundImage: `
              linear-gradient(${hexToRgba(theme.colors.textLight, 0.03)} 1px, transparent 1px),
              linear-gradient(90deg, ${hexToRgba(theme.colors.textLight, 0.03)} 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
          }}
        />
      )}
    </AbsoluteFill>
  );
};

// Dark variant with floating outlined shapes
interface DarkVariantProps {
  frame: number;
  theme: ReturnType<typeof useTheme>;
  shapeCount: number;
  showGrid?: boolean;
}

const DarkVariant: React.FC<DarkVariantProps> = ({ frame, theme, shapeCount, showGrid = true }) => {
  // Generate floating shapes
  const shapes = Array.from({ length: shapeCount }, (_, i) => {
    const baseX = (i * 250 + 100) % 1920;
    const baseY = (i * 180 + 50) % 1080;
    const size = 100 + (i % 3) * 50;
    const speed = 0.3 + (i % 4) * 0.1;
    const phase = i * 0.8;

    const x = baseX + Math.sin(frame * speed * 0.02 + phase) * 30;
    const y = baseY + Math.cos(frame * speed * 0.015 + phase) * 20;
    const rotation = frame * speed * 0.5 + i * 45;
    const opacity = 0.03 + (i % 3) * 0.01;

    return { x, y, size, rotation, opacity, isCircle: i % 2 === 0 };
  });

  const accent = theme.colors.accent || theme.colors.primaryLight;

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* Gradient overlay */}
      <div
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          background: `
            radial-gradient(ellipse at 20% 30%, ${theme.colors.primary}15 0%, transparent 50%),
            radial-gradient(ellipse at 80% 70%, ${accent}10 0%, transparent 50%)
          `,
        }}
      />

      {/* Floating shapes */}
      {shapes.map((shape, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: shape.x,
            top: shape.y,
            width: shape.size,
            height: shape.size,
            borderRadius: shape.isCircle ? '50%' : '20%',
            border: `1px solid ${i % 2 === 0 ? theme.colors.primary : accent}`,
            opacity: shape.opacity,
            transform: `rotate(${shape.rotation}deg)`,
          }}
        />
      ))}

      {/* Grid lines */}
      {showGrid && (
        <div
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
            `,
            backgroundSize: '100px 100px',
          }}
        />
      )}
    </AbsoluteFill>
  );
};
