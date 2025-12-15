/**
 * RGB Split Transition
 *
 * Chromatic aberration effect that separates RGB channels
 * with directional displacement. Creates a modern tech aesthetic
 * reminiscent of CRT displays and retro-futuristic visuals.
 *
 * Best for: Tech products, modern branding, energetic transitions
 */
import type {
  TransitionPresentation,
  TransitionPresentationComponentProps,
} from '@remotion/transitions';
import React, { useMemo } from 'react';
import { AbsoluteFill, interpolate } from 'remotion';

export type RgbSplitProps = {
  /** Direction of the split: 'horizontal' | 'vertical' | 'diagonal'. Default: 'horizontal' */
  direction?: 'horizontal' | 'vertical' | 'diagonal';
  /** Maximum pixel displacement. Default: 30 */
  displacement?: number;
  /** Include subtle blur on channels. Default: true */
  channelBlur?: boolean;
};

const RgbSplitPresentation: React.FC<
  TransitionPresentationComponentProps<RgbSplitProps>
> = ({ children, presentationDirection, presentationProgress, passedProps }) => {
  const {
    direction = 'horizontal',
    displacement = 30,
    channelBlur = true,
  } = passedProps;

  const progress = presentationDirection === 'exiting'
    ? 1 - presentationProgress
    : presentationProgress;

  // Split intensity peaks in the middle
  const splitIntensity = useMemo(() => {
    return interpolate(progress, [0, 0.5, 1], [0, 1, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
  }, [progress]);

  // Calculate channel offsets based on direction
  const getChannelOffset = (channel: 'red' | 'green' | 'blue') => {
    const multiplier = channel === 'red' ? -1 : channel === 'blue' ? 1 : 0;
    const offset = displacement * splitIntensity * multiplier;

    switch (direction) {
      case 'horizontal':
        return { x: offset, y: 0 };
      case 'vertical':
        return { x: 0, y: offset };
      case 'diagonal':
        return { x: offset * 0.7, y: offset * 0.7 };
    }
  };

  const redOffset = getChannelOffset('red');
  const blueOffset = getChannelOffset('blue');

  // Opacity for entering/exiting
  const opacity = presentationDirection === 'exiting'
    ? interpolate(progress, [0, 0.4], [1, 0], { extrapolateRight: 'clamp' })
    : interpolate(progress, [0.6, 1], [0, 1], { extrapolateLeft: 'clamp' });

  // Blur amount for channels (subtle motion blur effect)
  const blurAmount = channelBlur ? splitIntensity * 2 : 0;

  const containerStyle: React.CSSProperties = useMemo(() => ({
    width: '100%',
    height: '100%',
    position: 'relative',
  }), []);

  // Only show RGB separation when there's actual displacement
  const showSplit = splitIntensity > 0.05;

  return (
    <AbsoluteFill style={containerStyle}>
      {showSplit ? (
        <>
          {/* Red channel */}
          <AbsoluteFill
            style={{
              opacity: opacity,
              transform: `translate(${redOffset.x}px, ${redOffset.y}px)`,
              filter: blurAmount > 0 ? `blur(${blurAmount}px)` : undefined,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{
              width: '100%',
              height: '100%',
              filter: 'url(#rgbSplit-red)',
            }}>
              {children}
            </div>
          </AbsoluteFill>

          {/* Green channel (center, no offset) */}
          <AbsoluteFill
            style={{
              opacity: opacity,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{
              width: '100%',
              height: '100%',
              filter: 'url(#rgbSplit-green)',
            }}>
              {children}
            </div>
          </AbsoluteFill>

          {/* Blue channel */}
          <AbsoluteFill
            style={{
              opacity: opacity,
              transform: `translate(${blueOffset.x}px, ${blueOffset.y}px)`,
              filter: blurAmount > 0 ? `blur(${blurAmount}px)` : undefined,
              mixBlendMode: 'screen',
            }}
          >
            <div style={{
              width: '100%',
              height: '100%',
              filter: 'url(#rgbSplit-blue)',
            }}>
              {children}
            </div>
          </AbsoluteFill>
        </>
      ) : (
        <AbsoluteFill style={{ opacity }}>
          {children}
        </AbsoluteFill>
      )}

      {/* SVG filters for channel isolation */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <filter id="rgbSplit-red" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values="1 0 0 0 0
                      0 0 0 0 0
                      0 0 0 0 0
                      0 0 0 1 0"
            />
          </filter>
          <filter id="rgbSplit-green" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values="0 0 0 0 0
                      0 1 0 0 0
                      0 0 0 0 0
                      0 0 0 1 0"
            />
          </filter>
          <filter id="rgbSplit-blue" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values="0 0 0 0 0
                      0 0 0 0 0
                      0 0 1 0 0
                      0 0 0 1 0"
            />
          </filter>
        </defs>
      </svg>
    </AbsoluteFill>
  );
};

export const rgbSplit = (
  props: RgbSplitProps = {}
): TransitionPresentation<RgbSplitProps> => {
  return { component: RgbSplitPresentation, props };
};
