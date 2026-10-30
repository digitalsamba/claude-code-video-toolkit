/** Responsive video layout and conservative, configurable social safe zones. */
export const VIDEO_FORMATS = {
  landscape: {width: 1920, height: 1080},
  portrait: {width: 1080, height: 1920},
  square: {width: 1080, height: 1080},
} satisfies Record<string, {width: number; height: number}>;
export type VideoFormat = keyof typeof VIDEO_FORMATS;

export function videoLayout({width, height}: {width: number; height: number}) {
  if (![width, height].every(value => Number.isFinite(value) && value > 0)) {
    throw new Error('Video dimensions must be positive');
  }
  const portrait = height > width * 1.2;
  const wide = width > height * 1.3;
  const shortSide = Math.min(width, height);
  const safe = {
    top: Math.round(height * (portrait ? 0.08 : 0.06)),
    bottom: Math.round(height * (portrait ? 0.20 : 0.13)),
    left: Math.round(width * 0.07),
    right: Math.round(width * (portrait ? 0.13 : 0.07)),
  };
  return {portrait, wide, safe, gap: Math.round(shortSide * .045),
    fontSize: Math.round(shortSide * .065), captionFontSize: Math.round(shortSide * .043),
    contentWidth: width - safe.left - safe.right, contentHeight: height - safe.top - safe.bottom};
}
