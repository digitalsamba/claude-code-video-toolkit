/**
 * Vignette - Cinematic edge darkening overlay
 *
 * Adds subtle darkening around the edges of the frame for a
 * more polished, cinematic look.
 */

export interface VignetteProps {
  /** Intensity of the vignette effect (0-1). Default: 0.4 */
  intensity?: number;
  /** Size of the transparent center area (0-100%). Default: 50 */
  centerSize?: number;
}

export const Vignette: React.FC<VignetteProps> = ({
  intensity = 0.4,
  centerSize = 50,
}) => {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: `radial-gradient(ellipse at center, transparent ${centerSize}%, rgba(0,0,0,${intensity}) 100%)`,
        pointerEvents: 'none',
      }}
    />
  );
};
