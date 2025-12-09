// Vignette overlay - subtle cinematic darkening around edges
export const Vignette: React.FC = () => {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: `radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.4) 100%)`,
        pointerEvents: 'none',
      }}
    />
  );
};
