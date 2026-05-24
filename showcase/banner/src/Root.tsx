import { Composition } from 'remotion';
import { ToolkitBanner } from './ToolkitBanner';
import { themes } from './themes';

const shared = {
  component: ToolkitBanner,
  durationInFrames: 150,
  fps: 30,
  width: 1080,
  height: 500,
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Primary — the banner rendered for the toolkit README */}
      <Composition
        id="ToolkitBanner"
        {...shared}
        defaultProps={{ theme: themes.amber }}
      />

      {/* Alternates — kept registered so we can revisit or re-render if branding changes */}
      <Composition id="Banner-Outrun" {...shared} defaultProps={{ theme: themes.outrun }} />
      <Composition id="Banner-Dusk" {...shared} defaultProps={{ theme: themes.dusk }} />
      <Composition id="Banner-Midnight" {...shared} defaultProps={{ theme: themes.midnight }} />
    </>
  );
};
