import { AbsoluteFill } from 'remotion';
import { SynthwaveBackground } from './SynthwaveBackground';
import { Wordmark } from './Wordmark';
import { Pipeline } from './Pipeline';
import { CRTOverlay } from './CRTOverlay';
import type { Theme } from './themes';

export const ToolkitBanner: React.FC<{ theme: Theme }> = ({ theme }) => {
  return (
    <AbsoluteFill>
      <SynthwaveBackground theme={theme} />
      <Wordmark theme={theme} />
      <Pipeline theme={theme} />
      <CRTOverlay theme={theme} />
    </AbsoluteFill>
  );
};
