import { Composition } from "remotion";
import { DigitalSambaSkillDemo } from "./DigitalSambaSkillDemo";

// Video specs: 1920x1080, 30fps, ~2:53 (5190 frames)
const FPS = 30;
const WIDTH = 1920;
const HEIGHT = 1080;
const DURATION_FRAMES = 5190; // 173 seconds - extended for voiceover

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="DigitalSambaSkillDemo"
        component={DigitalSambaSkillDemo}
        durationInFrames={DURATION_FRAMES}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
