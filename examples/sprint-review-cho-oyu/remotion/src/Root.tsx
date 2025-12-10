import { Composition } from "remotion";
import { SprintReview } from "./SprintReview";

// Video configuration
// 30fps, 1920x1080, ~170 seconds total (~2:50) - Sprint Review Cut with Credits
// Add 80s for Director's Cut with Making Of
const FPS = 30;
const DURATION_SECONDS = 170;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SprintReview"
        component={SprintReview}
        durationInFrames={FPS * DURATION_SECONDS}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
