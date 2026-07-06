import React from "react";
import { Composition } from "remotion";
import script from "../../script.json";
import { Ad } from "./Ad";
import { BeatRenderer, type BeatRendererProps } from "./BeatRenderer";
import type { Beat } from "./lib/types";

const FPS = 30;
const WIDTH = 1080;
const HEIGHT = 1920;

export const RemotionRoot: React.FC = () => {
  const beats = script.beats as Beat[];
  const totalFrames = beats.reduce((sum, b) => sum + b.duration_frames, 0);

  return (
    <>
      {/* Full ad — all beats in sequence */}
      <Composition
        id="FullAd"
        component={Ad}
        durationInFrames={totalFrames}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />

      {/* Individual beat compositions for per-beat iteration */}
      {beats.map((beat) => (
        <Composition
          key={beat.id}
          id={`Beat-${beat.id}`}
          component={BeatRenderer}
          durationInFrames={beat.duration_frames}
          fps={FPS}
          width={WIDTH}
          height={HEIGHT}
          defaultProps={{ beatId: beat.id }}
        />
      ))}
    </>
  );
};
