import React from "react";
import { Series } from "remotion";
import script from "../../script.json";
import { HookBeat } from "./beats/HookBeat";
import { OddsCompareBeat } from "./beats/OddsCompareBeat";
import { StatBeat } from "./beats/StatBeat";
import { ProofBeat } from "./beats/ProofBeat";
import { CtaBeat } from "./beats/CtaBeat";
import type { Beat } from "./lib/types";

function renderBeat(beat: Beat): React.ReactNode {
  switch (beat.type) {
    case "hook":
      return <HookBeat beat={beat} />;
    case "odds_compare":
      return <OddsCompareBeat beat={beat} />;
    case "stat":
      return <StatBeat beat={beat} />;
    case "proof":
      return <ProofBeat beat={beat} />;
    case "cta":
      return <CtaBeat beat={beat} />;
    default:
      return null;
  }
}

export const Ad: React.FC = () => {
  return (
    <Series>
      {(script.beats as Beat[]).map((beat) => (
        <Series.Sequence key={beat.id} durationInFrames={beat.duration_frames}>
          {renderBeat(beat)}
        </Series.Sequence>
      ))}
    </Series>
  );
};
