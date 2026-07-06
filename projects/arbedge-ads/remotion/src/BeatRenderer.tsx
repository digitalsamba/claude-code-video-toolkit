import React from "react";
import script from "../../script.json";
import { HookBeat } from "./beats/HookBeat";
import { OddsCompareBeat } from "./beats/OddsCompareBeat";
import { StatBeat } from "./beats/StatBeat";
import { ProofBeat } from "./beats/ProofBeat";
import { CtaBeat } from "./beats/CtaBeat";
import type { Beat } from "./lib/types";

export interface BeatRendererProps {
  beatId?: string;
}

export const BeatRenderer: React.FC<BeatRendererProps> = ({ beatId = "" }) => {
  const beat = (script.beats as Beat[]).find((b) => b.id === beatId);
  if (!beat) {
    return (
      <div style={{ background: "#00010f", color: "#fff", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32 }}>
        Beat not found: {beatId}
      </div>
    );
  }

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
};
