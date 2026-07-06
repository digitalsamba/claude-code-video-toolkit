export type BeatType = "hook" | "odds_compare" | "stat" | "proof" | "cta";

export interface HookBeat {
  id: string;
  type: "hook";
  narration?: string;
  headline: string;
  subline?: string;
  duration_frames: number;
}

export interface OddsRow {
  bookmaker: string;
  odds: number;
  is_outlier?: boolean;
}

export interface OddsCompareBeat {
  id: string;
  type: "odds_compare";
  narration?: string;
  event: string;
  sport: string;
  team: string;
  rows: OddsRow[];
  pinnacle_odds: number;
  ev_percent: number;
  duration_frames: number;
}

export interface StatBeat {
  id: string;
  type: "stat";
  narration?: string;
  value: string;
  context: string;
  duration_frames: number;
}

export interface Opportunity {
  event: string;
  bookmaker: string;
  odds: number;
  ev_percent: number;
  sport: string;
}

export interface ProofBeat {
  id: string;
  type: "proof";
  narration?: string;
  opportunities: Opportunity[];
  duration_frames: number;
}

export interface CtaBeat {
  id: string;
  type: "cta";
  narration?: string;
  price: string;
  duration_frames: number;
}

export type Beat =
  | HookBeat
  | OddsCompareBeat
  | StatBeat
  | ProofBeat
  | CtaBeat;

export interface Script {
  title: string;
  beats: Beat[];
}
