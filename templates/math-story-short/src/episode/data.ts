/**
 * Seeded roulette simulation behind every chart in the episode.
 *
 * SEED was picked so the story beats land (found by scripts/seed-search.mjs):
 *   Max after 10 spins: +$2 | after 100: +$22 | after 10,000: -$272
 *   players ahead after 100 spins: 19 / 50 | after 10,000: 0 / 50
 * Change the seed and the narration in story-config.ts must change with it.
 */

/** mulberry32: tiny seeded PRNG, identical to the one in scripts/seed-search.mjs */
const rng = (seed: number) => {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

export const SEED = 45000;
export const SPINS = 10000;
export const PLAYERS = 50;

export type Outcome = 'red' | 'black' | 'green';

/** Bankroll after each spin (index = spins played) and what each spin landed on. */
const walk = (seed: number, n: number) => {
  const r = rng(seed);
  const path = new Int16Array(n + 1);
  const outcomes: Outcome[] = [];
  for (let i = 0; i < n; i++) {
    const pocket = r() * 37;
    const win = pocket < 18;
    path[i + 1] = path[i] + (win ? 1 : -1);
    if (i < 40) outcomes.push(win ? 'red' : pocket < 19 ? 'green' : 'black');
  }
  return { path, outcomes };
};

const max = walk(SEED, SPINS);
export const maxPath = max.path;
export const maxOutcomes = max.outcomes;

/** 49 other players; Max is players[0]. */
export const players: Int16Array[] = [maxPath, ...Array.from({ length: PLAYERS - 1 }, (_, k) => walk(SEED + k + 1, SPINS).path)];

export const at = (path: Int16Array, x: number) => path[Math.max(0, Math.min(SPINS, Math.round(x)))];

export const aheadAt = (x: number) => players.filter((p) => at(p, x) > 0).length;

// ── Casino scale: kept cents per $1 bet, 10k → 100M bets ──────────
// Log-spaced checkpoints of a long random walk: each step adds the expected
// 1/37 per bet plus Gaussian noise with variance = bets in the step.
export const CASINO_MIN = 1e4;
export const CASINO_MAX = 1e8;
const casinoPts: { n: number; kept: number }[] = (() => {
  const r = rng(SEED + 999);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const pts: { n: number; kept: number }[] = [];
  let n = CASINO_MIN;
  let sum = n / 37 + gauss() * Math.sqrt(n) * 1.4;
  pts.push({ n, kept: sum });
  const steps = 240;
  for (let i = 1; i <= steps; i++) {
    const next = Math.round(Math.pow(10, 4 + (i / steps) * 4));
    const dn = next - n;
    sum += dn / 37 + gauss() * Math.sqrt(dn);
    n = next;
    pts.push({ n, kept: sum });
  }
  return pts;
})();

/** Cents kept per $1 bet after n bets (interpolated between checkpoints). */
export const keptCents = (n: number) => {
  const i = casinoPts.findIndex((p) => p.n >= n);
  if (i <= 0) return (casinoPts[0].kept / casinoPts[0].n) * 100;
  const a = casinoPts[i - 1];
  const b = casinoPts[i];
  const t = (n - a.n) / (b.n - a.n);
  return ((a.kept + (b.kept - a.kept) * t) / n) * 100;
};
/** Dollars the casino has kept after n bets. */
export const casinoWins = (n: number) => (keptCents(n) * n) / 100;
