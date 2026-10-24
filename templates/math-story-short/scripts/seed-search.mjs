// Find a SEED (src/episode/data.ts) whose simulation hits the story beats.
// Edit the conditions below for a new episode, then: node scripts/seed-search.mjs
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const walk = (seed, n) => { const r = mulberry32(seed); const p = [0]; for (let i = 0; i < n; i++) p.push(p[i] + (r() * 37 < 18 ? 1 : -1)); return p; };
for (let s = 1; s < 20000; s++) {
  const m = walk(s * 1000, 10000);
  if (!(m[10] >= 2 && m[100] >= 16 && m[100] <= 30 && m[10000] <= -260 && m[10000] >= -360)) continue;
  let a100 = 0, a10k = 0;
  for (let k = 1; k < 50; k++) { const w = walk(s * 1000 + k, 10000); if (w[100] > 0) a100++; if (w[10000] > 0) a10k++; }
  if (m[100] > 0) a100++;
  if (a10k === 0 && a100 >= 18 && a100 <= 24) { console.log({ seed: s * 1000, m5: m[5], m10: m[10], m100: m[100], m10k: m[10000], a100 }); break; }
}
