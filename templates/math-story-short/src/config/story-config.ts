/**
 * Episode config: "Why the casino always wins" (the example episode).
 *
 * The numbers in the narration come from src/episode/data.ts (seeded
 * simulation), so the charts always show what Max says. If you change the
 * seed, re-read the printed checkpoints there and update the narration.
 *
 * Workflow:
 *   npm run scripts                                   # narration -> public/audio/scenes/*.txt
 *   uv run tools/voiceover.py --provider qwen3 --tone warm --scene-dir <project>/public/audio/scenes --json
 *   uv run tools/captions.py <project>/public/audio/scenes      # word timing for the captions
 *   uv run tools/sync_timing.py --config <project>/src/config/story-config.ts --apply
 */

import type { Story } from './types';

export const story: Story = {
  title: 'Why the casino always wins',
  fps: 30,
  width: 1080,
  height: 1920,
  outroSeconds: 3,
  backgroundMusicFile: 'audio/music.mp3',
  musicVolume: 0.12,

  chapters: [
    { title: 'Max at the casino' },
    { title: 'Max keeps playing' },
    { title: 'Just bad luck?' },
    { title: 'Why the casino wins' },
    { title: 'Millions of spins' },
  ],

  scenes: [
    // ── Cold open: the payoff first, then rewind ───────────────
    {
      id: '00-hook',
      chapter: 0,
      hook: true,
      visual: 'hook',
      narration: 'Red or black. Fifty-fifty. So why did Max just lose two hundred and seventy-two dollars?',
      audioFile: 'audio/scenes/00-hook.mp3',
      durationSeconds: 6.5,
      max: { mood: 'shocked', layout: 'side', look: 'up', emote: '!' },
    },

    // ── 01 Max at the casino ───────────────────────────────────
    {
      id: '01-intro',
      chapter: 0,
      visual: 'table',
      narration: 'Max is at the casino, playing roulette. He always bets one dollar on red.',
      audioFile: 'audio/scenes/01-intro.mp3',
      durationSeconds: 6.5,
      max: { mood: 'happy', layout: 'hero', look: 'up' },
    },
    {
      id: '02-fair',
      chapter: 0,
      visual: 'table',
      narration: "Red or black. One chance in two. So Max figures it's a fair game. He should come out about even.",
      audioFile: 'audio/scenes/02-fair.mp3',
      durationSeconds: 8.5,
      max: { mood: 'thinking', layout: 'hero', look: 'right', thought: '1 chance in 2.\nFair game!' },
    },
    {
      id: '03-first-spins',
      chapter: 0,
      visual: 'table',
      narration: 'Each red wins him a dollar. Black, or green, loses one. After ten spins, he is up two dollars.',
      audioFile: 'audio/scenes/03-first-spins.mp3',
      durationSeconds: 8.5,
      max: { mood: 'happy', layout: 'side', look: 'up', pose: 'point' },
    },

    // ── 02 Max keeps playing ───────────────────────────────────
    {
      id: '04-hundred',
      chapter: 1,
      visual: 'bankroll',
      narration: "Max keeps playing. Here's his money, spin after spin. After a hundred spins: plus twenty-two dollars. Not bad.",
      audioFile: 'audio/scenes/04-hundred.mp3',
      durationSeconds: 9.5,
      max: { mood: 'smug', layout: 'side', look: 'up', pose: 'point' },
    },
    {
      id: '05-ten-thousand',
      chapter: 1,
      visual: 'bankroll',
      narration: 'After ten thousand spins? Minus two hundred and seventy-two dollars.',
      audioFile: 'audio/scenes/05-ten-thousand.mp3',
      durationSeconds: 7,
      max: { mood: 'shocked', layout: 'side', look: 'up', emote: '!' },
    },
    {
      id: '06-what-went-wrong',
      chapter: 1,
      visual: 'bankroll',
      narration: "What went wrong? Wasn't it one chance in two?",
      audioFile: 'audio/scenes/06-what-went-wrong.mp3',
      durationSeconds: 5,
      max: { mood: 'confused', layout: 'side', look: 'up', emote: '?' },
    },

    // ── 03 Just bad luck? ──────────────────────────────────────
    {
      id: '07-other-players',
      chapter: 2,
      visual: 'players',
      narration: "Maybe Max was just unlucky. Let's check the other players. Forty-nine of them bet on red all night, too.",
      audioFile: 'audio/scenes/07-other-players.mp3',
      durationSeconds: 9,
      max: { mood: 'thinking', layout: 'side', look: 'up' },
    },
    {
      id: '08-after-hundred',
      chapter: 2,
      visual: 'players',
      narration: 'After a hundred spins, nineteen of the fifty are still ahead.',
      audioFile: 'audio/scenes/08-after-hundred.mp3',
      durationSeconds: 5.5,
      max: { mood: 'happy', layout: 'side', look: 'up', pose: 'point' },
    },
    {
      id: '09-not-a-single-one',
      chapter: 2,
      visual: 'players',
      narration: 'After ten thousand spins? Not a single one.',
      audioFile: 'audio/scenes/09-not-a-single-one.mp3',
      durationSeconds: 6,
      max: { mood: 'shocked', layout: 'side', look: 'up', emote: '!' },
    },

    // ── 04 Why the casino wins ─────────────────────────────────
    {
      id: '10-look-closer',
      chapter: 3,
      visual: 'wheel',
      narration: 'So why does the casino always win? Look closer. Eighteen red. Eighteen black. And one green pocket: the zero.',
      audioFile: 'audio/scenes/10-look-closer.mp3',
      durationSeconds: 10.5,
      max: { mood: 'thinking', layout: 'side', look: 'up' },
    },
    {
      id: '11-thirty-seven',
      chapter: 3,
      visual: 'edge',
      narration: 'Imagine thirty-seven spins where every pocket comes up once. Max wins eighteen times, and loses nineteen.',
      audioFile: 'audio/scenes/11-thirty-seven.mp3',
      durationSeconds: 8.5,
      max: { mood: 'thinking', layout: 'side', look: 'up' },
    },
    {
      id: '12-house-edge',
      chapter: 3,
      visual: 'edge',
      narration: "So every thirty-seven spins, he loses a dollar. That's about two point seven cents per spin. It's called the house edge.",
      audioFile: 'audio/scenes/12-house-edge.mp3',
      durationSeconds: 10,
      max: { mood: 'confused', layout: 'side', look: 'up', pose: 'point' },
    },
    {
      id: '13-tiny',
      chapter: 3,
      visual: 'edge',
      narration: 'Tiny. But always against you.',
      audioFile: 'audio/scenes/13-tiny.mp3',
      durationSeconds: 4,
      max: { mood: 'sad', layout: 'side', look: 'up' },
    },

    // ── 05 Millions of spins ───────────────────────────────────
    {
      id: '14-millions',
      chapter: 4,
      visual: 'casino',
      narration: "And the casino doesn't play one night. It plays millions of spins. One-dollar bets on red. A so-called fair game.",
      audioFile: 'audio/scenes/14-millions.mp3',
      durationSeconds: 9.5,
      max: { mood: 'thinking', layout: 'side', look: 'up' },
    },
    {
      id: '15-hundred-million',
      chapter: 4,
      visual: 'casino',
      narration: 'A hundred million bets, and the casino keeps about two point seven million dollars.',
      audioFile: 'audio/scenes/15-hundred-million.mp3',
      durationSeconds: 7,
      max: { mood: 'shocked', layout: 'side', look: 'up', emote: '!' },
    },
    {
      id: '16-paycheck',
      chapter: 4,
      visual: 'casino',
      narration: "For the casino, it isn't luck. It's a paycheck.",
      audioFile: 'audio/scenes/16-paycheck.mp3',
      durationSeconds: 5.5,
      max: { mood: 'smug', layout: 'hero', look: 'right' },
    },
  ],
};
