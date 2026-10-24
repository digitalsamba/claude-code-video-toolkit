// Synthesise the template's small SFX pack with ffmpeg (no API key needed).
// Writes public/sfx/*.mp3. Swap any file for a richer one from tools/sfx.py
// (ElevenLabs) keeping the same name, and the cues pick it up.
//   node scripts/make-sfx.mjs
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'sfx');
mkdirSync(out, { recursive: true });

// name -> [lavfi source, filter chain]
const SFX = {
  // airy sweep for chapter changes and visual swaps
  whoosh: ['anoisesrc=d=0.6:c=pink:a=0.6', 'highpass=f=300,lowpass=f=5000,afade=t=in:d=0.25,afade=t=out:st=0.3:d=0.3,volume=0.9'],
  // counter tick
  tick: ['sine=f=2400:d=0.03', 'afade=t=out:st=0.005:d=0.025,volume=0.5'],
  // ceramic chip clack: two bright transients
  chip: ['anoisesrc=d=0.09:c=white:a=0.8', 'bandpass=f=3200:w=1800,afade=t=out:st=0.01:d=0.08,aecho=0.6:0.4:22:0.5,volume=1.4'],
  // pop for cards and bubbles
  pop: ['sine=f=520:d=0.12', "vibrato=f=1:d=0,asetrate=44100*1.6,aresample=44100,afade=t=out:st=0.04:d=0.08,volume=0.7"],
  // heavy stamp
  thud: ['sine=f=58:d=0.5', 'afade=t=out:st=0.04:d=0.45,volume=2.2'],
  // tape rewind for the cold open
  rewind: ['sine=f=900:d=0.7', "vibrato=f=28:d=0.9,afade=t=in:d=0.05,afade=t=out:st=0.45:d=0.25,volume=0.5"],
  // low whoosh-down for the crash
  drop: ['sine=f=420:d=0.7', "vibrato=f=6:d=0.3,afade=t=out:st=0.2:d=0.5,asetrate=44100*0.7,aresample=44100,volume=0.8"],
};

for (const [name, [src, filters]] of Object.entries(SFX)) {
  const file = join(out, `${name}.mp3`);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', src, '-af', filters, '-ar', '44100', '-ac', '1', '-b:a', '128k', file]);
  console.log(`sfx/${name}.mp3`);
}
