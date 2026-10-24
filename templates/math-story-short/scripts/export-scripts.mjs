// Export each scene's narration from src/config/story-config.ts to
// public/audio/scenes/<id>.txt, the per-scene input `voiceover.py --scene-dir` reads.
// Run after editing narration: npm run scripts
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(root, 'src/config/story-config.ts'), 'utf8');

// Each scene object: id: '...', ... narration: "..." | '...'
const sceneRe = /id:\s*'([^']+)'[\s\S]*?narration:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g;
let n = 0;
for (const m of src.matchAll(sceneRe)) {
  const [, id, dq, sq] = m;
  const text = (dq ?? sq).replace(/\\(.)/g, '$1');
  const out = join(root, 'public/audio/scenes', `${id}.txt`);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, text + '\n');
  console.log(`${id}.txt  (${text.split(/\s+/).length} words)`);
  n++;
}
console.log(`\n${n} scene scripts written to public/audio/scenes/`);
