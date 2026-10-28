/**
 * Render one PNG per scene (end of scene by default) without a full video render.
 *
 *   node scripts/stills.mjs                 # out/stills/<id>.png, 14 frames before each cut
 *   node scripts/stills.mjs --at 0.5        # mid-scene instead
 *   node scripts/stills.mjs --frames 120,310   # explicit absolute frames
 *   node scripts/stills.mjs --out some/dir --scale 0.5
 */
import { bundle } from '@remotion/bundler';
import { renderStill, selectComposition } from '@remotion/renderer';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.resolve(root, opt('out', 'out/stills'));
const scale = Number(opt('scale', '1'));
const atFrac = opt('at', null);
const explicit = opt('frames', null);
mkdirSync(outDir, { recursive: true });

// Scene timing from the config, via a tiny TS-free parse (ids + durations in order)
import { readFileSync } from 'node:fs';
const cfg = readFileSync(path.join(root, 'src/config/story-config.ts'), 'utf8');
const fps = Number(/fps:\s*(\d+)/.exec(cfg)[1]);
const scenes = [];
for (const m of cfg.matchAll(/id:\s*'([^']+)'[\s\S]*?durationSeconds:\s*([\d.]+)/g)) scenes.push({ id: m[1], frames: Math.round(Number(m[2]) * fps) });
let from = 0;
for (const s of scenes) { s.from = from; from += s.frames; }

let targets;
if (explicit) targets = explicit.split(',').map((f) => ({ id: `f${f}`, frame: Number(f) }));
else targets = scenes.map((s) => ({ id: s.id, frame: atFrac ? s.from + Math.round(s.frames * Number(atFrac)) : s.from + s.frames - 14 }));

const serveUrl = await bundle({
  entryPoint: path.join(root, 'src/index.ts'),
  webpackOverride: (c) => ({ ...c, resolve: { ...c.resolve, modules: [path.join(root, 'node_modules'), ...(c.resolve?.modules || ['node_modules'])] } }),
});
const composition = await selectComposition({ serveUrl, id: 'MathStory' });
for (const t of targets) {
  const output = path.join(outDir, `${t.id}.png`);
  await renderStill({ composition, serveUrl, output, frame: t.frame, scale, imageFormat: 'png' });
  console.log(`${t.id}  frame ${t.frame}  -> ${path.relative(root, output)}`);
}
