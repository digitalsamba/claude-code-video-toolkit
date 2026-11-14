// Tests for lib/components/BlenderClip.tsx: the manifest lookup and where a cropped clip is placed.
//
//   node --test 'tests/js/*.test.mjs'   (needs `npm install` in templates/ds-release-notes for esbuild)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const TEMPLATE = path.join(ROOT, 'templates/ds-release-notes');
const require = createRequire(path.join(TEMPLATE, 'package.json'));
const esbuild = require('esbuild');
const out = mkdtempSync(path.join(tmpdir(), 'blenderclip-test-'));

const r = await esbuild.build({
  entryPoints: [path.join(ROOT, 'lib/components/BlenderClip.tsx')], bundle: true, format: 'esm', platform: 'node',
  write: false, nodePaths: [path.join(TEMPLATE, 'node_modules')], logLevel: 'silent',
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
const file = path.join(out, 'BlenderClip.mjs');
writeFileSync(file, r.outputFiles[0].text);
const { blenderClip, cropPlacement } = await import(pathToFileURL(file).href);

const manifest = {
  fps: 30, resolution: [1920, 1080],
  clips: [{ id: 'P02', file: 'P02.webm', frames: 90, seconds: 3, resolution: [1080, 1148], events: [{ frame: 12, kind: 'tap' }] }],
};
const pct = (v) => Number(String(v).replace('%', ''));

test('a crop places the whole clip so that the crop fills the box', () => {
  const clip = blenderClip(manifest, 'P02');
  const crop = { x: 414, y: 64, w: 380, h: 338 };
  const p = cropPlacement(clip, crop);
  // a box 270 px wide keeps the crop's shape, so it is 270 * 338 / 380 high
  const [bw, bh] = [270, (270 * crop.h) / crop.w];
  const [left, top] = [(pct(p.left) / 100) * bw, (pct(p.top) / 100) * bh];
  const [w, h] = [(pct(p.width) / 100) * bw, (pct(p.height) / 100) * bh];
  const k = bw / crop.w;
  assert.ok(Math.abs(w - clip.width * k) < 1e-6 && Math.abs(h - clip.height * k) < 1e-6, 'one scale both ways');
  // the crop's corners land on the box's corners
  assert.ok(Math.abs(left + crop.x * k) < 1e-6 && Math.abs(top + crop.y * k) < 1e-6);
  assert.ok(Math.abs(left + (crop.x + crop.w) * k - bw) < 1e-6 && Math.abs(top + (crop.y + crop.h) * k - bh) < 1e-6);
});

test('the whole clip as a crop changes nothing', () => {
  const clip = blenderClip(manifest, 'P02');
  const p = cropPlacement(clip, { x: 0, y: 0, w: 1080, h: 1148 });
  assert.deepEqual([pct(p.left), pct(p.top), pct(p.width), pct(p.height)], [0, 0, 100, 100]);
});

test('a crop outside the clip is named, with the clip size', () => {
  const clip = blenderClip(manifest, 'P02');
  assert.throws(() => cropPlacement(clip, { x: 900, y: 0, w: 380, h: 338 }), /P02.*not inside the clip \(1080x1148/);
  assert.throws(() => cropPlacement(clip, { x: 0, y: 0, w: 0, h: 10 }), /not inside the clip/);
});

test('a clip takes its size and transparency from the manifest', () => {
  const clip = blenderClip(manifest, 'P02');
  assert.equal(clip.width, 1080);
  assert.equal(clip.transparent, true);
  assert.equal(clip.event('tap'), 12);
  assert.throws(() => blenderClip(manifest, 'P99'), /no shot "P99".*P02/);
});
