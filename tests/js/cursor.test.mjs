// Tests for lib/components/Cursor.tsx: a cursor path built from visits on narration cues (visitPath, boxMid).
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
const out = mkdtempSync(path.join(tmpdir(), 'cursor-test-'));

const r = await esbuild.build({
  entryPoints: [path.join(ROOT, 'lib/components/Cursor.tsx')], bundle: true, format: 'esm', platform: 'node',
  write: false, nodePaths: [path.join(TEMPLATE, 'node_modules')], logLevel: 'silent',
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
const file = path.join(out, 'Cursor.mjs');
writeFileSync(file, r.outputFiles[0].text);
const { visitPath, boxMid, cursorAt } = await import(pathToFileURL(file).href);

test('the cursor is on each target before its cue, and clicks on the cue', () => {
  const a = { x: 100, y: 100 };
  const b = { x: 600, y: 300 };
  const { path: p, clicks } = visitPath({ x: 0, y: 0 }, [
    { at: 60, to: a, click: true },
    { at: 140, to: b },
  ]);
  assert.deepEqual(clicks, [60]);
  assert.deepEqual(cursorAt(56, p), { f: 56, ...a });
  assert.deepEqual({ x: cursorAt(60, p).x, y: cursorAt(60, p).y }, a);
  assert.deepEqual({ x: cursorAt(136, p).x, y: cursorAt(136, p).y }, b);
});

test('the cursor holds still until it has to leave, and never leaves before the click is over', () => {
  const { path: p } = visitPath({ x: 0, y: 0 }, [
    { at: 60, to: { x: 100, y: 100 }, click: true },
    { at: 70, to: { x: 500, y: 100 } }, // too close for a full move
  ]);
  const frames = p.map((k) => k.f);
  assert.deepEqual(frames, [...frames].sort((x, y) => x - y), 'keyframes in order');
  const leave = p.find((k, i) => i > 0 && k.x === 100 && p[i + 1]?.x === 500);
  assert.ok(leave && leave.f >= 68, `holds through the click (left at ${leave?.f})`);
});

test('a visit with no time to travel still moves over at least six frames', () => {
  const { path: p } = visitPath({ x: 0, y: 0 }, [{ at: 2, to: { x: 300, y: 0 } }]);
  assert.ok(p[p.length - 1].f - p[p.length - 2].f >= 6);
});

test('boxMid is the middle of a box, a control 38 px high by default', () => {
  assert.deepEqual(boxMid({ x: 100, y: 200, w: 80 }), { x: 140, y: 219 });
  assert.deepEqual(boxMid({ x: 100, y: 200, w: 80, h: 20 }, -10, 5), { x: 130, y: 215 });
});
