// Tests for lib/components/DataChart.tsx: the line head, its timing and the brand theme.
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
const out = mkdtempSync(path.join(tmpdir(), 'datachart-test-'));

async function load(entry) {
  const file = path.join(out, path.basename(entry).replace(/\.tsx?$/, '.mjs'));
  const r = await esbuild.build({
    entryPoints: [entry], bundle: true, format: 'esm', platform: 'node', write: false, jsx: 'automatic',
    nodePaths: [path.join(TEMPLATE, 'node_modules')], logLevel: 'silent',
    banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
  });
  writeFileSync(file, r.outputFiles[0].text);
  return import(pathToFileURL(file).href);
}

const { lineHead, headFrameAt, chartTheme } = await load(path.join(ROOT, 'lib/components/DataChart.tsx'));

const points = [{ x: 0, y: 0 }, { x: 10, y: 100 }, { x: 20, y: 100 }, { x: 30, y: 400 }];
const keys = [{ frame: 30, x: 0 }, { frame: 60, x: 10 }, { frame: 90, x: 30 }];

test('the head sits on the keys and holds before and after them', () => {
  assert.deepEqual(lineHead(points, 0, keys), { x: 0, y: 0, progress: 0 });
  assert.equal(lineHead(points, 60, keys).x, 10);
  assert.equal(lineHead(points, 60, keys).y, 100);
  const end = lineHead(points, 200, keys);
  assert.equal(end.x, 30);
  assert.equal(end.y, 400);
  assert.equal(end.progress, 1);
});

test('y is interpolated between data points, and the head eases between keys', () => {
  const mid = lineHead(points, 75, keys); // halfway between the 60 and 90 keys: eased to x 20
  assert.equal(mid.x, 20);
  assert.equal(mid.y, 100);
  const early = lineHead(points, 35, keys); // ease-in: well short of the linear 1/6
  assert.ok(early.x > 0 && early.x < 10 / 6, `x ${early.x}`);
  assert.ok(Math.abs(early.y - early.x * 10) < 1e-9);
});

test('headFrameAt finds the first frame the line reaches x, inside the right key span', () => {
  assert.equal(headFrameAt(keys, 0), 30);
  assert.equal(headFrameAt(keys, 10), 60);
  const f = headFrameAt(keys, 20);
  assert.ok(f > 60 && f <= 90);
  assert.ok(lineHead(points, f, keys).x >= 20);
  assert.ok(lineHead(points, f - 1, keys).x < 20);
  assert.equal(headFrameAt(keys, 99), 90); // never reached: the last key
});

test('chartTheme maps brand colours for dark and light backgrounds', () => {
  const colors = { primary: '#ea580c', accent: '#3771e0', textDark: '#1e293b', textLight: '#64748b',
    bgLight: '#ffffff', bgDark: '#1a1a2e' };
  const dark = chartTheme(colors, { primary: 'Inter' });
  assert.equal(dark.bg, '#1a1a2e');
  assert.equal(dark.text, '#ffffff');
  const light = chartTheme(colors, { primary: 'Inter' }, 'light');
  assert.equal(light.bg, '#ffffff');
  assert.equal(light.text, '#1e293b');
  assert.equal(light.muted, '#64748b');
  assert.equal(chartTheme({ ...colors, accent: undefined }, { primary: 'Inter' }).accent, '#ea580c');
});

const { turnProgress } = await load(path.join(ROOT, 'lib/components/PageTurns.tsx'));

test('a page turn starts `lead` frames before its cue and finishes `turnFrames` later', () => {
  assert.equal(turnProgress(93, 100), 0);          // at - lead - 1: not started
  assert.equal(turnProgress(94, 100), 0);          // starts at at - lead
  assert.equal(turnProgress(110, 100), 1);         // at - lead + turnFrames: done
  const mid = turnProgress(100, 100);              // on the word: mostly turned (ease-out)
  assert.ok(mid > 0.6 && mid < 1, `mid ${mid}`);
  assert.equal(turnProgress(100, 100, 10, 0), 0);  // lead 0: starts on the word
});
