// Tests for lib/components/Stage.tsx: the address bar that changes inside a scene and the camera that follows the
// cursor in a narrow frame (the howto template's `screen` scenes use both).
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
const out = mkdtempSync(path.join(tmpdir(), 'stage-test-'));

const r = await esbuild.build({
  entryPoints: [path.join(ROOT, 'lib/components/Stage.tsx')], bundle: true, format: 'esm', platform: 'node',
  write: false, nodePaths: [path.join(TEMPLATE, 'node_modules')], logLevel: 'silent',
  banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
});
const file = path.join(out, 'Stage.mjs');
writeFileSync(file, r.outputFiles[0].text);
const { urlAt, followCamera, cameraAt } = await import(pathToFileURL(file).href);

test('one address shows on every frame', () => {
  assert.equal(urlAt('app.example.com/settings', 0), 'app.example.com/settings');
  assert.equal(urlAt('app.example.com/settings', 900), 'app.example.com/settings');
});

test('an address list shows the last entry whose frame has come', () => {
  const url = [{ at: 0, url: 'example.com' }, { at: 96, url: 'app.example.com/call' }, { at: 200, url: 'app.example.com/done' }];
  assert.equal(urlAt(url, 0), 'example.com');
  assert.equal(urlAt(url, 95), 'example.com');
  assert.equal(urlAt(url, 96), 'app.example.com/call');
  assert.equal(urlAt(url, 500), 'app.example.com/done');
});

test('before the first entry, an address list shows the first', () => {
  assert.equal(urlAt([{ at: 30, url: 'example.com' }], 0), 'example.com');
});

test('the follow camera has a keyframe for each cursor keyframe, at one zoom', () => {
  const path = [{ f: 0, x: 900, y: 600 }, { f: 40, x: 412, y: 318 }, { f: 60, x: 412, y: 318 }];
  const cam = followCamera(path, 3);
  assert.deepEqual(cam.map((k) => k.f), [0, 40, 60]);
  assert.ok(cam.every((k) => k.z === 3));
  assert.deepEqual([cam[1].x, cam[1].y], [412, 318]);
});

test('the follow camera holds still for a small move and travels for a large one', () => {
  const path = [{ f: 0, x: 400, y: 300 }, { f: 20, x: 450, y: 330 }, { f: 40, x: 900, y: 330 }];
  const cam = followCamera(path, 3);
  assert.deepEqual([cam[1].x, cam[1].y], [400, 300], 'a move inside the dead zone leaves the camera where it was');
  assert.deepEqual([cam[2].x, cam[2].y], [900, 330]);
  // and between keyframes the camera is where cameraAt puts it: still at 10, on its way at 30
  assert.equal(cameraAt(10, cam).x, 400);
  assert.ok(cameraAt(30, cam).x > 400 && cameraAt(30, cam).x < 900);
});

test('`ahead` keeps the cursor left of centre', () => {
  const cam = followCamera([{ f: 0, x: 400, y: 300 }], 3, { ahead: 60 });
  assert.equal(cam[0].x, 460);
});

test('an empty path still gives a camera', () => {
  assert.equal(followCamera([], 2).length, 1);
});
