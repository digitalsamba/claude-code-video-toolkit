// Tests for lib/components/cues.ts and the ds-release-notes scene built on it.
//
//   node --test 'tests/js/*.test.mjs'   (needs `npm install` in templates/ds-release-notes for esbuild)
//
// TypeScript is bundled on the fly with the template's own esbuild, so this runs on any
// Node version without a TS loader.
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
const out = mkdtempSync(path.join(tmpdir(), 'cues-test-'));

async function load(entry) {
  const file = path.join(out, path.basename(entry).replace(/\.tsx?$/, '.mjs'));
  const r = await esbuild.build({
    entryPoints: [entry], bundle: true, format: 'esm', platform: 'node', write: false,
    nodePaths: [path.join(TEMPLATE, 'node_modules')], logLevel: 'silent',
    loader: { '.png': 'empty', '.svg': 'empty', '.jpg': 'empty' },
    banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
  });
  writeFileSync(file, r.outputFiles[0].text);
  return import(pathToFileURL(file).href);
}

const { voice, cueWords } = await load(path.join(ROOT, 'lib/components/cues.ts'));

const caps = (pairs) => pairs.map(([text, startMs, endMs]) => ({ text, startMs, endMs: endMs ?? startMs + 300 }));
const line = voice([
  { file: 'a', at: 10, captions: caps([['Choose', 0], [' the', 400], [' "Apply', 1000], [' to', 1400], [' everyone",', 1600]]) },
  { file: 'b', at: 100, captions: caps([['Then', 0], [' apply', 500], [' it,', 900], [' you’ve', 1200], [' done.', 1500]]) },
]);

test('a word lands on its start frame, offset by the scene placement', () => {
  assert.equal(line.cue('choose'), 10);
  assert.equal(line.cue('the'), 10 + 12);           // 400 ms at 30 fps
  assert.equal(line.cue('Choose', { offset: -3 }), 7);
});

test('punctuation, quotes and case are ignored; phrases match word runs', () => {
  assert.equal(line.cue('"Apply to everyone"'), 10 + 30);
  assert.equal(line.cue('you\'ve done'), 100 + 36); // curly apostrophe in the captions
  assert.deepEqual(cueWords('Save & Send, "Invites"!'), ['save', 'send', 'invites']);
});

test('repeated words: first by default, then nth, after or in', () => {
  assert.equal(line.cue('apply'), 40);
  assert.equal(line.cue('apply', { nth: 2 }), 115);
  assert.equal(line.cue('apply', { after: 40 }), 115);
  assert.equal(line.cue('apply', { in: 'b' }), 115);
});

test("edge 'end' is the last word's end", () => {
  assert.equal(line.cue('to everyone', { edge: 'end' }), 10 + Math.round((1900 * 30) / 1000));
});

test('a phrase that is not said fails loudly and lists what was said', () => {
  assert.throws(() => line.cue('save'), /cue\("save"\): not found\. The narration says: choose@10/);
  assert.throws(() => line.cue('apply', { nth: 3 }), /only 2 match/);
  assert.throws(() => line.cue('apply', { in: 'a', after: 50 }), /in a/);
});

test('lines give the scene its <Audio> placements', () => {
  assert.deepEqual(line.lines, [['a', 10], ['b', 100]]);
});

test('the release 650 link scene keeps its hand-measured frames when cued from the captions', async () => {
  const scene = await load(path.join(TEMPLATE, 'src/scenes/FreeCustomLink.tsx'));
  // the values release 650 shipped with, measured by hand off the same captions
  assert.deepEqual(scene.LINK_CLICKS, [70, 404]);
  assert.equal(scene.LINK_TYPE_AT, 86);
  assert.deepEqual(scene.LINK_VOICE.lines, [['06a_link', 10], ['06b_link', 256]]);
});
