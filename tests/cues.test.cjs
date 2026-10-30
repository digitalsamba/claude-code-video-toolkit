const assert = require('node:assert/strict');
const { test } = require('node:test');
const { parseCueDocument, cueFrame } = require('../scratch/cues-ts/cues.js');

test('maps generated millisecond cues at the actual composition FPS', () => {
  const document = parseCueDocument({ version: 1, fps: 30, cues: { click: { timeMs: 1050, frame: 32 } } });
  assert.equal(cueFrame({ document, name: 'click', fps: 30 }), 32);
  assert.equal(cueFrame({ document, name: 'click', fps: 60 }), 63);
  assert.equal(cueFrame({ document, name: 'click', fps: 29.97 }), 31);
});

test('rejects missing cues and invalid documents instead of guessing a frame', () => {
  const document = parseCueDocument({ version: 1, cues: { click: { timeMs: 1000 } } });
  assert.throws(() => cueFrame({ document, name: 'missing', fps: 30 }), /Unknown narration cue/);
  assert.throws(() => cueFrame({ document, name: 'toString', fps: 30 }), /Unknown narration cue/);
  assert.throws(() => cueFrame({ document, name: 'click', fps: 0 }), /FPS/);
  for (const bad of [null, [], { version: 2, cues: {} }, { version: 1, cues: {} },
    { version: 1, cues: { click: { timeMs: -1 } } },
    { version: 1, cues: { click: { timeMs: NaN } } },
    { version: 1, cues: { click: { timeMs: '1000' } } }]) {
    assert.throws(() => parseCueDocument(bad));
  }
});
