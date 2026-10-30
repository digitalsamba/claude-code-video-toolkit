const assert = require('node:assert/strict');
const {test} = require('node:test');
const {VIDEO_FORMATS, videoLayout} = require('../scratch/formats-ts/formats.js');
test('all formats reserve space and change composition instead of cropping', () => {
  for (const dimensions of Object.values(VIDEO_FORMATS)) {
    const layout = videoLayout(dimensions);
    assert.ok(layout.contentWidth > 0 && layout.contentHeight > 0);
    assert.ok(layout.captionFontSize >= 28);
    assert.ok(layout.safe.bottom > 0);
  }
  assert.equal(videoLayout(VIDEO_FORMATS.landscape).wide, true);
  assert.equal(videoLayout(VIDEO_FORMATS.portrait).portrait, true);
  assert.equal(videoLayout(VIDEO_FORMATS.square).wide, false);
  assert.throws(() => videoLayout({width: 0, height: 1080}));
});
