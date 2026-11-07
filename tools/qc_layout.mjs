#!/usr/bin/env node
/**
 * Layout and contrast audit for a bundled Remotion project. qc.py --layout runs it; it prints raw
 * per-sample observations as JSON and qc.py turns them into findings.
 *
 * It opens the bundle in Playwright and seeks frames with the hooks Remotion's own renderer uses
 * (remotion_setBundleMode, remotion_setFrame, remotion_renderReady). remotion_puppeteerTimeout is
 * left unset, so the page does not count as a render and <OffthreadVideo> plays as a plain video
 * with no frame server. At each sample it measures every visible text node: off-canvas, clipped by
 * its own box or a near container, its on-screen size, and its contrast against the pixels behind it.
 *
 * Adapted from the layout and contrast audits of HyperFrames' `check` (HeyGen, Apache-2.0).
 *
 *   node tools/qc_layout.mjs --bundle build/ [--composition ID | --prefer ID] [--every 1.0] [--max-samples 300]
 */
import { createRequire } from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
if (!args.bundle) {
  console.error('usage: qc_layout.mjs --bundle DIR [--composition ID] [--every SECONDS] [--max-samples N]');
  process.exit(2);
}
const every = Number(args.every ?? 1);
const maxSamples = Number(args['max-samples'] ?? 300);

function loadPlaywright() {
  try {
    return createRequire(path.join(here, '..', 'playwright', 'package.json'))('playwright');
  } catch {
    console.error('Playwright is not installed. Run: cd playwright && npm install');
    process.exit(3);
  }
}

async function launch(chromium) {
  // The user's Chrome first (no download), then Playwright's own Chromium.
  for (const opts of [{ channel: 'chrome' }, {}]) {
    try {
      return await chromium.launch(opts);
    } catch {
      // try the next
    }
  }
  console.error('No browser found. Install Chrome, or run: cd playwright && npx playwright install chromium');
  process.exit(3);
}

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif',
  '.svg': 'image/svg+xml', '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
  '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.otf': 'font/otf',
};

function serve(root) {
  // Static files with Range support, which <video> needs to seek.
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) {
        res.writeHead(404).end();
        return;
      }
      const type = TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream';
      const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? '');
      if (m) {
        const start = m[1] ? Number(m[1]) : 0;
        const end = m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
        res.writeHead(206, {
          'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1,
          'Content-Range': `bytes ${start}-${end}/${st.size}`,
        });
        fs.createReadStream(file, { start, end }).pipe(res);
      } else {
        res.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': st.size });
        fs.createReadStream(file).pipe(res);
      }
    });
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// Runs in the page at each sample. Returns one entry per visible, unoccluded text node.
function measureTexts() {
  const W = window.innerWidth, H = window.innerHeight;
  const out = [];
  const seen = new Set();
  const parseColor = (s) => {
    const m = /rgba?\(([^)]+)\)/.exec(s);
    if (!m) return null;
    const [r, g, b, a = 1] = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    return [r, g, b, a];
  };
  // A gradient wash (a 10% radial glow over the whole frame) doesn't hide what's under it
  const imageOpaque = (bgImage) => {
    if (bgImage === 'none') return false;
    if (bgImage.includes('url(')) return true;
    const alphas = [...bgImage.matchAll(/rgba?\(([^)]+)\)/g)].map((m) => {
      const parts = m[1].split(/[\s,/]+/).filter(Boolean);
      return parts.length > 3 ? Number(parts[3]) : 1;
    });
    return alphas.length > 0 && Math.max(...alphas) > 0.5;
  };
  const paints = (s, e) => {
    if (/^(IMG|VIDEO|CANVAS|svg|SVG)$/.test(e.tagName) || imageOpaque(s.backgroundImage)) return true;
    const c = parseColor(s.backgroundColor);
    return !!c && c[3] > 0.5;
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.textContent.replace(/\s+/g, ' ').trim();
    const el = node.parentElement;
    if (!text || !el || seen.has(el)) continue;
    seen.add(el);
    if (!/[\p{L}\p{N}]/u.test(text)) continue; // icons, arrows, emoji: no words to read
    if (el.closest('[data-qc-ignore]')) continue;

    // Effective opacity; mid-fade text is judged at another sample
    let opacity = 1;
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.display === 'none' || s.visibility === 'hidden') opacity = 0;
      opacity *= parseFloat(s.opacity);
    }
    if (opacity < 0.5) continue;

    const range = document.createRange();
    range.selectNodeContents(node);
    const rects = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
    if (!rects.length) continue;
    const r = {
      left: Math.min(...rects.map((x) => x.left)), top: Math.min(...rects.map((x) => x.top)),
      right: Math.max(...rects.map((x) => x.right)), bottom: Math.max(...rects.map((x) => x.bottom)),
    };
    const visW = Math.min(r.right, W) - Math.max(r.left, 0);
    const visH = Math.min(r.bottom, H) - Math.max(r.top, 0);
    if (visW <= 0 || visH <= 0) continue; // entirely off screen: not on screen, not a defect

    // Occluded: the first thing at the text's centre that paints is something else (a modal, a
    // card on top). Full-frame layers that paint nothing (cursor, camera) don't count.
    const cx = Math.max(0, Math.min(W - 1, (r.left + r.right) / 2));
    const cy = Math.max(0, Math.min(H - 1, (r.top + r.bottom) / 2));
    let occluded = false;
    for (const e of document.elementsFromPoint(cx, cy)) {
      if (e === el || el.contains(e) || e.contains(el)) break;
      const s = getComputedStyle(e);
      if (paints(s, e) && parseFloat(s.opacity) > 0.5) {
        occluded = true;
        break;
      }
    }
    if (occluded) continue;

    const cs = getComputedStyle(el);
    const box = el.getBoundingClientRect();
    const scale = el.offsetHeight ? box.height / el.offsetHeight : 1;
    const fontPx = parseFloat(cs.fontSize) * scale;

    const offCanvas = Math.max(-r.left, r.right - W, -r.top, r.bottom - H, 0);
    // Clipped by its own box or a container up to 3 levels up. A clip further up is a camera or
    // window crop (Stage), which frames the shot on purpose.
    let clipped = 0;
    let depth = 0;
    for (let e = el; e && e !== document.body && depth < 3; e = e.parentElement, depth++) {
      const s = getComputedStyle(e);
      if (/(hidden|clip)/.test(s.overflowX + s.overflowY)) {
        const c = e.getBoundingClientRect();
        clipped = Math.max(clipped, c.left - r.left, r.right - c.right, c.top - r.top, r.bottom - c.bottom);
      }
    }

    // Colour: none for gradient or image-filled text, which can't be judged from CSS
    let color = parseColor(cs.color);
    if (cs.webkitTextFillColor && !/currentcolor/i.test(cs.webkitTextFillColor)) {
      const fill = parseColor(cs.webkitTextFillColor);
      if (fill) color = fill;
    }
    // SVG text paints with `fill`, not `color` (charts, diagrams). A gradient or pattern fill
    // (url(#...)) can't be judged, like gradient HTML text.
    if (el instanceof SVGElement) {
      color = parseColor(cs.fill);
      if (color) color = [color[0], color[1], color[2], color[3] * parseFloat(cs.fillOpacity || '1')];
    }
    if (color && color[3] === 0) color = null;
    // Background: the first thing under the text that paints, from the paint stack at its centre
    // (a sibling layer can sit between the text and its ancestors' backgrounds). A solid, opaque
    // colour is exact; anything else (images, gradients, see-through layers) is sampled from the
    // screenshot around the text.
    let solidBg = null;
    for (const e of document.elementsFromPoint(cx, cy)) {
      if (el.contains(e) && e !== el) continue;
      const s = getComputedStyle(e);
      if (!paints(s, e)) continue;
      const c = parseColor(s.backgroundColor);
      let opaque = true;
      for (let a = e; a && a !== document.documentElement; a = a.parentElement) {
        if (parseFloat(getComputedStyle(a).opacity) < 0.99) opaque = false;
      }
      if (s.backgroundImage === 'none' && c && c[3] >= 0.99 && opaque) solidBg = c.slice(0, 3);
      break;
    }

    out.push({
      text: text.slice(0, 80),
      product: !!el.closest('[data-qc="product"]'),
      fontPx: Math.round(fontPx * 10) / 10,
      bold: parseInt(cs.fontWeight, 10) >= 700,
      bbox: [r.left, r.top, r.right, r.bottom].map(Math.round),
      offCanvas: Math.round(offCanvas),
      clipped: Math.round(Math.max(0, clipped)),
      color,
      opacity: Math.round(opacity * 100) / 100,
      solidBg,
    });
  }
  return out;
}

// Runs in the page: contrast for each text against its solid background or the median of a
// ring of screenshot pixels 3-6px outside its box.
async function contrastOf([pngB64, texts]) {
  const bmp = await createImageBitmap(await (await fetch(`data:image/png;base64,${pngB64}`)).blob());
  const ctx = new OffscreenCanvas(bmp.width, bmp.height).getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const lin = (c) => ((c /= 255) <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const lum = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  const ring = ([x0, y0, x1, y1]) => {
    const px = [];
    const grab = (x, y, w, h) => {
      x = Math.max(0, x);
      y = Math.max(0, y);
      w = Math.min(bmp.width - x, w);
      h = Math.min(bmp.height - y, h);
      if (w <= 0 || h <= 0) return;
      const d = ctx.getImageData(x, y, w, h).data;
      for (let i = 0; i < d.length; i += 4) px.push([d[i], d[i + 1], d[i + 2]]);
    };
    grab(x0 - 6, y0 - 6, x1 - x0 + 12, 3);
    grab(x0 - 6, y1 + 3, x1 - x0 + 12, 3);
    grab(x0 - 6, y0, 3, y1 - y0);
    grab(x1 + 3, y0, 3, y1 - y0);
    if (!px.length) return null;
    return [0, 1, 2].map((k) => px.map((p) => p[k]).sort((a, b) => a - b)[px.length >> 1]);
  };
  return texts.map((t) => {
    if (!t.color) return { ratio: null };
    const bg = t.solidBg ?? ring(t.bbox);
    if (!bg) return { ratio: null };
    const alpha = t.color[3] * t.opacity;
    const fg = t.color.slice(0, 3).map((c, k) => Math.round(c * alpha + bg[k] * (1 - alpha)));
    const [hi, lo] = [lum(fg), lum(bg)].sort((a, b) => b - a);
    return { ratio: Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100, fg, bg };
  });
}

const { chromium } = loadPlaywright();
const root = path.resolve(args.bundle);
const server = await serve(root);
const browser = await launch(chromium);
const result = { composition: null, samples: [], texts: [], failedRequests: [], pageErrors: [], notes: [] };
try {
  const page = await browser.newPage();
  page.on('response', (res) => {
    if (res.status() >= 400) result.failedRequests.push({ url: new URL(res.url()).pathname, status: res.status() });
  });
  page.on('pageerror', (err) => result.pageErrors.push(String(err.message).slice(0, 300)));
  // What Remotion's renderer sets before navigating (set-props-and-env.ts), minus the timeout
  // that would make the page think it is rendering.
  await page.addInitScript(() => {
    window.remotion_inputProps = '{}';
    window.remotion_envVariables = '{}';
    window.remotion_initialFrame = 0;
    window.remotion_attempt = 1;
    window.remotion_audioEnabled = false;
    window.remotion_videoEnabled = true;
    window.remotion_logLevel = 'error';
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);
  await page.waitForFunction(() => typeof window.getStaticCompositions === 'function', null, { timeout: 60000 });
  const comps = await page.evaluate(() => window.getStaticCompositions());
  // --composition must exist; --prefer (qc.py's guess from package.json) falls back to the first
  let comp = args.composition ? comps.find((c) => c.id === args.composition) : comps[0];
  if (!args.composition && args.prefer) {
    comp = comps.find((c) => c.id === args.prefer) ?? comps[0];
    if (comp && comp.id !== args.prefer) {
      result.notes.push(`package.json renders "${args.prefer}", which doesn't exist; audited "${comp.id}"`);
    }
  }
  if (!comp) {
    console.error(`Composition "${args.composition}" not found. Available: ${comps.map((c) => c.id).join(', ')}`);
    process.exitCode = 4;
  } else {
    result.composition = { id: comp.id, width: comp.width, height: comp.height, fps: comp.fps,
      durationInFrames: comp.durationInFrames };
    await page.setViewportSize({ width: comp.width, height: comp.height });
    await page.evaluate((c) => {
      window.remotion_setBundleMode({
        type: 'composition', compositionName: c.id,
        serializedResolvedPropsWithSchema: c.serializedResolvedPropsWithCustomSchema,
        compositionDurationInFrames: c.durationInFrames, compositionFps: c.fps,
        compositionHeight: c.height, compositionWidth: c.width,
        compositionDefaultCodec: c.defaultCodec, compositionDefaultOutName: c.defaultOutName,
        compositionDefaultVideoImageFormat: c.defaultVideoImageFormat,
        compositionDefaultPixelFormat: c.defaultPixelFormat,
        compositionDefaultProResProfile: c.defaultProResProfile,
        compositionDefaultSampleRate: c.defaultSampleRate,
      });
    }, comp);
    const ready = () => page.waitForFunction(() => window.remotion_renderReady === true, null, { timeout: 60000 });
    await ready();

    const step = Math.max(1, Math.round(comp.fps * every), Math.ceil(comp.durationInFrames / maxSamples));
    for (let f = Math.floor(step / 2); f < comp.durationInFrames; f += step) {
      await page.evaluate(([frame, id]) => window.remotion_setFrame(frame, id, 1), [f, comp.id]);
      await ready();
      await page.evaluate(() => document.fonts.ready);
      const texts = await page.evaluate(measureTexts);
      const png = await page.screenshot();
      const contrast = await page.evaluate(contrastOf, [png.toString('base64'), texts]);
      result.samples.push(f);
      texts.forEach((t, i) => {
        const { color, solidBg, opacity, ...rest } = t;
        result.texts.push({ frame: f, ...rest, ...contrast[i] });
      });
    }
  }
} finally {
  await browser.close();
  server.close();
}
process.stdout.write(JSON.stringify(result));
