#!/usr/bin/env node
/**
 * Render-baseline harness — detects visual changes across Remotion bumps.
 *
 * The idea: render a fixed composition (tests/render-baseline/) twice on the
 * SAME machine — once at the current pin, once at the candidate version — and
 * diff the two. Nothing is committed as a "golden" image, so cross-machine
 * font/GPU/Chrome noise never enters the picture; only the bump does.
 *
 * Commands (run from anywhere; paths are resolved from the repo root):
 *
 *   node scripts/render-baseline.mjs render <label>
 *       Render one PNG per block into tests/render-baseline/out/<label>/.
 *       Uses whatever Remotion version is installed in tests/render-baseline.
 *
 *   node scripts/render-baseline.mjs diff <labelA> <labelB> [--threshold=0]
 *       Compare the two sets. Writes diff PNGs + report.json/report.md into
 *       tests/render-baseline/out/diff-<labelA>-vs-<labelB>/.
 *       Exit 1 if any frame's changed-pixel share exceeds --threshold (percent).
 *       Default 0 → any change fails, which is what we want until we have seen
 *       what "noise" looks like on a few real bumps.
 *
 *   node scripts/render-baseline.mjs install <version>
 *       Rewrite remotion/@remotion/* pins in tests/render-baseline/package.json
 *       to <version> and npm install. Used by CI to set up the "after" side.
 *
 *   node scripts/render-baseline.mjs ab <versionB> [--threshold=0]
 *       Convenience: render "before" at the current pin, install <versionB>,
 *       render "after", diff, then restore the original pin. Local use.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROJECT = path.join(ROOT, 'tests', 'render-baseline');
const OUT = path.join(PROJECT, 'out');
const ENTRY = 'src/index.ts';
const COMP = 'Baseline';
const FRAMES_PER_BLOCK = 30;
const BLOCK_LABELS = ['easing', 'sequence-text', 'offthreadvideo', 'audio', 'transitions', 'components'];
const SCALE = 0.25;

function sh(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { cwd: PROJECT, stdio: 'inherit', shell: process.platform === 'win32', ...opts });
  if (r.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed (exit ${r.status})`);
  }
}

function installedVersion() {
  const pj = path.join(PROJECT, 'node_modules', 'remotion', 'package.json');
  return fs.existsSync(pj) ? JSON.parse(fs.readFileSync(pj, 'utf8')).version : null;
}

function pinnedVersion() {
  const pj = JSON.parse(fs.readFileSync(path.join(PROJECT, 'package.json'), 'utf8'));
  return pj.dependencies.remotion;
}

// ─── render ──────────────────────────────────────────────────

function render(label) {
  if (!label) throw new Error('render needs a label');
  const dir = path.join(OUT, label);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const ver = installedVersion();
  console.log(`\n▶ render "${label}" with remotion ${ver}`);
  // Bundle once, then render every still from the bundle (saves ~6 webpack runs).
  const bundle = path.join(OUT, `bundle-${label}`);
  fs.rmSync(bundle, { recursive: true, force: true });
  sh('npx', ['remotion', 'bundle', ENTRY, `--out-dir=${bundle}`, '--log=error']);
  BLOCK_LABELS.forEach((name, i) => {
    const frame = i * FRAMES_PER_BLOCK + Math.floor(FRAMES_PER_BLOCK / 2);
    const file = path.join(dir, `${String(i + 1).padStart(2, '0')}-${name}.png`);
    sh('npx', [
      'remotion', 'still', bundle, COMP, file,
      `--frame=${frame}`, `--scale=${SCALE}`, '--image-format=png', '--gl=swangle', '--log=error',
    ]);
  });
  fs.rmSync(bundle, { recursive: true, force: true });
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ label, remotion: ver, scale: SCALE, renderedAt: new Date().toISOString() }, null, 2));
  console.log(`  ${BLOCK_LABELS.length} stills → ${path.relative(ROOT, dir)}`);
}

// ─── diff ────────────────────────────────────────────────────

async function diff(a, b, thresholdPct = 0) {
  if (!a || !b) throw new Error('diff needs two labels');
  const { PNG } = await import(path.join(PROJECT, 'node_modules', 'pngjs', 'lib', 'png.js'));
  const pixelmatch = (await import(path.join(PROJECT, 'node_modules', 'pixelmatch', 'index.js'))).default;
  const dirA = path.join(OUT, a);
  const dirB = path.join(OUT, b);
  const dirD = path.join(OUT, `diff-${a}-vs-${b}`);
  fs.rmSync(dirD, { recursive: true, force: true });
  fs.mkdirSync(dirD, { recursive: true });
  const metaA = JSON.parse(fs.readFileSync(path.join(dirA, 'meta.json'), 'utf8'));
  const metaB = JSON.parse(fs.readFileSync(path.join(dirB, 'meta.json'), 'utf8'));

  const rows = [];
  let failed = false;
  for (const f of fs.readdirSync(dirA).filter((n) => n.endsWith('.png')).sort()) {
    const pa = PNG.sync.read(fs.readFileSync(path.join(dirA, f)));
    const pbPath = path.join(dirB, f);
    if (!fs.existsSync(pbPath)) {
      rows.push({ frame: f, status: 'missing-in-b', changedPct: 100 });
      failed = true;
      continue;
    }
    const pb = PNG.sync.read(fs.readFileSync(pbPath));
    if (pa.width !== pb.width || pa.height !== pb.height) {
      rows.push({ frame: f, status: 'size-mismatch', changedPct: 100, a: `${pa.width}x${pa.height}`, b: `${pb.width}x${pb.height}` });
      failed = true;
      continue;
    }
    const out = new PNG({ width: pa.width, height: pa.height });
    // Perceptual diff (what gates) plus an exact byte diff (for calibrating the threshold later).
    const changed = pixelmatch(pa.data, pb.data, out.data, pa.width, pa.height, { threshold: 0.1, includeAA: false });
    const exact = pixelmatch(pa.data, pb.data, null, pa.width, pa.height, { threshold: 0 });
    const pct = (changed / (pa.width * pa.height)) * 100;
    const status = pct > thresholdPct ? 'DIFFERS' : 'ok';
    if (status === 'DIFFERS') {
      failed = true;
      fs.writeFileSync(path.join(dirD, f.replace('.png', '.diff.png')), PNG.sync.write(out));
    }
    rows.push({ frame: f, status, changedPixels: changed, changedPct: Number(pct.toFixed(3)), exactPixels: exact });
  }

  const report = { a: metaA, b: metaB, thresholdPct, failed, frames: rows };
  fs.writeFileSync(path.join(dirD, 'report.json'), JSON.stringify(report, null, 2));

  const md = [
    `### Render baseline: remotion ${metaA.remotion} → ${metaB.remotion}`,
    '',
    `| Frame | Status | Changed px (perceptual) | % | Exact px |`,
    `|---|---|---:|---:|---:|`,
    ...rows.map((r) => `| ${r.frame} | ${r.status === 'ok' ? '✅ ok' : `⚠️ ${r.status}`} | ${r.changedPixels ?? '—'} | ${r.changedPct} | ${r.exactPixels ?? '—'} |`),
    '',
    failed
      ? `**Result: differs** (threshold ${thresholdPct}%). Diff images in \`${path.relative(ROOT, dirD)}\` — review before merging.`
      : `**Result: identical** within threshold ${thresholdPct}%.`,
  ].join('\n');
  fs.writeFileSync(path.join(dirD, 'report.md'), md + '\n');
  console.log('\n' + md);
  return !failed;
}

// ─── install ─────────────────────────────────────────────────

function install(version) {
  if (!version) throw new Error('install needs a version');
  const pjPath = path.join(PROJECT, 'package.json');
  const pj = JSON.parse(fs.readFileSync(pjPath, 'utf8'));
  for (const section of ['dependencies', 'devDependencies']) {
    for (const name of Object.keys(pj[section] || {})) {
      if (name === 'remotion' || name.startsWith('@remotion/')) pj[section][name] = version;
    }
  }
  fs.writeFileSync(pjPath, JSON.stringify(pj, null, 2) + '\n');
  console.log(`\n▶ install remotion ${version}`);
  sh('npm', ['install', '--no-audit', '--no-fund', '--loglevel=error']);
  const got = installedVersion();
  if (got !== version) throw new Error(`expected remotion ${version}, got ${got}`);
}

// ─── main ────────────────────────────────────────────────────

const [cmd, ...rest] = process.argv.slice(2);
const flags = Object.fromEntries(rest.filter((x) => x.startsWith('--')).map((x) => x.slice(2).split('=')));
const args = rest.filter((x) => !x.startsWith('--'));
const threshold = Number(flags.threshold ?? 0);

try {
  if (cmd === 'render') {
    render(args[0]);
  } else if (cmd === 'diff') {
    const ok = await diff(args[0], args[1], threshold);
    process.exit(ok ? 0 : 1);
  } else if (cmd === 'install') {
    install(args[0]);
  } else if (cmd === 'ab') {
    const original = pinnedVersion();
    if (installedVersion() !== original) install(original);
    render('before');
    let ok = false;
    try {
      install(args[0]);
      render('after');
      ok = await diff('before', 'after', threshold);
    } finally {
      install(original);
    }
    process.exit(ok ? 0 : 1);
  } else {
    console.error('usage: render-baseline.mjs render <label> | diff <a> <b> [--threshold=0] | install <version> | ab <version>');
    process.exit(2);
  }
} catch (err) {
  console.error(`\n✖ ${err.message}`);
  process.exit(1);
}
