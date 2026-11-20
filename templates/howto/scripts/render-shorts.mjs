/**
 * Render a 9:16 Short for every chapter: out/short-1.mp4, out/short-2.mp4, ...
 *
 *     npm run render:shorts                      # all of them
 *     npm run render:shorts -- --log error       # anything after `--` goes to each `remotion render`
 *     npm run render:shorts -- --only 2,4        # just these chapters
 *
 * It stops at the first render that fails and exits with its code, and it checks that every file it names was
 * written by this run: an old short-3.mp4 left by an earlier run passes every check that only looks at the file.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
let only = null;
const k = args.indexOf('--only');
if (k >= 0) {
  only = (args[k + 1] ?? '').split(',').map((n) => Number(n.trim())).filter((n) => Number.isInteger(n) && n > 0);
  args.splice(k, 2);
  if (!only.length) {
    console.error('--only takes chapter numbers, e.g. --only 2,4');
    process.exit(2);
  }
}

const read = spawnSync(process.execPath, [join(root, 'scripts', 'timeline.mjs')], { cwd: root, encoding: 'utf8' });
if (read.status !== 0) {
  console.error(read.stderr || 'scripts/timeline.mjs failed');
  process.exit(read.status ?? 1);
}
const chapters = JSON.parse(read.stdout).chapters.length;
const which = (only ?? Array.from({ length: chapters }, (_, i) => i + 1)).filter((n) => n <= chapters);
if (!which.length) {
  console.error(`No such chapter: the episode has ${chapters}`);
  process.exit(2);
}

const done = [];
for (const n of which) {
  const out = join('out', `short-${n}.mp4`);
  const started = Date.now();
  console.log(`\nShort ${n} of ${chapters} -> ${out}`);
  // one command string, so it runs the same through cmd.exe on Windows
  const run = spawnSync(['npx', 'remotion', 'render', 'src/index.ts', `Short-${n}`, out, ...args].join(' '), { cwd: root, stdio: 'inherit', shell: true });
  if (run.status !== 0) {
    console.error(`\nShort ${n} failed (exit ${run.status}). Rendered before it: ${done.join(', ') || 'none'}.`);
    process.exit(run.status ?? 1);
  }
  const file = join(root, out);
  if (!existsSync(file) || statSync(file).mtimeMs < started - 1000) {
    console.error(`\nShort ${n}: the render reported success but ${out} was not written by this run.`);
    process.exit(1);
  }
  done.push(out);
}
console.log(`\nRendered ${done.length} Short${done.length === 1 ? '' : 's'}: ${done.join(', ')}`);
