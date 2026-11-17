/**
 * Print the episode's timeline as JSON: every segment with the frame it starts on, and where each chapter
 * begins. It is the timeline the render plays (src/howto/timeline.ts), not a second copy of its arithmetic, so
 * a script that needs a time (scripts/chapters.py) cannot drift from the picture.
 *
 *     node scripts/timeline.mjs            # from templates/howto or a project made from it
 *
 * Needs the project's node_modules (esbuild comes with Remotion).
 */
import { build } from 'esbuild';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const entry = `
import { EPISODE } from './src/episode';
import { timeline } from './src/howto/timeline';
export const read = () => {
  let at = 0;
  const segments = timeline(EPISODE).map((s) => {
    const seg = { kind: s.kind, start: at, frames: s.frames, ...(s.kind === 'card' ? { chapter: s.chapter } : {}), ...(s.kind === 'scene' ? { id: s.spec.id, chapter: s.hook ? undefined : s.spec.chapter } : {}) };
    at += s.frames;
    return seg;
  });
  return {
    fps: 30,
    frames: at,
    series: EPISODE.series,
    number: EPISODE.number,
    title: EPISODE.title,
    hook: EPISODE.hook ? { ask: EPISODE.hook.ask } : undefined,
    chapters: EPISODE.chapters.map((c, i) => ({ title: c.title, ask: c.ask, start: segments.find((s) => s.kind === 'card' && s.chapter === i)?.start })),
    segments,
  };
};
`;
const dir = mkdtempSync(join(tmpdir(), 'howto-timeline-'));
try {
  const out = join(dir, 'timeline.mjs');
  const result = await build({ stdin: { contents: entry, resolveDir: root, loader: 'ts' }, bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
  writeFileSync(out, result.outputFiles[0].text);
  const { read } = await import(pathToFileURL(out).href);
  process.stdout.write(JSON.stringify(read(), null, 1) + '\n');
} catch (e) {
  console.error(`timeline.mjs: ${e.message ?? e}`);
  process.exitCode = 1;
} finally {
  rmSync(dir, { recursive: true, force: true });
}
