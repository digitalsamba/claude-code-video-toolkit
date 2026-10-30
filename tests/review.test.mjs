import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(new URL('../playwright/package.json', import.meta.url));
const {chromium} = require(process.env.VIDEO_TOOLKIT_PLAYWRIGHT || 'playwright');

test('review page persists notes and approvals, invalidates edits and fits mobile', {timeout: 45000}, async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'video-review-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  await mkdir(path.join(directory, 'public'));
  const project = {name: 'Release review', template: 'custom', scenes: [
    {id: 'intro', type: 'title', durationSeconds: 5, narration: 'Hello viewers'},
    {id: 'demo', type: 'demo', durationSeconds: 8, narration: 'See the product'}]};
  const manifest = path.join(directory, 'project.json');
  await writeFile(manifest, JSON.stringify(project));
  const server = spawn(process.env.VIDEO_TOOLKIT_PYTHON || 'python', [path.join(root, 'tools/review.py'), directory], {stdio: ['ignore', 'pipe', 'pipe']});
  t.after(() => server.kill());
  const url = await new Promise((resolve, reject) => {
    let text = '';
    server.stdout.on('data', data => {text += String(data); const match = text.match(/http:\/\/127.0.0.1:\d+/); if (match) resolve(match[0]);});
    server.on('error', reject);
    server.on('exit', code => reject(new Error(`Review server exited ${code}`)));
  });
  const browser = await chromium.launch({headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
  t.after(() => browser.close());
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url);
  await page.getByRole('heading', {name: 'Release review'}).waitFor();
  await page.getByLabel('Add a note').fill('<script>Keep this as text</script>');
  await page.getByRole('button', {name: 'Save note'}).click();
  await page.getByText('<script>Keep this as text</script>', {exact: true}).waitFor();
  await page.getByRole('button', {name: 'Approve scene', exact: true}).click();
  await page.getByRole('button', {name: 'Scene approved', exact: true}).waitFor();
  await page.reload();
  await page.getByRole('button', {name: 'Scene approved', exact: true}).waitFor();
  project.scenes[0].narration = 'Edited narration';
  await writeFile(manifest, JSON.stringify(project));
  await page.getByRole('button', {name: 'Refresh project'}).click();
  await page.getByText('Changed since approval', {exact: false}).waitFor();
  await mkdir(path.join(root, 'scratch'), {recursive: true});
  await page.screenshot({path: path.join(root, 'scratch/review-desktop.png'), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: path.join(root, 'scratch/review-mobile.png'), fullPage: true});
  assert.deepEqual(errors, []);
});
