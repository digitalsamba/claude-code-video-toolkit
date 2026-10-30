import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdtemp, readFile, rm, stat, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {chromium} from 'playwright';
import {installMasks, installRecorder, parseFlow, replayFlow} from '../scripts/flow.js';

test('flow boundary rejects invalid actions and dimensions', () => {
  const base = {version: 1, url: 'http://127.0.0.1', viewport: {width: 640, height: 360}, steps: []};
  assert.throws(() => parseFlow({...base, steps: [{kind: 'shell', command: 'echo no'}]}), /Unknown step/);
  assert.throws(() => parseFlow({...base, url: 'file:///private'}), /http/);
  assert.throws(() => parseFlow({...base, viewport: {width: -1, height: 360}}), /positive/);
  assert.throws(() => parseFlow({...base, steps: [{kind: 'fill', selector: '#name', value: 7}]}), /fill.value/);
});

test('record actions, keep secrets out of JSON, replay and record actual video', {timeout: 45000}, async t => {
  const server = createServer((_request, response) => {
    response.writeHead(200, {'Content-Type': 'text/html'});
    response.end(`<!doctype html><html><head></head><body>
      <label>Name <input id="name"></label><label>Password <input id="password" type="password"></label>
      <select id="tier"><option value="free">Free</option><option value="pro">Pro</option></select>
      <button data-testid="save" onclick="document.querySelector('#result').textContent=document.querySelector('#name').value+' '+document.querySelector('#tier').value">Save</button>
      <p id="result"></p></body></html>`);
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise<void>((resolve, reject) => {
    server.closeAllConnections();
    server.close(error => error ? reject(error) : resolve());
  }));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('No fixture server');
  const directory = await mkdtemp(path.join(tmpdir(), 'video-flow-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  const browser = await chromium.launch({headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
  t.after(() => browser.close());
  try {
  const flow = parseFlow({version: 1, url: `http://127.0.0.1:${address.port}/`,
    viewport: {width: 640, height: 360}, masks: ['#password'], steps: []});
  const recorder = await browser.newContext({viewport: flow.viewport});
  await installRecorder(recorder, flow);
  const page = await recorder.newPage();
  page.on('pageerror', error => process.stderr.write(`Recorder page error: ${error.message}\n`));
  await page.goto(flow.url);
  await page.locator('#name').fill('Alex');
  await page.locator('#password').fill('never-persist-this');
  await page.locator('#tier').selectOption('pro');
  await page.getByTestId('save').click();
  await page.evaluate(() => window.__toolkitFlush());
  assert.equal(await page.locator('#result').textContent(), 'Alex pro');
  assert.ok(flow.steps.some(step => step.kind === 'fill' && typeof step.value !== 'string'), JSON.stringify(flow.steps));
  assert.ok(!JSON.stringify(flow).includes('never-persist-this'));
  await recorder.close();
  const replay = await browser.newContext({viewport: flow.viewport, recordVideo: {dir: directory, size: flow.viewport}});
  await installMasks(replay, flow.masks);
  const target = await replay.newPage();
  await assert.rejects(replayFlow(target, flow, {variables: {}}), /Missing environment/);
  assert.equal(target.url(), 'about:blank');
  const events = await replayFlow(target, flow, {variables: {FLOW_SECRET_1: 'test-value'}});
  assert.equal(await target.locator('#result').textContent(), 'Alex pro');
  assert.equal(await target.locator('#password').inputValue(), 'test-value');
  assert.equal(await target.locator('#password').evaluate(element => getComputedStyle(element).filter), 'blur(12px)');
  assert.ok(events.find(event => event.kind === 'click')?.target);
  assert.ok(events.every(event => event.frame === Math.round(event.timestampMs * flow.fps / 1000)));
  await replay.close();
  const video = target.video();
  assert.ok(video);
  await video.saveAs(path.join(directory, 'verified.webm'));
  assert.ok((await stat(path.join(directory, 'verified.webm'))).size > 1000);

  const file = path.join(directory, 'flow.json');
  await writeFile(file, JSON.stringify(flow));
  const executableOption = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
    ? ['--executable-path', process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH] : [];
  const child = spawn(process.execPath, ['--import', 'tsx', 'scripts/flow.ts', 'replay', '--flow', file,
    '--output', path.join(directory, 'cli'), ...executableOption], {
    env: {...process.env, FLOW_SECRET_1: 'test-value'}, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', data => {stdout += String(data);});
  child.stderr.on('data', data => {stderr += String(data);});
  const exit = await new Promise<number | null>((resolve, reject) => {
    child.on('error', reject); child.on('close', resolve);
  });
  assert.equal(exit, 0, stderr);
  const result: unknown = JSON.parse(stdout);
  assert.ok(result);
  const eventFile: unknown = JSON.parse(await readFile(path.join(directory, 'cli.events.json'), 'utf8'));
  assert.ok(eventFile);
  assert.ok((await stat(path.join(directory, 'cli.webm'))).size > 1000);
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    throw error;
  }
});
