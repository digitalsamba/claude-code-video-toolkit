/** Record browser actions once and replay them with target/frame events. */
import {chromium} from 'playwright';
import type {BrowserContext, Page} from 'playwright';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {createInterface} from 'node:readline';

declare global {
  interface Window {
    __toolkitRecordStep: (step: unknown) => Promise<void>;
    __toolkitSecretName: (selector: string) => Promise<string>;
    __toolkitFlush: () => Promise<void>;
  }
}

export type FillValue = string | {env: string};
export type FlowStep =
  | {kind: 'click'; selector: string}
  | {kind: 'fill'; selector: string; value: FillValue}
  | {kind: 'select'; selector: string; value: string}
  | {kind: 'press'; selector: string; key: string}
  | {kind: 'navigate'; url: string}
  | {kind: 'waitFor'; selector: string}
  | {kind: 'waitForURL'; url: string}
  | {kind: 'wait'; milliseconds: number};
export type BrowserFlow = {
  version: 1;
  url: string;
  viewport: {width: number; height: number};
  fps: number;
  masks: string[];
  steps: FlowStep[];
};
export type FlowEvent = {
  step: number;
  kind: FlowStep['kind'];
  timestampMs: number;
  frame: number;
  selector?: string;
  target?: {x: number; y: number; width: number; height: number};
};

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function text(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${name} must be a non-empty string`);
  return value;
}
function positive(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
  return value;
}
function httpURL(value: unknown): string {
  const url = new URL(text(value, 'url'));
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only http/https URLs are supported');
  return url.href;
}
export function parseStep(value: unknown): FlowStep {
  if (!object(value)) throw new Error('Each step must be an object');
  switch (value.kind) {
    case 'click': case 'waitFor':
      return {kind: value.kind, selector: text(value.selector, 'selector')};
    case 'fill': {
      let fill: FillValue;
      if (typeof value.value === 'string') fill = value.value;
      else if (object(value.value)) fill = {env: text(value.value.env, 'value.env')};
      else throw new Error('fill.value must be a string or {env: "NAME"}');
      return {kind: 'fill', selector: text(value.selector, 'selector'), value: fill};
    }
    case 'select': return {kind: 'select', selector: text(value.selector, 'selector'), value: text(value.value, 'value')};
    case 'press': return {kind: 'press', selector: text(value.selector, 'selector'), key: text(value.key, 'key')};
    case 'navigate': case 'waitForURL': return {kind: value.kind, url: httpURL(value.url)};
    case 'wait': {
      const milliseconds = positive(value.milliseconds, 'milliseconds');
      if (milliseconds > 600000) throw new Error('A wait cannot exceed ten minutes');
      return {kind: 'wait', milliseconds};
    }
    default: throw new Error(`Unknown step kind: ${String(value.kind)}`);
  }
}
export function parseFlow(value: unknown): BrowserFlow {
  if (!object(value) || value.version !== 1) throw new Error('Flow version must be 1');
  if (!object(value.viewport)) throw new Error('viewport is required');
  const width = positive(value.viewport.width, 'viewport.width');
  const height = positive(value.viewport.height, 'viewport.height');
  const fps = positive(value.fps ?? 30, 'fps');
  if (!Number.isInteger(width) || !Number.isInteger(height) || width > 7680 || height > 7680) {
    throw new Error('viewport dimensions must be integers up to 7680');
  }
  if (!Number.isInteger(fps) || fps > 120) throw new Error('fps must be an integer up to 120');
  if (!Array.isArray(value.steps)) throw new Error('steps must be an array');
  const masks = value.masks ?? [];
  if (!Array.isArray(masks)) throw new Error('masks must be an array');
  return {version: 1, url: httpURL(value.url), viewport: {width, height}, fps,
    masks: masks.map((item: unknown) => text(item, 'mask')), steps: value.steps.map(parseStep)};
}

export async function installMasks(context: BrowserContext, masks: string[]): Promise<void> {
  if (masks.length === 0) return;
  await context.addInitScript((selectors: string[]) => {
    document.addEventListener('DOMContentLoaded', () => {
      const style = document.createElement('style');
      style.textContent = `${selectors.join(',')} {filter: blur(12px) !important;}`;
      document.head.appendChild(style);
    });
  }, masks);
}

export async function installRecorder(context: BrowserContext, flow: BrowserFlow): Promise<void> {
  const secrets = new Map<string, string>();
  await context.exposeBinding('__toolkitRecordStep', ({page, frame}, value: unknown) => {
    if (frame !== page.mainFrame()) return;
    const step = parseStep(value);
    const last = flow.steps.at(-1);
    if (step.kind === 'fill' && last?.kind === 'fill' && last.selector === step.selector) {
      flow.steps[flow.steps.length - 1] = step;
    } else flow.steps.push(step);
  });
  await context.exposeBinding('__toolkitSecretName', ({page, frame}, selector: string) => {
    if (frame !== page.mainFrame()) return null;
    if (!secrets.has(selector)) secrets.set(selector, `FLOW_SECRET_${secrets.size + 1}`);
    return secrets.get(selector);
  });
  const recorderSource = await readFile(new URL('./flow-recorder.browser.js', import.meta.url), 'utf8');
  await context.addInitScript({content: `(() => {const masks = ${JSON.stringify(flow.masks)};\n${recorderSource}\n})();`});
  context.on('page', page => {
    let initial = true;
    page.on('framenavigated', frame => {
      if (frame !== page.mainFrame() || !frame.url().startsWith('http')) return;
      if (initial) {initial = false; return;}
      flow.steps.push({kind: 'waitForURL', url: frame.url()});
    });
  });
}

export async function replayFlow(page: Page, flow: BrowserFlow,
  options: {variables?: NodeJS.ProcessEnv; timeoutMs?: number; onEvent?: (event: FlowEvent) => void} = {}): Promise<FlowEvent[]> {
  const variables = options.variables ?? process.env;
  // Resolve secrets before navigation so a missing variable cannot fail halfway through a demo.
  for (const step of flow.steps) {
    if (step.kind === 'fill' && typeof step.value !== 'string' && variables[step.value.env] === undefined) {
      throw new Error(`Missing environment variable ${step.value.env}`);
    }
  }
  page.setDefaultTimeout(options.timeoutMs ?? 10000);
  const events: FlowEvent[] = [];
  const start = performance.now();
  await page.goto(flow.url, {waitUntil: 'domcontentloaded'});
  for (const [index, step] of flow.steps.entries()) {
    const elapsed = performance.now() - start;
    const event: FlowEvent = {step: index, kind: step.kind, timestampMs: elapsed,
      frame: Math.round(elapsed * flow.fps / 1000)};
    if ('selector' in step) {
      event.selector = step.selector;
      if (step.kind !== 'waitFor') {
        await page.locator(step.selector).waitFor({state: 'visible'});
        await page.locator(step.selector).scrollIntoViewIfNeeded();
        const box = await page.locator(step.selector).boundingBox();
        if (box) event.target = box;
      }
    }
    event.timestampMs = performance.now() - start;
    event.frame = Math.round(event.timestampMs * flow.fps / 1000);
    switch (step.kind) {
      case 'click': await page.locator(step.selector).click(); break;
      case 'fill': {
        const value = typeof step.value === 'string' ? step.value : variables[step.value.env];
        if (value === undefined) throw new Error('Missing fill value');
        await page.locator(step.selector).fill(value); break;
      }
      case 'select': await page.locator(step.selector).selectOption(step.value); break;
      case 'press': await page.locator(step.selector).press(step.key); break;
      case 'navigate': await page.goto(step.url, {waitUntil: 'domcontentloaded'}); break;
      case 'waitFor': await page.locator(step.selector).waitFor({state: 'visible'}); break;
      case 'waitForURL': await page.waitForURL(step.url); break;
      case 'wait': await page.waitForTimeout(step.milliseconds); break;
      default: {const unreachable: never = step; throw new Error(`Unhandled step ${String(unreachable)}`);}
    }
    events.push(event);
    options.onEvent?.(event);
  }
  return events;
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  const {values, positionals} = parseArgs({args: argv, allowPositionals: true, options: {
    flow: {type: 'string'}, url: {type: 'string'}, output: {type: 'string', default: 'output/demo'},
    duration: {type: 'string', default: '120'}, mask: {type: 'string', multiple: true},
    'storage-state': {type: 'string'}, 'executable-path': {type: 'string'},
    headed: {type: 'boolean', default: false}, help: {type: 'boolean', default: false},
  }});
  if (values.help) {
    process.stdout.write('flow.ts record --url URL --flow flow.json [--mask SELECTOR]\nflow.ts replay --flow flow.json [--output output/demo] [--headed]\n');
    return;
  }
  const command = positionals[0];
  if (!['record', 'replay'].includes(command) || !values.flow) throw new Error('Use record or replay with --flow FILE. See --help.');
  const flow = command === 'replay'
    ? parseFlow(JSON.parse(await readFile(values.flow, 'utf8')))
    : parseFlow({version: 1, url: values.url, viewport: {width: 1920, height: 1080}, masks: values.mask ?? [], steps: []});
  const duration = Number(values.duration);
  if (!Number.isFinite(duration) || duration <= 0 || duration > 3600) throw new Error('--duration must be between 0 and 3600 seconds');
  const output = path.resolve(values.output);
  await mkdir(path.dirname(output), {recursive: true});
  const browser = await chromium.launch({headless: command !== 'record' && !values.headed,
    executablePath: values['executable-path']});
  let context: BrowserContext | undefined;
  try {
    context = await browser.newContext({viewport: flow.viewport, storageState: values['storage-state'],
      recordVideo: {dir: path.join(path.dirname(output), '.recordings'), size: flow.viewport}});
    await installMasks(context, flow.masks);
    if (command === 'record') await installRecorder(context, flow);
    const page = await context.newPage();
    let events: FlowEvent[] = [];
    if (command === 'record') {
      await page.goto(flow.url, {waitUntil: 'domcontentloaded'});
      process.stderr.write('Recording actions. Press Enter to save, or wait for --duration.\n');
      await new Promise<void>(resolve => {
        const terminal = createInterface({input: process.stdin, output: process.stderr});
        const timer = setTimeout(() => {terminal.close(); resolve();}, duration * 1000);
        terminal.once('line', () => {clearTimeout(timer); terminal.close(); resolve();});
      });
      await page.evaluate(() => window.__toolkitFlush());
      await mkdir(path.dirname(path.resolve(values.flow)), {recursive: true});
      await writeFile(values.flow, JSON.stringify(flow, null, 2) + '\n');
    } else events = await replayFlow(page, flow);
    await context.close();
    await page.video()?.saveAs(`${output}.webm`);
    await page.video()?.delete();
    await writeFile(`${output}.events.json`, JSON.stringify({version: 1, fps: flow.fps, viewport: flow.viewport, events}, null, 2) + '\n');
    process.stdout.write(JSON.stringify({flow: values.flow, video: `${output}.webm`, events: `${output}.events.json`, steps: flow.steps.length}) + '\n');
  } finally {
    await context?.close();
    await browser.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
