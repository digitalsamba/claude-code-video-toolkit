/**
 * One session, every line kind: setting up Modal in an agent harness. The frames here are fixed numbers; in a
 * narrated video they come from `cue()` so the session follows the voice.
 */
import React from 'react';
import { AbsoluteFill, staticFile } from 'remotion';
import brand from '../../../brands/video-toolkit/brand.json';
import {
  loadMonoFont,
  submittedAt,
  terminalTheme,
  TerminalSession,
  type SessionFocus,
  type SessionHighlight,
  type SessionLine,
  type SessionSkin,
  type SessionTheme,
} from '../../../lib/components';

loadMonoFont('JetBrains Mono', staticFile('fonts/JetBrainsMono-Regular.woff2'), staticFile('fonts/JetBrainsMono-Bold.woff2'));

const base = terminalTheme(brand);
export const THEME: SessionTheme = { ...base, font: `'JetBrains Mono', ${base.font}` };

const ask = { at: 12, user: 'set up modal for me, I want the free tier' };
const SENT = submittedAt(ask);

// the beats, in frames from the start
const B = {
  think: SENT + 4,
  say1: SENT + 40,
  check: SENT + 80,
  todo: SENT + 120,
  permit: SENT + 170,
  yes: SENT + 230,
  login: SENT + 236,
  loginDone: SENT + 300,
  pick: SENT + 330,
  picked: SENT + 400,
  deploy: SENT + 410,
  deployed: SENT + 500,
  env: SENT + 520,
  wrap: SENT + 600,
  end: SENT + 760,
};
export const TOTAL = B.end;

export const LINES: SessionLine[] = [
  ask,
  { at: B.think, spinner: 'Checking', until: B.say1 },
  { at: B.say1, say: "Modal's free Starter plan gives you **$30 of compute a month** once a card is on the account, and the toolkit's apps scale to zero when idle. First, what's already here:" },
  { at: B.check, tool: 'Bash', arg: 'uv run modal --version', result: ['modal client version 1.2.1'], resultAt: B.check + 20 },
  {
    at: B.todo,
    todo: [
      { text: 'Log in to Modal', done: B.loginDone + 4 },
      { text: 'Choose which apps to deploy', done: B.picked + 4 },
      { text: 'Deploy and write endpoint URLs to .env', done: B.deployed + 4 },
    ],
  },
  { at: B.permit, permission: 'uv run modal setup', description: 'Log in to Modal in the browser', select: [{ at: B.permit + 30, index: 0 }], answer: B.yes },
  { at: B.login, tool: 'Bash', arg: 'uv run modal setup', result: ['Opened modal.com in your browser', '{ok|✓} Token written to `~/.modal.toml`'], resultAt: B.loginDone, id: 'token' },
  {
    at: B.pick,
    ask: 'Which apps should I deploy?',
    header: 'Apps',
    options: [
      { label: 'All of them', note: 'each scales to zero, so idle apps cost nothing' },
      { label: 'Voice and music only', note: 'qwen3_tts, music_gen' },
      { label: 'Let me choose', note: 'one by one' },
    ],
    select: [{ at: B.pick + 25, index: 1 }, { at: B.pick + 45, index: 0 }],
    answer: B.picked,
  },
  { at: B.deploy, spinner: 'Deploying', until: B.deployed },
  { at: B.deploy + 4, tool: 'Bash', arg: 'uv run tools/modal_setup.py --all', result: ['{ok|✓} 7 apps deployed · URLs read from the Modal API'], more: 14, resultAt: B.deployed, id: 'deploy' },
  {
    at: B.env,
    diff: '.env',
    rows: [
      { n: 11, sign: ' ', text: '# Cloud GPU' },
      { n: 12, sign: '+', text: 'MODAL_QWEN3_TTS_ENDPOINT_URL=https://…modal.run' },
      { n: 13, sign: '+', text: 'MODAL_MUSIC_GEN_ENDPOINT_URL=https://…modal.run' },
      { n: 14, sign: '+', text: 'MODAL_FLUX2_ENDPOINT_URL=https://…modal.run' },
    ],
    id: 'env',
  },
  { at: B.wrap, say: 'Done. Try `/video` to start a project, or `uv run tools/modal_setup.py --list` to see what is deployed. A voice line costs about **$0.01**.' },
];

export const FOCUS: SessionFocus[] = [
  { at: B.loginDone + 6, id: 'token', zoom: 1.5, x: 0.35 },
  { at: B.loginDone + 24, zoom: 1 },
  { at: B.env + 8, id: 'env', zoom: 1.45, x: 0.3 },
  { at: B.env + 70, zoom: 1 },
];

export const HIGHLIGHT: SessionHighlight[] = [{ id: 'deploy', from: B.deployed + 2, to: B.env }];

export const STATUS = ['{accent|Opus 5.5} {dim|·} {code|my-video} {dim|·} main', '{dim|▸▸ auto mode on (shift+tab to cycle)}'];

export const SessionScene: React.FC<{ skin: 'claude-code' | 'codex' | 'shell' | SessionSkin; vertical?: boolean }> = ({ skin, vertical }) => (
  <AbsoluteFill style={{ background: brand.colors.bgAlt, alignItems: 'center', justifyContent: 'center' }}>
    <TerminalSession
      skin={skin}
      theme={THEME}
      lines={LINES}
      status={skin === 'shell' ? [] : STATUS}
      title="~/work/my-video"
      width={vertical ? 1000 : 1680}
      height={vertical ? 1700 : 940}
      fontSize={vertical ? 34 : 28}
      focus={FOCUS}
      highlight={HIGHLIGHT}
    />
  </AbsoluteFill>
);
