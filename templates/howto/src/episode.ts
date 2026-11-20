/**
 * Getting started 1: What it costs. The one file a new episode replaces (with its narration in public/audio).
 *
 * Every frame is a word: `cue('three hundred')` is the frame Marilyn says it. Every figure on screen comes from
 * facts.json, which says where each one came from and when it was checked.
 */
import facts from '../facts.json';
import type { SessionLine } from '../../../lib/components';
import type { Episode } from './howto/types';

const money = (n: number) => `$${n.toFixed(2)}`;
const F = facts;

export const EPISODE: Episode = {
  series: 'Getting started',
  number: 1,
  version: F.toolkitVersion.value,
  title: ['What it costs:', 'mostly nothing'],
  next: ['Next: the whole setup,', 'in one conversation'],
  link: 'github.com/digitalsamba/claude-code-video-toolkit',
  music: { file: 'audio/music/bed_2.mp3', volume: 0.35 }, // under narration at -16 LUFS
  hook: {
    ask: 'will this cost me money?',
    scene: {
      id: 'hook',
      kind: 'ledger',
      chapter: 0,
      props: (cue) => ({
        title: { text: 'Two weeks of making videos', at: cue('first two weeks') },
        rows: [
          { label: 'Video projects', detail: '1 to 15 November', value: F.bill.projectsInNovember, at: cue('fifteen') },
          { label: 'Cloud bill', detail: 'every AI model we ran', value: F.bill.months.Nov, format: money, at: cue('cloud bill'), valueAt: cue('twenty two') },
          { label: 'What we paid', detail: 'the rest of this video is why', detailAt: cue('none', { edge: 'end' }), value: 0, format: () => '$0', at: cue('we paid'), valueAt: cue('none') },
        ],
        note: `modal billing report, every app in our workspace, ${F.bill.checked}`,
      }),
    },
  },
  chapters: [
    { title: ["What's free"], field: 'explain', ask: "what's free?" },
    { title: ['Thirty dollars', 'a month'], field: 'explain', ask: 'what do the AI models cost?' },
    { title: ['What we spend'], field: 'explain', ask: 'what do you really spend?' },
    { title: ['Setting it up'], field: 'work', ask: 'ok, how do I set it up?' },
  ],
  scenes: [
    {
      id: 'free',
      kind: 'ledger',
      chapter: 0,
      props: (cue) => ({
        title: { text: 'Making the video', at: cue('making'), then: { text: 'costs nothing', at: cue('costs nothing') } },
        rows: [
          { label: 'Rendering', detail: 'Remotion, on your own machine', detailAt: cue('own machine'), value: '$0', at: cue('remotion') },
          { label: 'Captions', detail: 'Whisper, on your machine', value: '$0', at: cue('captions') },
          { label: 'Music', detail: 'ACE-Step API, free key', detailAt: cue('free api key'), value: '$0', at: cue('music') },
          { label: 'Checks and Shorts', detail: 'qc.py, cutdown.py', detailAt: cue('shorts'), value: '$0', at: cue('every check') },
          { label: 'Claude Code', detail: 'runs on your Claude plan', detailAt: cue('run claude code'), value: 'your plan', at: cue('one thing') , valueAt: cue('claude plan') },
        ],
        allAt: cue('all free'),
      }),
    },
    {
      id: 'thirty',
      kind: 'figure',
      chapter: 1,
      props: (cue) => ({
        title: { text: 'The AI models run on cloud GPUs,', at: cue('cloud gpus'), then: { text: 'the part you may worry about', at: cue('the part'), newLine: true } },
        prefix: '$',
        value: F.modalStarterCredit.value,
        at: cue('thirty dollars'),
        lines: [
          { text: ['of compute every month, free', "on Modal's Starter plan"], at: cue('of compute') },
          { text: ['once a card is on file'], at: cue('once a card'), add: true },
          { text: ['Idle apps scale to zero'], at: cue('scale to zero') },
          { text: ['You pay for the seconds a GPU works'], at: cue('only pay'), add: true },
        ],
        note: `modal.com/pricing, checked ${F.modalStarterCredit.checked}. A card must be on file.`,
      }),
      short: (cue) => ({
        title: { text: 'Models run on cloud GPUs,', at: cue('cloud gpus'), then: { text: 'the part you may worry about', at: cue('the part'), newLine: true } },
        lines: [
          { text: ['of compute every', 'month, free'], at: cue('of compute') },
          { text: ['once a card', 'is on file'], at: cue('once a card'), add: true },
          { text: ['Idle apps', 'scale to zero'], at: cue('scale to zero') },
          { text: ['You pay for the', 'seconds a GPU works'], at: cue('only pay'), add: true },
        ],
        note: `modal.com/pricing, ${F.modalStarterCredit.checked}`,
      }),
    },
    {
      id: 'buys',
      kind: 'ledger',
      chapter: 1,
      props: (cue) => ({
        hero: { text: '$30' },
        title: { text: 'What it buys', at: cue('what does') },
        rows: [
          { label: 'Video clips', show: { video: 'buys/clip.mp4' }, detail: `LTX-2.3, about ${money(F.prices.ltxClip)} a clip`, value: Math.floor(30 / F.prices.ltxClip / 50) * 50, at: cue('two hundred') },
          { label: 'Voice lines', show: { wave: true }, detail: `Qwen3-TTS, about ${money(F.prices.voiceLine)} a line`, value: Math.floor(30 / F.prices.voiceLine / 1000) * 1000, at: cue('three thousand') },
          { label: 'Images', show: { images: [1, 2, 3, 4].map((i) => `buys/image-${i}.jpg`) }, detail: `FLUX.2, about a cent an image`, value: Math.floor(30 / F.prices.fluxImage / 500) * 500, at: cue('two and a half') },
        ],
        note: `Unit costs from our Modal bill for a measured batch, ${F.prices.checked}`,
      }),
    },
    {
      id: 'spent',
      kind: 'bars',
      chapter: 2,
      props: (cue) => ({
        title: { text: ['Our Modal bill, everything included'], at: cue('our own bill') },
        bars: [
          { label: 'Aug', value: F.bill.months.Aug, at: cue('august'), valueAt: cue('eleven') },
          { label: 'Sep', value: F.bill.months.Sep, at: cue('september'), sub: { text: 'busiest', at: cue('busiest') }, valueAt: cue('twenty six') },
          { label: 'Oct', value: F.bill.months.Oct, at: cue('october'), valueAt: cue('fourteen') },
          {
            label: '1 to 15 Nov',
            value: F.bill.months.Nov,
            at: cue('first half of november'),
            valueAt: cue('twenty two'),
            sub: { text: `${F.bill.projectsInNovember} projects`, at: cue('fifteen') },
          },
        ],
        limit: { value: F.modalStarterCredit.value, label: '$30 free credit', at: cue('everything') },
        format: money,
        after: { text: 'Every month inside the free credit', at: cue('every month') },
        note: `modal billing report, every app in our workspace, ${F.bill.checked}`,
      }),
      short: (cue) => ({
        title: { text: ['Our Modal bill,', 'everything included'], at: cue('our own bill') },
        after: { text: 'Inside the free credit', at: cue('every month') },
        note: `modal billing report, ${F.bill.checked}`,
      }),
    },
    {
      id: 'extras',
      kind: 'ledger',
      chapter: 2,
      props: (cue) => ({
        title: { text: 'Two things cost extra,', at: cue('two things'), then: { text: 'both optional', at: cue('both are optional') } },
        rows: [
          { label: 'ElevenLabs voices', detail: "like the one you're hearing", detailAt: cue("like the one"), value: F.prices.elevenLabs, at: cue('elevenlabs') },
          { label: 'Ideogram images', detail: F.prices.ideogram, detailAt: cue('few cents'), value: 'per image', at: cue('ideogram') },
        ],
        after: { text: 'Without them:', at: cue('without them'), then: { text: 'free models, inside the credit', at: cue('free voices') } },
      }),
      short: (cue) => ({
        title: { text: 'Two things cost extra,', at: cue('two things'), then: { text: 'both optional', at: cue('both are optional'), newLine: true } },
        after: { text: 'Without them:', at: cue('without them'), then: { text: 'free models, inside the credit', at: cue('free voices'), newLine: true } },
      }),
    },
    {
      id: 'setup',
      kind: 'session',
      chapter: 3,
      props: (cue) => {
        const lines: SessionLine[] = [
          { at: 16, user: 'set up modal for me, I want the free tier', perChar: 1, submit: cue('one sentence', { edge: 'end' }) },
          { at: cue('checks'), tool: 'Bash', arg: 'uv run modal --version', result: ['modal client version 1.2.1'], resultAt: cue('installed') },
          { at: cue('logs you in'), permission: 'uv run modal setup', description: 'Log in to Modal in the browser', answer: cue('say yes') },
          { at: cue('say yes', { offset: 4 }), tool: 'Bash', arg: 'uv run modal setup', result: ['Opened modal.com in your browser', '{ok|✓} Token written to `~/.modal.toml`'], resultAt: cue('say yes', { offset: 30 }) },
          { at: cue('deploys'), spinner: 'Deploying', until: cue('the models') },
          { at: cue('the models'), tool: 'Bash', arg: 'uv run tools/modal_setup.py --all', result: ['{ok|✓} 8 apps deployed, endpoint URLs written to `.env`', '{dim|ltx2 (video clips) is one more command: it needs a Hugging Face token}'], resultAt: cue('ready'), id: 'deployed' },
          { at: cue('cost nothing'), say: 'Done. Every app **scales to zero**, so nothing is billed until you use it.', wps: 7 },
        ];
        return {
          lines,
          status: ['{accent|Opus} {dim|·} {code|claude-code-video-toolkit} {dim|·} main'],
          highlight: [{ id: 'deployed', from: cue('ready') + 4 }],
          focus: [
            { at: cue('add a card'), zoom: 1.35, x: 0.32, y: 0.78, over: 24 },
            { at: cue('say yes', { edge: 'end' }), zoom: 1, over: 20 },
          ],
        };
      },
      short: () => ({ focus: [] }),
    },
    {
      id: 'start',
      kind: 'command',
      chapter: 3,
      props: (cue) => ({
        command: '/setup',
        at: cue('slash setup'),
        lines: [{ text: "Says what's free before it spends a cent", at: cue('tells you') }],
        note: 'In Claude Code, from the toolkit folder',
      }),
    },
  ],
};
