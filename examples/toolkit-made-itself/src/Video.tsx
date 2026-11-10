/**
 * Made with itself: a video about claude-code-video-toolkit that shows its own making (BRIEF.md, design.md).
 *
 *   open      identity  01  the sentence that started it, typed into Claude Code
 *   narrate   record    02  voiceover.py's real run, then the scene script lit word by word
 *   score     record    03  music_gen + pick_music's real run, then the four takes, the pick playing
 *   generate  record    04  the three FLUX tries and the LTX clip, then the measured bill
 *   compose   record    05  this scene's own source beside the terminal it drives
 *   check     lab       06  qc.py on the draft render: what it found and the fix
 *   close     identity  07  one sentence in, this video out
 *
 * Every terminal line comes from capture/ (pick() throws if it isn't there); every number from a
 * measurement (bill.json from scripts/bill.py, timing.json from scripts/timing.py).
 */
import React from 'react';
import { AbsoluteFill, Audio, Img, interpolate, OffthreadVideo, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { BarChart, Counter, PageTurns, Terminal, TerminalLine, typedUntil, voice, VoicePhrase } from '../../../lib/components';
import { COLORS, Field, FIELDS, FONT, MONO, TERM } from './fields';
import { CapturedLine, has, pick, pickExact, run } from './captured';
import { composeLines, composeReady } from './compose';
import bill from './bill.json';
import source from './source.json';
import timing from './timing.json';
import c01 from '../public/audio/scenes/01_open.captions.json';
import c02 from '../public/audio/scenes/02_narrate.captions.json';
import c03 from '../public/audio/scenes/03_score.captions.json';
import c04 from '../public/audio/scenes/04_generate.captions.json';
import c05 from '../public/audio/scenes/05_compose.captions.json';
import c06 from '../public/audio/scenes/06_check.captions.json';
import c07 from '../public/audio/scenes/07_close.captions.json';

const T = timing as { phrases: Record<string, number>; scenes: Record<string, number>; starts: Record<string, number>; total: number };
const ph = (file: string, captions: VoicePhrase['captions']) => ({ file, at: T.phrases[file], captions });
const V1 = voice([ph('01_open', c01)]);
const V2 = voice([ph('02_narrate', c02)]);
const V3 = voice([ph('03_score', c03)]);
const V4 = voice([ph('04_generate', c04)]);
const V5 = voice([ph('05_compose', c05)]);
const V6 = voice([ph('06_check', c06)]);
const V7 = voice([ph('07_close', c07)]);
export const TOTAL = T.total;

const PROMPT = 'Make a 90 second video about this toolkit, using this toolkit.';
const SITE = 'digitalsamba.github.io/claude-code-video-toolkit';
const PICK = 3;
const MUSIC_S = 100; // the takes' length (music_gen --duration) // the music take in use (bg_3, pick_music's recommendation; Conal's ear has the last word)

// ---------------------------------------------------------------- layout
const L = 120;
const rise = (frame: number, at: number, fps: number) => (frame < at ? 0 : Math.min(1, spring({ frame: frame - at, fps, config: { damping: 200 } })));

const Scene: React.FC<{ id: string; field: Field; lines: [string, number][]; children: React.ReactNode }> = ({ id, field, lines, children }) => {
  const len = T.scenes[id];
  const frame = useCurrentFrame();
  // the ground cuts hard between acts; only the content eases in and out
  const o = interpolate(frame, [0, 8, len - 8, len], [0, 1, 1, 0], { extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ background: field.bg, fontFamily: FONT }}>
      {lines.map(([file, at]) => (
        <Sequence key={file} from={at}><Audio src={staticFile(`audio/scenes/${file}.mp3`)} /></Sequence>
      ))}
      <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>
    </AbsoluteFill>
  );
};

const Sfx: React.FC<{ at: number; file: string; volume?: number; frames?: number }> = ({ at, file, volume = 0.3, frames }) => (
  <Sequence from={Math.max(0, Math.round(at))} durationInFrames={frames}><Audio src={staticFile(`audio/sfx/${file}.mp3`)} volume={volume} /></Sequence>
);

const Heading: React.FC<{ field: Field; title: string; sub: string }> = ({ field, title, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = rise(frame, 2, fps);
  return (
    <div style={{ position: 'absolute', left: L, top: 84, color: field.ink, opacity: t, transform: `translateY(${(1 - t) * 20}px)` }}>
      <div style={{ fontSize: 72, fontWeight: 700, letterSpacing: '-0.035em', lineHeight: 1 }}>{title}</div>
      <div style={{ fontSize: 36, marginTop: 14, color: field.muted }}>{sub}</div>
    </div>
  );
};

/** A long shell command as rows that type one after another, the way a terminal wraps it. */
const commandRows = (cmd: string, at: number, width = 96, perChar = 0.5): TerminalLine[] => {
  const rows: string[] = [];
  let row = '';
  for (const word of cmd.split(' ')) {
    if (row && row.length + 1 + word.length > width) { rows.push(row); row = word; } else row = row ? `${row} ${word}` : word;
  }
  rows.push(row);
  const out: TerminalLine[] = [];
  let t = at;
  rows.forEach((r, i) => {
    const line = { at: t, command: r, perChar, prompt: i === 0 ? '$' : ' ' };
    out.push(line);
    t = typedUntil(line) + 1;
  });
  return out;
};
const endOf = (lines: TerminalLine[]) => {
  const last = lines[lines.length - 1];
  return 'command' in last ? typedUntil(last) : last.at;
};
/** Typing sound under each typed command row. */
const typingSfx = (lines: TerminalLine[]) =>
  lines.filter((l): l is Extract<TerminalLine, { command: string }> => 'command' in l && l.command.length > 0).map((l) => (
    <Sfx key={`t${l.at}`} at={l.at} file="typing" volume={0.18} frames={Math.max(4, typedUntil(l) - l.at)} />
  ));

// ---------------------------------------------------------------- 1 open
const OpenScene: React.FC = () => {
  const f = FIELDS.identity;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const typing: TerminalLine = { at: 14, command: PROMPT, perChar: 1 };
  const tag = rise(frame, V1.cue('everything'), fps);
  return (
    <Scene id="open" field={f} lines={V1.lines}>
      {typingSfx([typing])}
      <Terminal
        title="claude-code-video-toolkit · Claude Code"
        theme={TERM}
        lines={[typing]}
        width={1680}
        height={210}
        fontSize={34}
        style={{ position: 'absolute', left: L, top: 300 }}
      />
      <div style={{ position: 'absolute', left: L, top: 620, color: f.ink, opacity: tag, transform: `translateY(${(1 - tag) * 20}px)` }}>
        <div style={{ fontSize: 112, fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1 }}>Type a prompt, get a video.</div>
        <div style={{ fontSize: 36, marginTop: 24, color: f.muted }}>This one was made by the toolkit it shows.</div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- 2 narrate
const SCENE_FILES = ['01_open', '02_narrate', '03_score', '05_compose', '07_close'];
const NarrateScene: React.FC = () => {
  const f = FIELDS.record;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const vo = run('voiceover');
  const cmd = commandRows(vo.command, 6);
  let t = endOf(cmd) + 8;
  const results: TerminalLine[] = [];
  for (const s of SCENE_FILES) {
    results.push({ at: t, ...pick('voiceover', `scenes/${s}.mp3`), kind: 'dim' });
    results.push({ at: t + 2, ...pick('voiceover', '"wpm"', `scenes/${s}.mp3`), kind: 'ok' });
    t += 9;
  }
  const turnAt = V2.cue("that's me");
  // the script page: the phrase's own words, lit as she says them
  const start = T.phrases['02_narrate'];
  const ms = ((frame - start) / fps) * 1000;
  const wpm = pick('voiceover', '"wpm"', 'scenes/02_narrate.mp3').text.match(/[0-9.]+/)?.[0];
  const secs = pick('voiceover', '"duration_seconds"', 'scenes/02_narrate.mp3').text.match(/[0-9.]+/)?.[0];
  const page = (
    <AbsoluteFill style={{ background: f.bg }}>
      <Heading field={f} title="Narrate" sub="One file a scene, voiced and paced in words a minute" />
      <div style={{ position: 'absolute', left: L, top: 290, width: 1680, background: '#ffffff', borderRadius: 12, padding: '44px 56px', boxSizing: 'border-box' }} data-qc="product">
        <div style={{ fontFamily: MONO, fontSize: 28, color: f.muted }}>public/audio/scenes/02_narrate.txt</div>
        <div style={{ marginTop: 28, fontSize: 56, lineHeight: 1.3, fontWeight: 500, letterSpacing: '-0.01em' }}>
          {c02.map((w, i) => (
            <span key={i} style={{ color: ms >= w.startMs ? f.accent : f.ink }}>{w.text.endsWith(' ') ? w.text : `${w.text} `}</span>
          ))}
        </div>
        <div style={{ marginTop: 32, fontFamily: MONO, fontSize: 28, color: f.muted }}>
          02_narrate.mp3 · {secs} s · {wpm} wpm · ElevenLabs, eleven_v4, voice Marilyn
        </div>
      </div>
    </AbsoluteFill>
  );
  return (
    <Scene id="narrate" field={f} lines={V2.lines}>
      {typingSfx(cmd)}
      {results.filter((l) => 'kind' in l && l.kind === 'ok').map((l) => <Sfx key={`p${l.at}`} at={l.at} file="pop" volume={0.16} />)}
      <Sfx at={turnAt - 6} file="turn" volume={0.35} />
      <PageTurns
        before={
          <AbsoluteFill style={{ background: f.bg }}>
            <Heading field={f} title="Narrate" sub="tools/voiceover.py, as it ran for this video" />
            <Terminal title="claude-code-video-toolkit" theme={TERM} lines={[...cmd, ...results]} width={1680} height={680} fontSize={24} style={{ position: 'absolute', left: L, top: 290 }} />
          </AbsoluteFill>
        }
        pages={[{ at: turnAt, content: page }]}
      />
    </Scene>
  );
};

// ---------------------------------------------------------------- 3 score
const ScoreScene: React.FC = () => {
  const f = FIELDS.record;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const gen = commandRows(run('music_gen').command, 4, 96, 0.25);
  let t = endOf(gen) + 4;
  const saved: TerminalLine[] = [1, 2, 3, 4].map((n) => ({ at: t + n * 5, ...pick('music_gen', `bg_${n}.mp3 (`) }));
  t += 26;
  const pk = commandRows(run('pick_music').command, t, 96, 0.25);
  t = endOf(pk) + 4;
  const table: TerminalLine[] = [
    { at: t, ...pick('pick_music', 'take') },
    ...(() => {
      const ls = run('pick_music').lines;
      const head = ls.findIndex((l) => l.text.startsWith('take'));
      return ls.slice(head + 1, head + 5).map((l, i) => ({ at: t + 3 + i * 3, ...l }));
    })(),
    { at: t + 18, ...pick('pick_music', 'Recommended'), kind: 'accent' as const },
  ];
  const turnAt = V3.cue('picked');
  const hearAt = V3.cue('hearing');
  const sitsAt = V3.cue('sits');
  const movesAt = V3.cue('moves');
  const cutAt = V3.cue('cut');
  // where the music is right now: it starts with the video, so its position is the video's time
  const now = (T.starts.score + frame) / fps;
  const cuts = Object.values(T.starts).slice(1).map((s) => s / fps);
  const W = 1500;
  const wave = (n: number, i: number) => {
    const on = n === PICK && frame >= sitsAt;
    const shown = rise(frame, turnAt + i * 3, fps);
    return (
      <div key={n} style={{ position: 'relative', height: 132, marginBottom: 22, opacity: shown }}>
        <div style={{ position: 'absolute', left: 0, top: 46, fontFamily: MONO, fontSize: 28, color: on ? f.accent : f.muted, fontWeight: on ? 700 : 400 }}>bg_{n}</div>
        <div style={{ position: 'absolute', left: 160, top: 0, width: W, height: 120, background: on ? f.accent : '#c9ced8', WebkitMaskImage: `url(${staticFile(`img/wave-${n}.png`)})`, WebkitMaskSize: '100% 100%', maskImage: `url(${staticFile(`img/wave-${n}.png`)})`, maskSize: '100% 100%' }} />
        {on && cuts.map((c, k) => {
          const ct = rise(frame, movesAt + ((cutAt - movesAt) * k) / Math.max(1, cuts.length - 1), fps);
          return <div key={c} style={{ position: 'absolute', left: 160 + (c / MUSIC_S) * W, top: 60 - 72 * ct, width: 4, height: 144 * ct, background: f.ink, opacity: 0.45 }} />;
        })}
        {on && frame >= sitsAt && <div style={{ position: 'absolute', left: 160 + (now / MUSIC_S) * W - 2, top: -10, width: 4, height: 140, background: f.ink }} />}
      </div>
    );
  };
  const page = (
    <AbsoluteFill style={{ background: f.bg }}>
      <Heading field={f} title="Score" sub="Four takes; the one under a voice that moves on every cut" />
      <div style={{ position: 'absolute', left: L, top: 330 }}>{[1, 2, 3, 4].map(wave)}</div>
      <div style={{ position: 'absolute', left: L + 160, top: 930, fontSize: 28, color: f.muted, opacity: rise(frame, sitsAt, fps) }}>
        Lines mark this video's cuts. The bar is where you are.
      </div>
    </AbsoluteFill>
  );
  const lines = [...gen, ...saved, ...pk, ...table];
  return (
    <Scene id="score" field={f} lines={V3.lines}>
      {typingSfx([...gen, ...pk])}
      {saved.map((l) => <Sfx key={`p${l.at}`} at={l.at} file="pop" volume={0.16} />)}
      <Sfx at={turnAt - 6} file="turn" volume={0.35} />
      <Sfx at={sitsAt} file="click" volume={0.4} />
      <PageTurns
        before={
          <AbsoluteFill style={{ background: f.bg }}>
            <Heading field={f} title="Score" sub="tools/music_gen.py and tools/pick_music.py, as they ran" />
            <Terminal title="claude-code-video-toolkit" theme={TERM} lines={lines} width={1680} height={680} fontSize={24} style={{ position: 'absolute', left: L, top: 290 }} />
          </AbsoluteFill>
        }
        pages={[{ at: turnAt, content: page }]}
      />
    </Scene>
  );
};

// ---------------------------------------------------------------- 4 generate
const usd = (v: number) => (v === 0 ? 'free' : `$${v.toFixed(v < 0.1 ? 3 : 2)}`);
const GenerateScene: React.FC = () => {
  const f = FIELDS.record;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const triesAt = V4.cue('three tries');
  const clipAt = T.phrases['04_generate'];
  const billAt = V4.cue('whole bill');
  // the clip plays once, slowed just enough to last until the bill turns over it (never below 0.7x)
  const rate = Math.max(0.7, Math.min(1, (5.04 * fps) / Math.max(1, billAt - clipAt)));
  const tries = [
    { seed: 7, src: 'img/desk-7.png', note: 'two handles on the mug' },
    { seed: 11, src: 'img/desk-11.png', note: 'fine' },
    { seed: 23, src: 'img/desk-23.png', note: 'picked' },
  ];
  const stills = (
    <div style={{ position: 'absolute', left: 1300, top: 290, display: 'flex', flexDirection: 'column', gap: 26 }}>
      {tries.map((s, i) => {
        const t = rise(frame, triesAt + i * 6, fps);
        const on = s.seed === 23;
        return (
          <div key={s.seed} style={{ opacity: t, transform: `translateY(${(1 - t) * 16}px)`, display: 'flex', gap: 22, alignItems: 'center' }}>
            <Img src={staticFile(s.src)} style={{ width: 300, borderRadius: 8, outline: on ? `4px solid ${f.accent}` : 'none', outlineOffset: 4 }} />
            <div style={{ fontSize: 28, color: on ? f.accent : f.muted, fontWeight: on ? 700 : 400, lineHeight: 1.3 }}>seed {s.seed}<br />{s.note}</div>
          </div>
        );
      })}
    </div>
  );
  const page = (
    <AbsoluteFill style={{ background: f.bg }}>
      <Heading field={f} title="What it cost" sub={`Measured from this build: ElevenLabs history, Modal GPU time`} />
      <div style={{ position: 'absolute', left: L - 20, top: 300 }}>
        <BarChart
          bars={bill.rows.map((r) => ({ label: r.label, value: r.usd, note: r.short, highlightAt: r.label === 'Voice' ? V4.cue('voice and all') : undefined }))}
          width={1180}
          height={640}
          theme={f.chart}
          start={billAt + 4}
          stagger={5}
          formatValue={usd}
          fontSize={28}
          radius={4}
        />
      </div>
      <div style={{ position: 'absolute', left: 1400, top: 420, color: f.ink }}>
        <div style={{ fontSize: 36, color: f.muted }}>The whole bill</div>
        <Counter value={bill.total_usd} start={V4.cue('under a dollar')} frames={30} format={(v) => `$${v.toFixed(2)}`} style={{ fontSize: 160, fontWeight: 700, letterSpacing: '-0.04em', color: f.accent }} />
      </div>
    </AbsoluteFill>
  );
  return (
    <Scene id="generate" field={f} lines={V4.lines}>
      {tries.map((_, i) => <Sfx key={i} at={triesAt + i * 6} file="pop" volume={0.18} />)}
      <Sfx at={billAt - 6} file="turn" volume={0.35} />
      <PageTurns
        before={
          <AbsoluteFill style={{ background: f.bg }}>
            <Heading field={f} title="Generate" sub="FLUX.2 for the still, LTX-2.3 to move it, on Modal" />
            <div style={{ position: 'absolute', left: L, top: 290, width: 1120, height: 630, borderRadius: 12, overflow: 'hidden', background: '#d9dde4' }}>
              <Img src={staticFile('img/desk.png')} style={{ position: 'absolute', width: 1120, height: 630, objectFit: 'cover' }} />
              <Sequence from={clipAt}>
                <OffthreadVideo src={staticFile('video/desk.mp4')} playbackRate={rate} muted style={{ width: 1120, height: 630, objectFit: 'cover' }} />
              </Sequence>
            </div>
            {stills}
          </AbsoluteFill>
        }
        pages={[{ at: billAt, content: page }]}
      />
    </Scene>
  );
};

// ---------------------------------------------------------------- 5 compose
const ComposeScene: React.FC = () => {
  const f = FIELDS.record;
  const frame = useCurrentFrame();
  const lines = composeReady() ? composeLines(V5.cue) : [];
  // light the source line whose `at` cue has passed most recently
  const cueAt = (src: string) => {
    const m = src.match(/cue\('([^']+)'\)/);
    return m ? V5.cue(m[1]) : Infinity;
  };
  const lit = source.reduce((best, s, i) => (cueAt(s) <= frame ? i : best), -1);
  return (
    <Scene id="compose" field={f} lines={V5.lines}>
      {typingSfx(lines)}
      <Heading field={f} title="Compose" sub="React, rendered with Remotion. This is the code for this scene." />
      <div data-qc="product" style={{ position: 'absolute', left: L, top: 280, width: 1680, height: 450, background: '#ffffff', borderRadius: 12, padding: '32px 36px', boxSizing: 'border-box', fontFamily: MONO, fontSize: 24, lineHeight: 1.45, color: f.ink, whiteSpace: 'pre', overflow: 'hidden' }}>
        <div style={{ color: f.muted, marginBottom: 12 }}>src/compose.tsx</div>
        {source.map((s, i) => (
          <div key={i} style={{ background: i === lit ? '#dfe7fb' : 'transparent', color: i === lit ? COLORS.primaryDark : f.ink, margin: '0 -12px', padding: '0 12px', borderRadius: 4 }}>{s || ' '}</div>
        ))}
      </div>
      <Terminal
        title="claude-code-video-toolkit"
        theme={TERM}
        lines={lines.length ? lines : [{ at: 0, text: '(the draft render fills this in)', kind: 'dim' }]}
        width={1680}
        height={270}
        fontSize={24}
        style={{ position: 'absolute', left: L, top: 760 }}
      />
    </Scene>
  );
};

// ---------------------------------------------------------------- 6 check
/** A long output line as display rows, the way the terminal wraps it (the words are the tool's). */
const wrapped = (l: CapturedLine, at: number, width = 112): TerminalLine => {
  const rows: string[] = [];
  let row = '';
  for (const w of l.text.split(' ')) {
    if (row && row.length + 1 + w.length > width) { rows.push(row); row = `    ${w}`; } else row = row ? `${row} ${w}` : w;
  }
  rows.push(row);
  return { at, text: rows.join('\n'), kind: l.kind, rows: rows.length };
};
const CheckScene: React.FC = () => {
  const f = FIELDS.lab;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const qc = run('qc');
  const cmd = commandRows(qc.command, 6, 96, 0.4);
  let t = endOf(cmd) + 10;
  // qc's real lines, a subset in their real order, printed across her list of checks; each finding
  // turns red on the word that names it
  const named = (at: number) => (frame >= at ? 'err' : 'out') as CapturedLine['kind'];
  const holdsAt = V6.cue('two still holds');
  const loudAt = V6.cue('twelve decibels');
  const order: [CapturedLine, number | null][] = [
    [pick('qc', 'render: '), null],
    [pickExact('qc', 'render:'), null],
    [{ ...pick('qc', 'LUFS —') }, loudAt],
    [pickExact('qc', 'frames:'), null],
    [{ ...pick('qc', '0:31.0') }, holdsAt],
    [{ ...pick('qc', '0:57.2') }, holdsAt + 8],
    [pick('qc', 'PASS'), null],
  ];
  const span = holdsAt - 4 - t;
  const found: TerminalLine[] = order.map(([l, at], k) =>
    wrapped({ ...l, kind: at === null ? l.kind : named(at) }, t + Math.round((span * k) / (order.length - 1))),
  );
  const fixedAt = V6.cue('both are fixed');
  const final = has('qc-final') ? run('qc-final') : null;
  const lufs = (name: string) => pick(name, 'render: ').text.match(/(-?[0-9.]+) LUFS/)?.[1];
  const after = rise(frame, fixedAt + 6, fps);
  const page = (
    <AbsoluteFill style={{ background: f.bg }}>
      <Heading field={f} title="Check" sub="The same checks, on the next render" />
      <div style={{ position: 'absolute', left: L, top: 320, display: 'flex', gap: 120, color: f.ink, opacity: after }}>
        {[
          ['Loudness', `${lufs('qc')} LUFS`, final ? `${lufs('qc-final')} LUFS` : 'pending'],
          ['Still holds', '2', final ? String(final.lines.filter((l) => l.text.includes('with nothing on screen changing')).length) : 'pending'],
        ].map(([label, was, now]) => (
          <div key={label}>
            <div style={{ fontSize: 36, color: f.muted }}>{label}</div>
            <div style={{ fontSize: 48, marginTop: 18, color: f.muted, textDecoration: 'line-through' }}>{was}</div>
            <div style={{ fontSize: 112, fontWeight: 700, letterSpacing: '-0.04em', color: f.accent, marginTop: 8 }}>{now}</div>
          </div>
        ))}
      </div>
      {final && (
        <Terminal title="claude-code-video-toolkit" theme={TERM} lines={[{ at: fixedAt + 20, ...pick('qc-final', 'PASS') , kind: 'ok' }]} width={1680} height={150} fontSize={24} chrome={false} style={{ position: 'absolute', left: L, top: 800, opacity: after }} />
      )}
    </AbsoluteFill>
  );
  return (
    <Scene id="check" field={f} lines={V6.lines}>
      <Sfx at={0} file="hit" volume={0.4} />
      {typingSfx(cmd)}
      <Sfx at={fixedAt - 6} file="turn" volume={0.35} />
      <PageTurns
        before={
          <AbsoluteFill style={{ background: f.bg }}>
            <Heading field={f} title="Check" sub="tools/qc.py on the draft of this video" />
            <Terminal title="claude-code-video-toolkit" theme={TERM} lines={[...cmd, ...found]} width={1680} height={640} fontSize={24} style={{ position: 'absolute', left: L, top: 290 }} />
          </AbsoluteFill>
        }
        pages={[{ at: fixedAt, content: page }]}
      />
    </Scene>
  );
};

// ---------------------------------------------------------------- 7 close
const CloseScene: React.FC = () => {
  const f = FIELDS.identity;
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = rise(frame, V7.cue('this video out'), fps);
  const b = rise(frame, V7.cue('open source'), fps);
  return (
    <Scene id="close" field={f} lines={V7.lines}>
      <Terminal title="claude-code-video-toolkit · Claude Code" theme={TERM} lines={[{ at: 0, command: PROMPT, perChar: 0.001 }]} width={1680} height={210} fontSize={34} style={{ position: 'absolute', left: L, top: 200 }} />
      <div style={{ position: 'absolute', left: L, top: 500, color: f.ink }}>
        <div style={{ fontSize: 112, fontWeight: 700, letterSpacing: '-0.04em', lineHeight: 1, opacity: a, transform: `translateY(${(1 - a) * 20}px)` }}>One sentence in,<br />this video out.</div>
      </div>
      <div style={{ position: 'absolute', left: L, bottom: 96, color: f.ink, opacity: b }}>
        <div style={{ fontSize: 48, fontWeight: 700, letterSpacing: '-0.02em' }}>{SITE}</div>
        <div style={{ fontSize: 36, marginTop: 12, color: f.muted }}>Open source. Open Claude Code and type /video.</div>
      </div>
    </Scene>
  );
};

const SCENES: [string, React.FC][] = [
  ['open', OpenScene], ['narrate', NarrateScene], ['score', ScoreScene], ['generate', GenerateScene],
  ['compose', ComposeScene], ['check', CheckScene], ['close', CloseScene],
];

const Music: React.FC<{ take: number }> = ({ take }) => {
  const frame = useCurrentFrame();
  const v = interpolate(frame, [0, 45, TOTAL - 75, TOTAL], [0, 0.12, 0.12, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return <Audio src={staticFile(`audio/music/bg_${take}.mp3`)} volume={v} />;
};

/** `music`: the take to play (default PICK), or 0 for none (a render to audition takes against). */
export const MadeWithItself: React.FC<{ music?: number }> = ({ music = PICK }) => (
  <AbsoluteFill style={{ background: COLORS.bgAlt }}>
    {music > 0 && <Music take={music} />}
    {SCENES.map(([id, C]) => (
      <Sequence key={id} from={T.starts[id]} durationInFrames={T.scenes[id]}><C /></Sequence>
    ))}
  </AbsoluteFill>
);
