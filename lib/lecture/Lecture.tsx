// A lecture at a chalkboard, as data: the board, what each segment chalks on it, the windows the camera goes
// into, the presenter's clips in front, and the narration. A lecture project (templates/lecture) passes in its
// lecture.json and timeline.json and gets the compositions back; it writes no code of its own, beyond any
// drawing it wants that chalk lines and strokes cannot make (src/custom.tsx).
import React from 'react';
import { AbsoluteFill, Audio, Easing, Img, Loop, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { Captions, captionsFor } from '../components/Captions';
import { loadFont } from '@remotion/google-fonts/Inter';
import { Board, BoardDrawn } from './Board';
import { Camera } from './Camera';
import { Chalk, ChalkDefs, SIZE, Seg, Stroke, Tap, WHITE, YELLOW, arrow, box, cross, line, ring, tick, under } from './chalk';
import defaults from './defaults.json';
import { Closes, Inset, Presenter, PropSounds, Room, presenterBox } from './Presenter';
import { makeTimeline, type BoardItem, type Lecture, type LectureTimeline, type Short, type Timeline } from './timeline';

export const { fontFamily: SANS } = loadFont('normal', { weights: ['400', '600'], subsets: ['latin'] });

/** Drawings a project adds (src/custom.tsx): each is given the frame it starts (`at`) and its item's props. */
export type Custom = Record<string, React.FC<{ at: number } & Record<string, unknown>>>;

const HX = 70;   // headings' left edge
const HB = 205;  // a heading's top. One size and one edge for every heading (l: at xl a heading in a segment with
                 // a window ran into the pane under it, so the two kinds of segment had two heading sizes)
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

/** A pane: a cleaner, darker slate in a plain sans, with a title and a source line, so something real never reads
 * as a sketch. A picture (public/), or a code listing set in a monospace. */
type Rect = { x: number; y: number; w: number; h: number };
/** Lines a pane's source line takes at its width (a short's panes are narrower for their type). */
const sourceLines = (source: string | undefined, w: number, t: number) => (source && source.length * 22 * t * 0.5 > w - 40 ? 2 : 1);
const Pane: React.FC<{ item: NonNullable<BoardItem['pane']>; at: number; out?: number; videoFrames?: number; rect?: Rect; t?: number }> = ({ item, at, out, videoFrames, rect, t = 1 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < at || (out !== undefined && frame > out + 8)) return null;
  const o = interpolate(frame, [at, at + 10], [0, 1], clamp) * (out === undefined ? 1 : interpolate(frame, [out, out + 8], [1, 0], clamp));
  // (a short places it on its own frame, `rect`, and sets its title and source line `t` times their size)
  const { x, y, w, h } = rect ?? item;
  const { title, source, image, video, code } = item;
  const box = { position: 'absolute', left: 20, top: 56 * t, width: w - 40, height: h - (106 + 26 * (sourceLines(source, w, t) - 1)) * t, objectFit: 'contain' } as const;
  // a video plays once from the pane's start, `from` seconds in, then holds its last frame
  const skip = Math.round((item.from ?? 0) * fps);
  const last = Math.max(skip, (videoFrames ?? Infinity) - 1);
  return (
    <div data-qc="product" style={{ position: 'absolute', left: x, top: y, width: w, height: h, opacity: o, background: '#1b2422',
      border: '2px solid rgba(243,240,230,0.75)', borderRadius: 6, fontFamily: SANS, color: WHITE, overflow: 'hidden' }}>
      <div style={{ position: 'absolute', left: 20, top: 12 * t, fontSize: 26 * t, fontWeight: 600 }}>{title}</div>
      {image ? <Img src={staticFile(image)} style={box} /> : null}
      {video ? (
        // (placed so its own frame is the video's: plays from `at`, then holds the last frame. Not <Freeze>: under
        // the board's nested sequences it showed a frame some 200 ahead of the one asked for)
        <Sequence from={frame - Math.min(frame - at + skip, last)} layout="none">
          <OffthreadVideo src={staticFile(video)} muted style={box} />
        </Sequence>
      ) : null}
      {code ? (
        <pre style={{ position: 'absolute', left: 20, top: 56, right: 20, margin: 0, fontFamily: 'Menlo, Consolas, monospace', fontSize: 26,
          lineHeight: 1.35, color: '#d9e0dc', whiteSpace: 'pre' }}>{code}</pre>
      ) : null}
      {source ? <div style={{ position: 'absolute', left: 20, bottom: 10 * t, fontSize: 22 * t, color: '#aebbb4' }}>{source}</div> : null}
    </div>
  );
};

/** A short's pane that takes over: the picture fills `box` and the view moves from region to region on its words. */
const Takeover: React.FC<{ item: NonNullable<BoardItem['pane']>; at: number; out: number; keys: { at: number; region: number[] }[];
  box: Rect; videoFrames?: number; t: number }> = ({ item, at, out, keys, box, videoFrames, t }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < at - 6 || frame > out + 6) return null;
  const o = interpolate(frame, [at - 6, at, out, out + 6], [0, 1, 1, 0], clamp);
  const tk = item.takeover!;
  // the region now: each key eases in over 18 frames from the one before
  let r = keys[0].region;
  for (const k of keys.slice(1)) {
    const e = Easing.inOut(Easing.cubic)(interpolate(frame, [k.at, k.at + 18], [0, 1], clamp));
    if (e <= 0) break;
    r = r.map((v, i) => v + (k.region[i] - v) * e);
  }
  const s = Math.min(box.w / r[2], box.h / r[3]);
  const left = (box.w - r[2] * s) / 2 - r[0] * s;
  const top = -r[1] * s; // (a region wider than the box sits at its top: the space under it is the captions')
  const pic = { position: 'absolute', left, top, width: tk.size[0] * s, height: tk.size[1] * s } as const;
  const skip = Math.round((item.from ?? 0) * fps);
  const last = Math.max(skip, (videoFrames ?? Infinity) - 1);
  return (
    <div data-qc="product" style={{ position: 'absolute', left: box.x, top: box.y, width: box.w, height: box.h, overflow: 'hidden', opacity: o, background: '#14161a' }}>
      {item.image ? <Img src={staticFile(item.image)} style={pic} /> : null}
      {item.video ? (
        <Sequence from={frame - Math.min(frame - at + skip, last)} layout="none">
          <OffthreadVideo src={staticFile(item.video)} muted style={pic} />
        </Sequence>
      ) : null}
      {tk.label ? (
        // (named while the whole picture is up; gone as the view moves in, so it covers nothing being shown)
        <div style={{ position: 'absolute', right: 16, bottom: 14, padding: '4px 12px', borderRadius: 4, background: 'rgba(16,18,22,0.85)',
          fontFamily: SANS, fontSize: 22 * t, color: '#c9d0cc',
          opacity: keys.length > 1 ? interpolate(frame, [keys[1].at, keys[1].at + 8], [1, 0], clamp) : 1 }}>{tk.label}</div>
      ) : null}
    </div>
  );
};

const strokePath = (item: BoardItem, seed: number): string => {
  const p = item.points ?? [];
  switch (item.stroke) {
    case 'line': return line(p[0], p[1], p[2], p[3], seed);
    case 'arrow': return arrow(p[0], p[1], p[2], p[3], seed);
    case 'box': return box(p[0], p[1], p[2], p[3], seed);
    case 'ring': return ring(p[0], p[1], p[2], p[3]);
    case 'tick': return tick(p[0], p[1], p[2]);
    case 'cross': return cross(p[0], p[1], p[2]);
    case 'under': return under(p[0], p[1], p[2], p[3], seed);
    default: return item.d ?? '';
  }
};

/** One segment's items on the board, chalked on their words, wiped before the next segment. */
const SegmentBoard: React.FC<{ L: LectureTimeline; lecture: Lecture; n: number; custom: Custom; written?: number; until?: number }> = ({ L, lecture, n, custom, written, until }) => {
  const seg = lecture.segments[n];
  // `written`: the frame a short starts on, with this segment's heading already up (its first frame is the cover)
  return (
    <Seg from={written ?? (n === 0 && L.first > 0 ? L.first : L.S(n) - 12)} to={until ?? L.N(n) - 20}>
      {/* after the room shots, the title they showed on the wall is still up, and goes with the first segment */}
      {n === 0 && L.first > 0 ? <Title lecture={lecture} /> : null}
      {seg.heading ? (
        <Chalk x={HX} y={HB} size={SIZE.l} at={written !== undefined ? -100 : L.S(n) - 4} quiet={written !== undefined}>{seg.heading}</Chalk>
      ) : null}
      {(seg.board ?? []).map((item, k) => {
        const at = L.at(n, item.at);
        if (item.chalk !== undefined) {
          return (
            <Chalk key={k} x={item.x ?? HX} y={item.y ?? 360} size={SIZE[item.size ?? 'm']} color={item.color === 'yellow' ? YELLOW : WHITE}
              weight={item.weight} tilt={item.tilt} width={item.width} dur={item.dur} quiet={item.quiet} at={at}>{item.chalk}</Chalk>
          );
        }
        if (item.stroke) {
          return <Stroke key={k} d={strokePath(item, k + 2)} at={at} dur={item.dur} width={6} dotted={item.dotted}
            color={item.color === 'yellow' ? YELLOW : WHITE} quiet={false} />;
        }
        if (item.tap) return <Tap key={k} at={at} />;
        if (item.component) {
          const C = custom[item.component];
          if (!C) throw new Error(`${seg.id}: no drawing called "${item.component}" in src/custom.tsx (have: ${Object.keys(custom).join(', ') || 'none'})`);
          return <C key={k} at={at} {...(item.props ?? {})} />;
        }
        return null;
      })}
    </Seg>
  );
};

/** The panes are drawn without the chalk filter (they are real things, not chalk). */
const SegmentPanes: React.FC<{ L: LectureTimeline; lecture: Lecture; n: number }> = ({ L, lecture, n }) => (
  <>
    {(lecture.segments[n].board ?? []).filter((i) => i.pane).map((item, k) => (
      <Pane key={k} item={item.pane!} at={L.at(n, item.at)} out={item.pane!.out !== undefined ? L.at(n, item.pane!.out) : L.N(n) - 20}
        videoFrames={item.pane!.video ? L.tl.videos?.[item.pane!.video] : undefined} />
    ))}
  </>
);

/** The lecture's title, chalked at the top of the board: the still the room shots put on the wall. */
export const Title: React.FC<{ lecture: Lecture }> = ({ lecture }) => (
  <>
    <Chalk x={HX} y={52} size={SIZE.l} at={-100} quiet>{lecture.title}</Chalk>
    <Stroke d={under(HX, 52, SIZE.l, Math.min(1500, lecture.title.length * 30), 3)} at={-100} width={5} />
  </>
);

/** The compositions of a lecture: the video, the bare board (a still the video uses) and the titled board. */
export const makeLecture = (lecture: Lecture, timeline: Timeline, custom: Custom = {}) => {
  const L = makeTimeline(lecture, timeline);
  const tone = defaults.room_tone;
  const Video: React.FC = () => (
    <AbsoluteFill style={{ background: '#000' }}>
      <ChalkDefs />
      {L.first > 0 ? <Sequence durationInFrames={L.first}><Room L={L} /></Sequence> : null}
      <Sequence from={L.first} layout="none">
        <AbsoluteFill style={{ overflow: 'hidden', background: '#222d2a' }}>
          <Sequence from={-L.first} layout="none">
            <Camera moves={L.tl.moves}>
              <Board />
              <AbsoluteFill style={{ filter: 'url(#chalkFx)' }}>
                {lecture.segments.map((_, n) => <SegmentBoard key={n} L={L} lecture={lecture} n={n} custom={custom} />)}
              </AbsoluteFill>
              {lecture.segments.map((_, n) => <SegmentPanes key={n} L={L} lecture={lecture} n={n} />)}
              <Presenter L={L} />
            </Camera>
            <Inset L={L} />
            <Closes L={L} />
          </Sequence>
        </AbsoluteFill>
      </Sequence>
      {L.tl.opening?.voice ? (
        // the line said over the walk in (opening.voice)
        <Sequence from={L.tl.opening.voice.startFrame} layout="none">
          <Audio src={staticFile(`audio/scenes/${L.tl.opening.voice.id}.mp3`)} />
        </Sequence>
      ) : null}
      {L.tl.segments.map((s) => (
        <Sequence key={s.id} from={s.startFrame} layout="none">
          <Audio src={staticFile(`audio/scenes/${s.id}.mp3`)} />
        </Sequence>
      ))}
      <Loop durationInFrames={tone.frames} layout="none">
        <Audio src={staticFile(tone.file)} volume={(f) => tone.volume * Math.min(1, f / 20, (tone.frames - f) / 20)} />
      </Loop>
      <PropSounds L={L} lecture={lecture} />
    </AbsoluteFill>
  );
  const BoardTitle: React.FC = () => (
    <AbsoluteFill>
      <ChalkDefs />
      <Board />
      <AbsoluteFill style={{ filter: 'url(#chalkFx)' }}><Title lecture={lecture} /></AbsoluteFill>
    </AbsoluteFill>
  );
  // The board alone on any frame of the video (its chalk and panes, no presenter, the camera at home): the still a
  // closer shot's hall hangs on its wall (`tools/lecture.py boards`).
  const BoardAt: React.FC = () => (
    <AbsoluteFill style={{ background: '#222d2a' }}>
      <ChalkDefs />
      <Board />
      <AbsoluteFill style={{ filter: 'url(#chalkFx)' }}>
        {lecture.segments.map((_, n) => <SegmentBoard key={n} L={L} lecture={lecture} n={n} custom={custom} />)}
      </AbsoluteFill>
      {lecture.segments.map((_, n) => <SegmentPanes key={n} L={L} lecture={lecture} n={n} />)}
    </AbsoluteFill>
  );
  const shorts = (lecture.shorts ?? []).map((spec) => makeShort(L, lecture, spec, custom, tone));
  const sv = { ...defaults.short, ...(lecture.shorts?.[0] ?? {}) };
  const BoardShortBg: React.FC = () => <BoardDrawn w={sv.width} h={sv.height} tray={shortTray(L, sv)} />;
  return { Video, BoardBg: BoardDrawn, BoardTitle, BoardAt, BoardShortBg, shorts, durationInFrames: L.tl.totalFrames, fps: L.tl.fps, L };
};

type ShortLayout = typeof defaults.short;

/** Where a short's tray is: the board's own, as the presenter's clip carries it down to the short's frame. */
const shortTray = (L: LectureTimeline, o: Pick<ShortLayout, 'presenter_top' | 'width'>) => {
  const B = presenterBox(L);
  return Math.round(o.presenter_top + (1032 - B.top) * (o.width / B.width));
};

/**
 * A short (lecture.json "shorts"): some consecutive segments as a vertical video, from the same clips and audio. The
 * board's chalk column on top, the presenter's clip full width below it, a pane full width where the long video goes
 * into a window, the captions between them. Frames inside are the lecture's own, shifted to start at the first
 * segment's clip: every cue, pane and sound is where it is in the long video.
 */
const makeShort = (L: LectureTimeline, full: Lecture, spec: Short, custom: Custom, tone: typeof defaults.room_tone) => {
  // a short can replace a segment's board (its own chalk line, a pane that takes over); the cues stay the lecture's
  const lecture: Lecture = { ...full, segments: full.segments.map((s) => (spec.board?.[s.id] ? { ...s, board: spec.board[s.id] } : s)) };
  const o: ShortLayout = { ...defaults.short, ...spec, captions: { ...defaults.short.captions, ...(spec.captions ?? {}) } };
  const ids = lecture.segments.map((s) => s.id);
  const idx = spec.segments.map((id) => {
    const n = ids.indexOf(id);
    if (n < 0) throw new Error(`short "${spec.id}": no segment "${id}" (have: ${ids.join(', ')})`);
    return n;
  });
  if (!idx.length || idx.some((n, i) => i > 0 && n !== idx[i - 1] + 1)) {
    throw new Error(`short "${spec.id}": its segments must follow each other in the lecture (${spec.segments.join(', ')})`);
  }
  const n0 = idx[0];
  const n1 = idx[idx.length - 1];
  const from = L.clips[n0].from;
  const to = Math.min(L.E(n1) + o.tail, L.clips[n1].until);
  const [cx, cy, cw] = o.column;
  const k = o.width / cw;
  const B = presenterBox(L);
  const pk = o.width / B.width;
  const pbox = { left: 0, top: o.presenter_top, width: o.width, height: Math.round(B.height * pk) };
  const paneRect = (p: NonNullable<BoardItem['pane']>): Rect => {
    const w = o.width - 2 * o.gutter;
    const h = Math.round((p.h * w) / p.w + 26 * o.chrome * (sourceLines(p.source, w, o.chrome) - 1));
    const [b0, b1] = o.pane_band;
    return { x: o.gutter, y: Math.round(Math.max(b0, (b0 + b1 - h) / 2)), w, h };
  };
  // `cover`: [segment id, n] puts that segment's nth pane (from 0) on frame 0, the Shorts cover
  const cover = spec.cover
    ? (lecture.segments.find((x) => x.id === spec.cover![0])?.board ?? []).filter((i) => i.pane)[spec.cover[1]]?.pane
    : undefined;
  if (spec.cover && !cover) throw new Error(`short "${spec.id}": no pane ${spec.cover[1]} in segment "${spec.cover[0]}"`);
  // panes that take the frame: when, where the view goes, and the frames the presenter steps out for
  const takeovers = idx.flatMap((n) => (lecture.segments[n].board ?? []).filter((i) => i.pane?.takeover).map((item) => {
    const p = item.pane!;
    const at = L.at(n, item.at);
    const end = p.out !== undefined ? L.at(n, p.out) : L.N(n) - 20;
    // (one that runs to the last word holds to the short's end: the short ends on it, as on a walk in that lands)
    const out = n === n1 && end >= L.E(n1) - 1 ? to + 6 : end;
    return { item: p, at, out, keys: p.takeover!.keys.map((k) => ({ at: L.at(n, k.at), region: k.region })) };
  }));
  const [ty0, ty1] = o.takeover_band;
  const tbox: Rect = { x: 0, y: ty0, w: o.width, h: ty1 - ty0 };
  const Video: React.FC = () => (
    <AbsoluteFill style={{ background: '#222d2a' }}>
      <ChalkDefs />
      <Board still="board/bg-short.png" w={o.width} h={o.height} tray={shortTray(L, o)} />
      <Sequence from={-from} layout="none">
        {/* the board's chalk, a column of it, scaled to the width */}
        <div style={{ position: 'absolute', left: 0, top: 0, width: 1920, height: 1080, transformOrigin: '0 0',
          transform: `translate(${-cx * k}px, ${o.top - cy * k}px) scale(${k})` }}>
          <AbsoluteFill style={{ filter: 'url(#chalkFx)' }}>
            {idx.map((n) => <SegmentBoard key={n} L={L} lecture={lecture} n={n} custom={custom} written={n === n0 ? from : undefined}
              until={n === n1 ? to + 30 : undefined} />)}
          </AbsoluteFill>
        </div>
        {cover ? (
          // the cover: one of the short's panes, up from frame 0 and gone as the first word is said
          <Pane item={cover} at={from - 10} out={L.S(n0) - 8} rect={paneRect(cover)} t={o.chrome}
            videoFrames={cover.video ? L.tl.videos?.[cover.video] : undefined} />
        ) : null}
        {takeovers.map((tk, j) => (
          <Takeover key={`t${j}`} item={tk.item} at={tk.at} out={tk.out} keys={tk.keys} box={tbox} t={o.chrome}
            videoFrames={tk.item.video ? L.tl.videos?.[tk.item.video] : undefined} />
        ))}
        {idx.map((n) => (lecture.segments[n].board ?? []).filter((i) => i.pane && !i.pane.takeover).map((item, j) => {
          const out = item.pane!.out !== undefined ? L.at(n, item.pane!.out) : L.N(n) - 20;
          return (
            // (a pane that would fade out in the short's last frames stays up: the short ends on it)
            <Pane key={`${n}-${j}`} item={item.pane!} at={L.at(n, item.at)} out={out >= to - 10 ? undefined : out}
              videoFrames={item.pane!.video ? L.tl.videos?.[item.pane!.video] : undefined} rect={paneRect(item.pane!)} t={o.chrome} />
          );
        }))}
        {idx.map((n) => (
          <Sequence key={n} from={L.S(n)} layout="none">
            <Audio src={staticFile(`audio/scenes/${L.tl.segments[n].id}.mp3`)} />
          </Sequence>
        ))}
        <PropSounds L={L} lecture={lecture} />
      </Sequence>
      <Presenter L={L} box={pbox} k={pk} offset={from} hidden={o.takeover_presenter === 'stay' ? [] : takeovers.map((t) => [t.at, t.out] as [number, number])} />
      <Sequence from={-from} layout="none">
        {idx.map((n) => (
          <Sequence key={n} from={L.S(n)} layout="none">
            <Captions src={captionsFor(`audio/scenes/${L.tl.segments[n].id}.mp3`)} fontFamily={SANS} {...o.captions} />
          </Sequence>
        ))}
      </Sequence>
      <Loop durationInFrames={tone.frames} layout="none">
        <Audio src={staticFile(tone.file)} volume={(f) => tone.volume * Math.min(1, f / 20, (tone.frames - f) / 20)} />
      </Loop>
    </AbsoluteFill>
  );
  return { id: spec.id, Video, durationInFrames: to - from, width: o.width, height: o.height, from };
};
