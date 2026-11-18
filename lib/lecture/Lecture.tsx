// A lecture at a chalkboard, as data: the board, what each segment chalks on it, the windows the camera goes
// into, the presenter's clips in front, and the narration. A lecture project (templates/lecture) passes in its
// lecture.json and timeline.json and gets the compositions back; it writes no code of its own, beyond any
// drawing it wants that chalk lines and strokes cannot make (src/custom.tsx).
import React from 'react';
import { AbsoluteFill, Audio, Img, Loop, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { loadFont } from '@remotion/google-fonts/Inter';
import { Board, BoardDrawn } from './Board';
import { Camera } from './Camera';
import { Chalk, ChalkDefs, SIZE, Seg, Stroke, Tap, WHITE, YELLOW, arrow, box, cross, line, ring, tick, under } from './chalk';
import defaults from './defaults.json';
import { Inset, Presenter, PropSounds, Room } from './Presenter';
import { makeTimeline, type BoardItem, type Lecture, type LectureTimeline, type Timeline } from './timeline';

export const { fontFamily: SANS } = loadFont('normal', { weights: ['400', '600'], subsets: ['latin'] });

/** Drawings a project adds (src/custom.tsx): each is given the frame it starts (`at`) and its item's props. */
export type Custom = Record<string, React.FC<{ at: number } & Record<string, unknown>>>;

const HX = 70;   // headings' left edge
const HB = 205;  // a heading's top. One size and one edge for every heading (l: at xl a heading in a segment with
                 // a window ran into the pane under it, so the two kinds of segment had two heading sizes)
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;

/** A pane: a cleaner, darker slate in a plain sans, with a title and a source line, so something real never reads
 * as a sketch. A picture (public/), or a code listing set in a monospace. */
const Pane: React.FC<{ item: NonNullable<BoardItem['pane']>; at: number; out?: number; videoFrames?: number }> = ({ item, at, out, videoFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame < at || (out !== undefined && frame > out + 8)) return null;
  const o = interpolate(frame, [at, at + 10], [0, 1], clamp) * (out === undefined ? 1 : interpolate(frame, [out, out + 8], [1, 0], clamp));
  const { x, y, w, h, title, source, image, video, code } = item;
  const box = { position: 'absolute', left: 20, top: 56, width: w - 40, height: h - 106, objectFit: 'contain' } as const;
  // a video plays once from the pane's start, `from` seconds in, then holds its last frame
  const skip = Math.round((item.from ?? 0) * fps);
  const last = Math.max(skip, (videoFrames ?? Infinity) - 1);
  return (
    <div data-qc="product" style={{ position: 'absolute', left: x, top: y, width: w, height: h, opacity: o, background: '#1b2422',
      border: '2px solid rgba(243,240,230,0.75)', borderRadius: 6, fontFamily: SANS, color: WHITE, overflow: 'hidden' }}>
      <div style={{ position: 'absolute', left: 20, top: 12, fontSize: 26, fontWeight: 600 }}>{title}</div>
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
      {source ? <div style={{ position: 'absolute', left: 20, bottom: 10, fontSize: 22, color: '#aebbb4' }}>{source}</div> : null}
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
const SegmentBoard: React.FC<{ L: LectureTimeline; lecture: Lecture; n: number; custom: Custom }> = ({ L, lecture, n, custom }) => {
  const seg = lecture.segments[n];
  return (
    <Seg from={n === 0 && L.first > 0 ? L.first : L.S(n) - 12} to={L.N(n) - 20}>
      {/* after the room shots, the title they showed on the wall is still up, and goes with the first segment */}
      {n === 0 && L.first > 0 ? <Title lecture={lecture} /> : null}
      {seg.heading ? (
        <Chalk x={HX} y={HB} size={SIZE.l} at={L.S(n) - 4}>{seg.heading}</Chalk>
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
          </Sequence>
        </AbsoluteFill>
      </Sequence>
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
  return { Video, BoardBg: BoardDrawn, BoardTitle, durationInFrames: L.tl.totalFrames, fps: L.tl.fps, L };
};
