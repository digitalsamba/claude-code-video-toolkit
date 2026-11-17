// The whole lecture is lecture.json (what is said, chalked and acted) and timeline.json (when, written by
// `uv run tools/lecture.py timeline`). The code that draws it is lib/lecture; this file only registers it.
import { Composition, Still } from 'remotion';
import { makeLecture, type Lecture, type Timeline } from '../../../lib/lecture';
import lecture from '../lecture.json';
import timeline from '../timeline.json';
import { CUSTOM } from './custom';

const L = makeLecture(lecture as Lecture, timeline as Timeline, CUSTOM);

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Lecture" component={L.Video} durationInFrames={L.durationInFrames} fps={L.fps} width={1920} height={1080} />
    <Still id="BoardBg" component={L.BoardBg} width={1920} height={1080} />
    <Still id="BoardTitle" component={L.BoardTitle} width={1920} height={1080} />
  </>
);
