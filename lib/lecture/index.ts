/**
 * A lecture at a chalkboard: a presenter rendered in Blender (transparent clips, one a segment) in front of a board
 * Remotion draws, the narration's words driving the chalk, the camera's windows and the acting alike.
 *
 * A project passes its lecture.json and timeline.json (tools/lecture.py writes the timeline) to makeLecture and
 * registers what comes back. templates/lecture is the worked example; docs/blender.md "A lecture" is the method.
 */
export { makeLecture, Title } from './Lecture';
export type { Custom } from './Lecture';
export { makeTimeline } from './timeline';
export type { At, BoardItem, Lecture, LectureTimeline, Segment, Short, Timeline } from './timeline';
export { Chalk, Stroke, Tap, Seg, ChalkDefs, SIZE, WHITE, YELLOW, HAND, line, box, arrow, tick, cross, ring, under } from './chalk';
export { Board, BoardDrawn } from './Board';
export { Camera, cameraAt } from './Camera';
export { Presenter, Inset, PropSounds, Room, useManifest, presenterBox, insetRegion } from './Presenter';
