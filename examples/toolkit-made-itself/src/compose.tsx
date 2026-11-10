/**
 * Scene 5's terminal. The part between the <shown> markers is printed on screen as the code panel
 * (scripts/source.py copies it into src/source.json), so the code you see is the code that runs.
 */
import type { TerminalLine } from '../../../lib/components';
import { has, pick, pickLast, run } from './captured';

type Cue = (phrase: string) => number;

// <shown>
export const composeLines = (cue: Cue): TerminalLine[] => {
  const render = run('render');
  return [
    { at: cue('every scene'), command: render.command, prompt: '$', perChar: 0.5 },
    { at: cue('rendered'), ...pick('render', 'Composition') },
    { at: cue('remotion'), ...pick('render', 'Codec') },
    { at: cue('component'), ...pickLast('render', 'Encoded') },
    { at: cue('each line'), ...pick('render', ' MB'), kind: 'ok' },
    { at: cue('names'), command: '', prompt: '$' },
  ];
};
// </shown>

export const composeReady = () => has('render');
