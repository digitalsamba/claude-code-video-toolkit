import {Composition} from 'remotion';
import {VIDEO_FORMATS} from '../../../lib/formats';
import {Storyboard} from './Storyboard';
import type {StoryboardProps} from './Storyboard';
import {parseStoryProject} from './model';
import project from '../project.json';

export function RemotionRoot() {
  return <>{Object.entries(VIDEO_FORMATS).map(([format, size]) => <Composition
    key={format} id={format === 'landscape' ? 'Storyboard' : format === 'portrait' ? 'StoryboardPortrait' : 'StoryboardSquare'}
    component={Storyboard} defaultProps={{project, safeZones: false}} width={size.width} height={size.height}
    fps={30} durationInFrames={300} calculateMetadata={({props}: {props: StoryboardProps}) => ({
      durationInFrames: parseStoryProject(props.project).scenes.reduce((total, scene) => total + Math.max(1, Math.round(scene.durationSeconds * 30)), 0),
    })} />)}</>;
}
