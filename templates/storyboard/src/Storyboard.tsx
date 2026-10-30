import {AbsoluteFill, Audio, Img, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Captions} from '../../../lib/components/Captions';
import {videoLayout} from '../../../lib/formats';
import {parseStoryProject} from './model';
import type {StoryProject, StoryScene} from './model';

function Scene({scene, palette, safeZones, direction}: {scene: StoryScene; palette: StoryProject['palette']; safeZones: boolean; direction: StoryProject['direction']}) {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const layout = videoLayout({width, height});
  const fontSize = Math.max(32, Math.round(layout.fontSize * Math.min(1.25, 28 / Math.max(28, scene.title.length))));
  const opacity = interpolate(frame, [0, 10], [0, 1], {extrapolateRight: 'clamp'});
  return <AbsoluteFill style={{backgroundColor: palette.background, color: palette.foreground, fontFamily: 'Georgia, serif', direction}}>
    <div style={{position: 'absolute', top: layout.safe.top, left: layout.safe.left,
      width: layout.contentWidth, height: layout.contentHeight - layout.captionFontSize * 2,
      display: 'flex', flexDirection: layout.wide ? 'row' : 'column', gap: layout.gap, opacity}}>
      <div style={{flex: layout.wide ? '1 1 48%' : '0 0 auto', minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
        <div style={{fontSize: Math.max(28, layout.fontSize * .25), letterSpacing: 4, textTransform: 'uppercase', color: palette.accent, marginBottom: layout.gap}}>{scene.type}</div>
        <h1 style={{fontSize, lineHeight: 1.05, margin: 0, overflowWrap: 'anywhere'}}>{scene.title}</h1>
        <p style={{fontSize: layout.fontSize * .43, lineHeight: 1.5, marginTop: layout.gap, overflowWrap: 'anywhere'}}>{scene.text}</p>
      </div>
      <div style={{position: 'relative', flex: '1 1 52%', minHeight: 0, minWidth: 0,
        background: '#ffffff0d', border: `2px solid ${palette.accent}55`, borderRadius: layout.gap / 2, overflow: 'hidden',
        display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
        {scene.asset?.kind === 'video' ? <OffthreadVideo src={staticFile(scene.asset.file)} muted style={{width: '100%', height: '100%', objectFit: 'contain'}} />
          : scene.asset?.kind === 'image' ? <Img src={staticFile(scene.asset.file)} style={{width: '100%', height: '100%', objectFit: 'contain'}} />
          : <div style={{fontSize: layout.fontSize * .6, padding: layout.gap, textAlign: 'center', color: palette.accent}}>{scene.type === 'cta' ? 'Make your next video.' : 'Your product, in focus.'}</div>}
        {scene.narratorFile && <OffthreadVideo src={staticFile(scene.narratorFile)} muted style={{position: 'absolute', right: layout.gap / 2, bottom: layout.gap / 2, width: '30%', aspectRatio: '1', objectFit: 'cover', borderRadius: '50%'}} />}
      </div>
    </div>
    {scene.audioFile && <Audio src={staticFile(scene.audioFile)} />}
    {scene.captionsFile && <Captions src={scene.captionsFile} fontSize={layout.captionFontSize} bottom={layout.safe.bottom} maxWidth={`${layout.contentWidth}px`} />}
    {safeZones && <div data-safe-zone style={{position: 'absolute', inset: `${layout.safe.top}px ${layout.safe.right}px ${layout.safe.bottom}px ${layout.safe.left}px`, border: `3px dashed ${palette.accent}`, pointerEvents: 'none'}} />}
  </AbsoluteFill>;
}

export type StoryboardProps = {project: unknown; safeZones: boolean};
export function Storyboard({project: input, safeZones}: StoryboardProps) {
  const project = parseStoryProject(input);
  const {fps} = useVideoConfig();
  let start = 0;
  return <AbsoluteFill>{project.scenes.map(scene => {
    const from = start;
    const durationInFrames = Math.max(1, Math.round(scene.durationSeconds * fps));
    start += durationInFrames;
    return <Sequence key={scene.id} from={from} durationInFrames={durationInFrames}>
      <Scene scene={scene} palette={project.palette} direction={project.direction} safeZones={safeZones} />
    </Sequence>;
  })}</AbsoluteFill>;
}
