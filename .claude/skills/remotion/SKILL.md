---
name: remotion
description: Create programmatic videos using React with Remotion. Use when building video compositions, animations, or rendering videos programmatically. Triggers include creating video content with React, animating elements frame-by-frame, rendering videos to MP4/WebM/GIF, using spring/interpolate animations, building video editing apps, or deploying video rendering to AWS Lambda.
---

# Remotion

Remotion is a framework for creating videos programmatically using React. Videos are functions of frames over time.

## Core Concepts

**Video Properties:**
- `width`, `height`: Dimensions in pixels
- `fps`: Frames per second
- `durationInFrames`: Total frames (duration = durationInFrames / fps)

**First frame is 0, last frame is durationInFrames - 1.**

## Project Setup

```bash
npx create-video@latest     # Create new project
npx remotion studio         # Launch preview studio
npx remotion render src/index.ts CompositionId out.mp4  # Render video
```

## Essential Hooks

```tsx
import { useCurrentFrame, useVideoConfig } from 'remotion';

const MyComponent = () => {
  const frame = useCurrentFrame();  // Current frame number (0-indexed)
  const { fps, durationInFrames, width, height } = useVideoConfig();
  return <div>Frame {frame} of {durationInFrames}</div>;
};
```

## Composition Registration

Register compositions in `src/Root.tsx`:

```tsx
import { Composition } from 'remotion';

export const RemotionRoot = () => (
  <>
    <Composition
      id="MyVideo"
      component={MyComponent}
      durationInFrames={150}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{ title: 'Hello' }}
    />
    <Composition id="AnotherVideo" ... />
  </>
);
```

## Animation with interpolate()

Map frame values to animation properties:

```tsx
import { interpolate, useCurrentFrame } from 'remotion';

const frame = useCurrentFrame();

// Fade in from frame 0-20
const opacity = interpolate(frame, [0, 20], [0, 1], {
  extrapolateRight: 'clamp',  // Prevent values > 1
});

// Fade in and out
const { durationInFrames } = useVideoConfig();
const fadeOpacity = interpolate(
  frame,
  [0, 20, durationInFrames - 20, durationInFrames],
  [0, 1, 1, 0]
);
```

**Options:** `extrapolateLeft`, `extrapolateRight`: `'extend'` | `'clamp'` | `'identity'` | `'wrap'`

## Spring Animations

Physics-based animations:

```tsx
import { spring, useCurrentFrame, useVideoConfig } from 'remotion';

const frame = useCurrentFrame();
const { fps } = useVideoConfig();

const scale = spring({
  frame,
  fps,
  config: { damping: 10, stiffness: 100, mass: 1 },
  durationInFrames: 30,  // Optional: constrain duration
});

// Combine spring with interpolate for custom ranges
const translateX = interpolate(spring({ frame, fps }), [0, 1], [0, 200]);
```

## Easing Functions

```tsx
import { interpolate, Easing } from 'remotion';

const value = interpolate(frame, [0, 30], [0, 100], {
  easing: Easing.bezier(0.25, 0.1, 0.25, 1),
  // Or: Easing.ease, Easing.inOut(Easing.quad), Easing.bounce, etc.
});
```

## Sequencing Content

**Sequence** - Time-shift children (useCurrentFrame becomes relative):

```tsx
import { Sequence } from 'remotion';

<Sequence from={30} durationInFrames={60}>
  <MyComponent />  {/* useCurrentFrame() starts at 0 when parent is at frame 30 */}
</Sequence>
```

**Series** - Play items one after another:

```tsx
import { Series } from 'remotion';

<Series>
  <Series.Sequence durationInFrames={30}><Intro /></Series.Sequence>
  <Series.Sequence durationInFrames={60}><Main /></Series.Sequence>
  <Series.Sequence durationInFrames={30}><Outro /></Series.Sequence>
</Series>
```

**Loop** - Repeat content:

```tsx
import { Loop } from 'remotion';

<Loop durationInFrames={30} times={3}>  {/* or times={Infinity} */}
  <Animation />
</Loop>
```

**Freeze** - Pause at specific frame:

```tsx
import { Freeze } from 'remotion';

<Freeze frame={10}>
  <MyComponent />  {/* Always renders as if at frame 10 */}
</Freeze>
```

## Layout & Media Components

```tsx
import { AbsoluteFill, Img, Video, Audio, OffthreadVideo, staticFile } from 'remotion';

// Full-screen container
<AbsoluteFill style={{ backgroundColor: 'white' }}>
  
  {/* Images - waits for load */}
  <Img src={staticFile('image.png')} />
  
  {/* Video - synchronized with timeline */}
  <Video src={staticFile('clip.mp4')} volume={0.5} />
  
  {/* OffthreadVideo - better performance for heavy videos */}
  <OffthreadVideo src={staticFile('large.mp4')} />
  
  {/* Audio */}
  <Audio src={staticFile('music.mp3')} volume={0.8} />
  
</AbsoluteFill>
```

**Video/Audio props:** `volume`, `playbackRate`, `muted`, `loop`, `startFrom`, `endAt`

**`playbackRate` limitations:**
- Must be a **constant value** (e.g., `playbackRate={2}`)
- Dynamic interpolation like `playbackRate={interpolate(frame, [0, 100], [1, 5])}` won't work correctly
- For variable speeds, extreme speeds (>4x), or guaranteed audio sync, pre-process with FFmpeg instead (see FFmpeg skill)

## Static Files

Place files in `public/` folder:

```tsx
import { staticFile } from 'remotion';

const videoSrc = staticFile('videos/intro.mp4');  // -> /public/videos/intro.mp4
```

## Input Props

Pass dynamic data to compositions:

```tsx
// In composition
const { title, color } = getInputProps();

// CLI: npx remotion render ... --props='{"title":"Hello","color":"red"}'
// Or from file: --props=./data.json
```

## Async Data Loading

```tsx
import { delayRender, continueRender } from 'remotion';

const MyComponent = () => {
  const [data, setData] = useState(null);
  const [handle] = useState(() => delayRender());  // Block render

  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(d => {
        setData(d);
        continueRender(handle);  // Unblock render
      });
  }, [handle]);

  if (!data) return null;
  return <div>{data.title}</div>;
};
```

## Color Interpolation

```tsx
import { interpolateColors, useCurrentFrame } from 'remotion';

const frame = useCurrentFrame();
const color = interpolateColors(frame, [0, 30], ['#ff0000', '#0000ff']);
```

## CLI Rendering

```bash
# Render to MP4
npx remotion render src/index.ts MyComposition out.mp4

# Common flags
--codec=h264|h265|vp8|vp9|gif|prores
--quality=80           # JPEG quality for frames
--scale=2              # 2x resolution
--frames=0-59          # Render specific frames
--props='{"key":"value"}'
--concurrency=4        # Parallel renders
```

## Programmatic Rendering

```tsx
import { bundle } from '@remotion/bundler';
import { renderMedia, selectComposition } from '@remotion/renderer';

const bundleLocation = await bundle({ entryPoint: './src/index.ts' });

const composition = await selectComposition({
  serveUrl: bundleLocation,
  id: 'MyComposition',
  inputProps: { title: 'Hello' },
});

await renderMedia({
  composition,
  serveUrl: bundleLocation,
  codec: 'h264',
  outputLocation: 'out.mp4',
  inputProps: { title: 'Hello' },
});
```

## Lambda Rendering

Deploy and render in AWS Lambda for parallel, serverless rendering:

```bash
# Setup
npx remotion lambda policies user
npx remotion lambda functions deploy

# Deploy site
npx remotion lambda sites create src/index.ts --site-name=mysite

# Render
npx remotion lambda render <serve-url> MyComposition out.mp4
```

```tsx
import { renderMediaOnLambda } from '@remotion/lambda';

const { bucketName, renderId } = await renderMediaOnLambda({
  region: 'us-east-1',
  functionName: 'remotion-render',
  serveUrl: 'https://...',
  composition: 'MyComposition',
  codec: 'h264',
  inputProps: { title: 'Hello' },
});
```

## Player Component (Embed in React Apps)

```tsx
import { Player } from '@remotion/player';

<Player
  component={MyComposition}
  durationInFrames={150}
  fps={30}
  compositionWidth={1920}
  compositionHeight={1080}
  style={{ width: '100%' }}
  controls
  loop
  autoPlay
  inputProps={{ title: 'Hello' }}
/>
```

## Scene Transitions

The toolkit includes a transitions library at `lib/transitions/` for scene-to-scene effects.

### Using TransitionSeries

```tsx
import { TransitionSeries, linearTiming } from '@remotion/transitions';
// Import custom transitions from lib (adjust path based on your project location)
import { glitch, lightLeak, clockWipe, checkerboard } from '../../../../lib/transitions';
// Or import from @remotion/transitions for official ones
import { slide, fade } from '@remotion/transitions/slide';

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <TitleSlide />
  </TransitionSeries.Sequence>
  <TransitionSeries.Transition
    presentation={glitch({ intensity: 0.8 })}
    timing={linearTiming({ durationInFrames: 30 })}
  />
  <TransitionSeries.Sequence durationInFrames={120}>
    <ContentSlide />
  </TransitionSeries.Sequence>
</TransitionSeries>
```

### Available Transitions

**Custom (lib/transitions/):**

| Transition | Options | Best For |
|------------|---------|----------|
| `glitch()` | `intensity`, `slices`, `rgbShift` | Tech demos, edgy reveals, cyberpunk |
| `rgbSplit()` | `direction`, `displacement` | Modern tech, energetic transitions |
| `zoomBlur()` | `direction`, `blurAmount` | CTAs, high-energy moments, impact |
| `lightLeak()` | `temperature`, `direction` | Celebrations, film aesthetic, warm moments |
| `clockWipe()` | `startAngle`, `direction`, `segments` | Time-related content, playful reveals |
| `pixelate()` | `maxBlockSize`, `gridSize`, `scanlines`, `glitchArtifacts`, `randomness` | Retro/gaming, digital transformations |
| `checkerboard()` | `gridSize`, `pattern`, `stagger`, `squareAnimation` | Playful reveals, structured transitions |

**Checkerboard patterns:** `sequential`, `random`, `diagonal`, `alternating`, `spiral`, `rows`, `columns`, `center-out`, `corners-in`

**Official Remotion (@remotion/transitions/):**

| Transition | Description |
|------------|-------------|
| `slide()` | Push scene from direction |
| `fade()` | Simple crossfade |
| `wipe()` | Edge reveal |
| `flip()` | 3D card flip |

### Transition Examples

```tsx
// Tech/cyberpunk feel
glitch({ intensity: 0.8, slices: 8, rgbShift: true })

// Warm celebration
lightLeak({ temperature: 'warm', direction: 'right' })

// High energy zoom
zoomBlur({ direction: 'in', blurAmount: 20 })

// Chromatic aberration
rgbSplit({ direction: 'diagonal', displacement: 30 })

// Clock sweep reveal
clockWipe({ direction: 'clockwise', startAngle: 0 })

// Retro pixelation
pixelate({ maxBlockSize: 50, glitchArtifacts: true })

// Checkerboard patterns
checkerboard({ pattern: 'diagonal', gridSize: 8 })
checkerboard({ pattern: 'spiral', gridSize: 10 })
checkerboard({ pattern: 'center-out', squareAnimation: 'scale' })
```

### Timing Functions

```tsx
import { linearTiming, springTiming } from '@remotion/transitions';

// Linear: constant speed
linearTiming({ durationInFrames: 30 })

// Spring: physics-based with bounce
springTiming({ config: { damping: 200 }, durationInFrames: 45 })
```

### Duration Guidelines

| Type | Frames | Notes |
|------|--------|-------|
| Quick cut | 15-20 | Fast, punchy |
| Standard | 30-45 | Most common |
| Dramatic | 50-60 | Slow reveals |
| Glitch effects | 20-30 | Should feel sudden |
| Light leak | 45-60 | Needs time to sweep |

### Preview Transitions

Run the showcase gallery to see all transitions:

```bash
cd showcase/transitions && npm run studio
```

## Best Practices

1. **Frame-based animations only** - Avoid CSS transitions/animations; they cause flickering
2. **Use fps from useVideoConfig()** - Make animations frame-rate independent
3. **Clamp interpolations** - Use `extrapolateRight: 'clamp'` to prevent runaway values
4. **Use OffthreadVideo** - Better performance than `<Video>` for complex compositions
5. **delayRender for async** - Always block rendering until data is ready
6. **staticFile for assets** - Reference files from public/ folder correctly

## Advanced API

For detailed API documentation on all hooks, components, renderer, Lambda, and Player APIs, see [reference.md](reference.md).

## License Note

Remotion has a special license. Companies may need to obtain a license for commercial use. Check https://remotion.dev/license

---

## Feedback & Contributions

If this skill is missing information or could be improved:

- **Missing a pattern?** Describe what you needed
- **Found an error?** Let me know what's wrong
- **Want to contribute?** I can help you:
  1. Update this skill with improvements
  2. Create a PR to github.com/digitalsamba/claude-code-video-toolkit

Just say "improve this skill" and I'll guide you through updating `.claude/skills/remotion/SKILL.md`.
