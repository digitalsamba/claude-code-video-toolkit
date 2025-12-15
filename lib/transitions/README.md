# Transitions Library

Scene transition effects for Remotion videos. Combines official `@remotion/transitions` with custom presentations for a comprehensive transition toolkit.

## Installation

The transitions package is installed in each template. If setting up manually:

```bash
npm install @remotion/transitions @remotion/paths @remotion/shapes
```

## Usage

Transitions work with Remotion's `TransitionSeries` component:

```tsx
import { TransitionSeries, linearTiming, springTiming } from '@remotion/transitions';
import { glitch, rgbSplit, lightLeak } from '../../../lib/transitions';

export const MyVideo = () => {
  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={90}>
        <TitleScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={glitch({ intensity: 0.8 })}
        timing={linearTiming({ durationInFrames: 20 })}
      />

      <TransitionSeries.Sequence durationInFrames={120}>
        <ContentScene />
      </TransitionSeries.Sequence>

      <TransitionSeries.Transition
        presentation={lightLeak({ temperature: 'warm' })}
        timing={springTiming({ config: { damping: 200 } })}
      />

      <TransitionSeries.Sequence durationInFrames={90}>
        <EndScene />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
```

## Available Transitions

### Custom Transitions (this library)

| Transition | Description | Best For |
|------------|-------------|----------|
| `glitch()` | Digital distortion with slice displacement and RGB separation | Tech demos, cyberpunk, edgy reveals |
| `rgbSplit()` | Chromatic aberration with channel separation | Modern tech, energetic transitions |
| `zoomBlur()` | Radial motion blur with scale | CTAs, reveals, high-energy moments |
| `lightLeak()` | Cinematic lens flare and overexposure | Emotional moments, celebrations, film aesthetic |
| `clockWipe()` | Radial wipe like clock hands | Time-related content, playful reveals |
| `pixelate()` | Digital mosaic dissolution | Retro/gaming, digital transformations |

### Official Transitions (re-exported)

| Transition | Description |
|------------|-------------|
| `slide()` | Scene slides in from a direction |
| `fade()` | Simple crossfade |
| `wipe()` | Edge wipe reveal |
| `flip()` | 3D card flip |

## Transition Options

### glitch(options?)

```tsx
glitch({
  intensity: 0.8,      // Effect strength (0-1). Default: 0.8
  slices: 8,           // Horizontal slice count. Default: 8
  rgbShift: true,      // RGB channel separation. Default: true
  scanLines: true,     // CRT scan line overlay. Default: true
})
```

### rgbSplit(options?)

```tsx
rgbSplit({
  direction: 'horizontal',  // 'horizontal' | 'vertical' | 'diagonal'. Default: 'horizontal'
  displacement: 30,         // Max pixel offset. Default: 30
  channelBlur: true,        // Motion blur on channels. Default: true
})
```

### zoomBlur(options?)

```tsx
zoomBlur({
  direction: 'in',     // 'in' (toward viewer) | 'out' (away). Default: 'in'
  blurAmount: 20,      // Max blur pixels. Default: 20
  scaleAmount: 1.15,   // Scale multiplier. Default: 1.15
  origin: 'center',    // 'center' | 'top' | 'bottom' | 'left' | 'right'. Default: 'center'
})
```

### lightLeak(options?)

```tsx
lightLeak({
  temperature: 'warm',    // 'warm' | 'cool' | 'rainbow'. Default: 'warm'
  direction: 'right',     // 'left' | 'right' | 'top' | 'bottom' | 'center'. Default: 'right'
  intensity: 0.8,         // Overexposure strength (0-1). Default: 0.8
  flareArtifacts: true,   // Lens flare spots. Default: true
})
```

### clockWipe(options?)

```tsx
clockWipe({
  startAngle: 0,              // Starting angle in degrees. Default: 0 (12 o'clock)
  direction: 'clockwise',     // 'clockwise' | 'counterclockwise'. Default: 'clockwise'
  segments: 1,                // Number of wipe arms. Default: 1
  softEdge: true,             // Soft glow on edge. Default: true
})
```

### pixelate(options?)

```tsx
pixelate({
  maxBlockSize: 40,    // Max pixel block size. Default: 40
  posterize: true,     // Reduce color depth. Default: true
  pattern: 'uniform',  // 'uniform' | 'random'. Default: 'uniform'
})
```

## Timing Functions

### linearTiming

Constant speed transition:

```tsx
linearTiming({ durationInFrames: 30 })  // 1 second at 30fps
```

### springTiming

Physics-based with bounce:

```tsx
springTiming({
  config: {
    damping: 200,      // Higher = less bounce
    stiffness: 100,    // Higher = snappier
    mass: 1,           // Higher = slower
  },
  durationInFrames: 45,  // Optional max duration
})
```

## Choosing Transitions

| Video Type | Recommended Transitions |
|------------|------------------------|
| **Tech/Product Demo** | `glitch`, `rgbSplit`, `slide` |
| **Corporate/Professional** | `fade`, `wipe`, `zoomBlur` |
| **Celebration/Launch** | `lightLeak`, `zoomBlur` |
| **Retro/Gaming** | `pixelate`, `glitch` |
| **Cinematic** | `lightLeak`, `fade`, `wipe` |
| **Playful/Creative** | `clockWipe`, `flip` |
| **High Energy** | `zoomBlur`, `rgbSplit`, `glitch` |

## Transition Duration Guidelines

| Transition Type | Recommended Duration | Notes |
|-----------------|---------------------|-------|
| Quick cut | 10-15 frames | Fast, punchy |
| Standard | 20-30 frames | Most common |
| Dramatic | 40-60 frames | Slow reveals |
| Glitch effects | 15-25 frames | Should feel sudden |
| Light leak | 30-45 frames | Needs time to sweep |

## Combining with Audio

Add sound effects to transitions:

```tsx
import { Audio, Sequence } from 'remotion';

// Play whoosh sound during transition
<Sequence from={transitionStartFrame} durationInFrames={30}>
  <Audio src={staticFile('sfx/whoosh.mp3')} volume={0.5} />
</Sequence>
```

## Transition Gallery

A visual showcase component is included for previewing all transitions:

```tsx
import { TransitionGallery, transitionGalleryConfig } from '../../../lib/transitions';

// Register in Root.tsx
<Composition
  id={transitionGalleryConfig.id}
  component={TransitionGallery}
  durationInFrames={transitionGalleryConfig.durationInFrames}
  fps={transitionGalleryConfig.fps}
  width={transitionGalleryConfig.width}
  height={transitionGalleryConfig.height}
/>
```

Then run `npm run studio` and select "TransitionGallery" to preview all transitions.

### Single Transition Preview

For interactive previews (e.g., with `@remotion/player`):

```tsx
import { SingleTransitionPreview, transitionMap } from '../../../lib/transitions';

// Preview a specific transition
<SingleTransitionPreview transitionName="glitch" />

// Access transition config programmatically
const { presentation, duration } = transitionMap.lightLeak;
```

## Technical Notes

1. **TransitionSeries vs Series**: `TransitionSeries` allows overlapping scenes during transitions. Regular `Series` does not.

2. **Duration calculation**: Total video duration = sum of sequence durations - sum of transition durations (because scenes overlap).

3. **Performance**: Complex transitions (glitch, pixelate) use SVG filters which may impact preview performance. Final renders are unaffected.

4. **Browser compatibility**: All transitions use standard CSS/SVG features. Tested in Chrome (Remotion's render target).
