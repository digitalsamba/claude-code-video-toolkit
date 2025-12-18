# Creating Templates

Templates are reusable video structures. Each template defines:
- Slide components (title, overview, demos, summary)
- Default configuration schema
- Theme integration

## Template Structure

```
templates/
└── my-template/
    ├── package.json
    ├── tsconfig.json
    ├── src/
    │   ├── index.ts           # Entry point
    │   ├── Root.tsx           # Composition registration
    │   ├── config/
    │   │   ├── types.ts       # Config type definitions
    │   │   ├── theme.ts       # Theme system
    │   │   └── defaults.ts    # Default values
    │   └── components/
    │       ├── TitleSlide.tsx
    │       ├── ContentSlide.tsx
    │       └── ...
    └── public/
        └── .gitkeep
```

## Starting a New Template

1. **Copy an existing template**
   ```bash
   cp -r templates/sprint-review templates/my-template
   cd templates/my-template
   ```

2. **Update package.json**
   ```json
   {
     "name": "my-template",
     "description": "Description of your template"
   }
   ```

3. **Define your config schema**
   Edit `src/config/types.ts`:
   ```typescript
   export interface MyTemplateConfig {
     title: string;
     subtitle?: string;
     sections: Section[];
     duration: number;
   }
   ```

4. **Create components**
   Build React components for each slide type.

5. **Register compositions**
   In `Root.tsx`:
   ```tsx
   <Composition
     id="MyTemplate"
     component={MyTemplateVideo}
     durationInFrames={config.duration * 30}
     fps={30}
     width={1920}
     height={1080}
   />
   ```

## Template Best Practices

### Use the Theme System

Always pull colors and fonts from the theme:

```tsx
import { useTheme } from '../config/theme';

const MySlide: React.FC = () => {
  const theme = useTheme();

  return (
    <div style={{ backgroundColor: theme.colors.bgLight }}>
      <h1 style={{ color: theme.colors.textDark }}>
        Title
      </h1>
    </div>
  );
};
```

### Frame-Based Animations

Use Remotion's interpolate, not CSS transitions:

```tsx
import { useCurrentFrame, interpolate } from 'remotion';

const AnimatedElement: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: 'clamp',
  });

  return <div style={{ opacity }}>Content</div>;
};
```

### Configurable Duration

Let users control timing via config:

```tsx
const demoSection = sprintConfig.demos.map((demo, i) => (
  <Series.Sequence
    key={i}
    durationInFrames={demo.durationSeconds * 30}
  >
    <DemoSlide demo={demo} />
  </Series.Sequence>
));
```

### Scene Transitions

Use the transitions library for scene-to-scene effects:

```tsx
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { glitch, lightLeak, checkerboard } from '../../../../lib/transitions';

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

Available transitions: `glitch`, `rgbSplit`, `zoomBlur`, `lightLeak`, `clockWipe`, `pixelate`, `checkerboard`, plus official Remotion transitions (`slide`, `fade`, `wipe`, `flip`).

See [lib/transitions/README.md](../lib/transitions/README.md) for full options.

### Asset References

Use `staticFile()` for assets:

```tsx
import { staticFile, Video } from 'remotion';

<Video src={staticFile('demos/my-demo.mp4')} />
```

## Publishing Templates

To make a template available to others:

1. Ensure it works standalone (`npm install && npm run studio`)
2. Document the config schema
3. Add to `_internal/toolkit-registry.json`:
   ```json
   "templates": {
     "my-template": {
       "path": "templates/my-template/",
       "description": "What this template is for",
       "status": "beta",
       "created": "2025-12-10",
       "updated": "2025-12-10"
     }
   }
   ```

## Template Ideas

- **Product Demo**: Problem → Solution → Demo → CTA flow
- **Tutorial**: Chapter-based with progress indicator
- **Changelog**: Version header with feature list
- **Comparison**: Before/after split screen
