# Record and replay a browser demo

The flow recorder saves browser interactions as editable JSON. Replaying the
same flow produces a new WebM recording plus click/target events for planning
camera moves. Existing one-off recording scripts still work.

Install dependencies in `playwright/`, then run:

```bash
cd playwright
npm ci
npx playwright install chromium
npm run flow -- record --url http://localhost:3000 --flow output/onboarding.json --mask '#email'
npm run flow -- replay --flow output/onboarding.json --output output/onboarding-v2
```

Recording opens a browser. Perform the demo and press Enter in the terminal to
save. `--duration 120` sets the maximum recording time. Replay runs headlessly;
`--headed` shows it. `--storage-state login.json` reuses an existing Playwright
authentication state. `--executable-path PATH` selects an installed Chromium
browser when the bundled browser is unavailable.

The recorder captures main-frame clicks, text fills, selects, Enter/Tab/Escape
keys and navigation waits. It prefers `data-testid`, then IDs, then a CSS path.
Review and edit selectors when the app's DOM changes. Popups, iframes, drag/drop,
file uploads and scroll-only demonstrations require manually authored steps or
the existing recording scripts.

Passwords and fields matching `--mask` are stored as environment references,
never their entered values. Provide those variables when replaying, for example
`FLOW_SECRET_1`. Masked elements are blurred in replay recordings. Missing
variables fail before any page navigation. Reuse sanitized demonstration data
for other fields.

```json
{
  "version": 1,
  "url": "http://localhost:3000",
  "viewport": {"width": 1920, "height": 1080},
  "fps": 30,
  "masks": ["#email"],
  "steps": [
    {"kind": "fill", "selector": "#email", "value": {"env": "DEMO_EMAIL"}},
    {"kind": "click", "selector": "[data-testid=continue]"},
    {"kind": "waitFor", "selector": "#dashboard"},
    {"kind": "wait", "milliseconds": 1000}
  ]
}
```

Other supported steps are `select`, `press`, `navigate`, and `waitForURL`.
Action failures exit nonzero and close the browser. All actions use Playwright
locators and its normal visibility/actionability checks.

Replay writes `<output>.webm` and `<output>.events.json`. Events include the
selector, target bounding box, milliseconds since replay started, and the
corresponding frame at the declared FPS. These timings describe replay actions;
they are not a promise of exact video-encoder frame alignment. Convert to MP4
when needed:

```bash
ffmpeg -i output/onboarding-v2.webm -c:v libx264 -r 30 -movflags +faststart output/onboarding-v2.mp4
npm run test:flows
npm run check:flows
```
