# New Brand Profile

Help me create a new brand profile for video production. I'll guide you through extracting brand elements and setting up the configuration.

## Your Tasks

1. **Gather Brand Information**
   Ask me the following questions using the AskUserQuestion tool:

   **Question 1 - Brand Basics:**
   - Brand name (e.g., "Acme Corp", "My Startup")
   - Brand description (one sentence describing the brand)
   - Website URL (optional, for color extraction)

   **Question 2 - Color Source:**
   - Extract colors from website URL?
   - Or provide colors manually?
   - Or use a color palette generator based on primary color?

   **Question 3 - If extracting from URL:**
   Use WebFetch to analyze the website and extract:
   - Primary brand color (main accent)
   - Secondary/light variant
   - Text colors (dark, medium, light)
   - Background colors
   - Any accent colors

   **Question 4 - If manual colors:**
   Ask for these colors (hex format):
   - Primary color (main brand color)
   - Primary light (hover/highlight variant)
   - Text dark (headings)
   - Text medium (body text)
   - Text light (captions/muted)
   - Background light
   - Background dark

   **Question 5 - Typography:**
   - Primary font family (e.g., "Inter", "Roboto", "system-ui")
   - Monospace font (for code, optional)

   **Question 6 - Logo:**
   Ask about logo assets:
   - Do you have a logo file to add?
   - Logo format (PNG, SVG preferred)
   - Do you need a light version for dark backgrounds?

   **Question 7 - Voice Settings (Optional):**
   - Do you want to configure ElevenLabs voice settings?
   - If yes: Voice ID, or use default?
   - Voice style preferences (stability, similarity boost)

2. **Create the Brand Profile**
   After gathering info:

   a) Create brand directory: `brands/{brand-name-lowercase}/`

   b) Create `brand.json` with the collected/extracted colors:
   ```json
   {
     "name": "Brand Name",
     "description": "Brand description",
     "version": "1.0.0",
     "website": "https://example.com",
     "colors": {
       "primary": "#...",
       "primaryLight": "#...",
       "textDark": "#...",
       "textMedium": "#...",
       "textLight": "#...",
       "bgLight": "#ffffff",
       "bgDark": "#...",
       "bgOverlay": "rgba(255, 255, 255, 0.95)",
       "divider": "#e2e8f0",
       "shadow": "rgba(0, 0, 0, 0.12)"
     },
     "fonts": {
       "primary": "..., sans-serif",
       "mono": "ui-monospace, SFMono-Regular, monospace"
     },
     "spacing": {
       "xs": 8,
       "sm": 16,
       "md": 24,
       "lg": 48,
       "xl": 80,
       "xxl": 120
     },
     "borderRadius": {
       "sm": 6,
       "md": 10,
       "lg": 16
     },
     "typography": {
       "h1": { "size": 88, "weight": 700 },
       "h2": { "size": 72, "weight": 700 },
       "h3": { "size": 48, "weight": 600 },
       "body": { "size": 44, "weight": 400 },
       "label": { "size": 34, "weight": 600, "letterSpacing": 2 }
     },
     "assets": {
       "logo": "assets/logo.png",
       "logoLight": "assets/logo-light.png"
     }
   }
   ```

   c) Create `voice.json`:
   ```json
   {
     "voiceId": "YOUR_VOICE_ID_HERE",
     "description": "Voice description",
     "settings": {
       "stability": 0.75,
       "similarityBoost": 0.9,
       "style": 0.2,
       "useSpeakerBoost": true
     },
     "model": "eleven_multilingual_v2"
   }
   ```

   d) Create `assets/` directory for logo files

3. **Color Extraction Tips**
   When extracting from a website:
   - Look for CSS custom properties (--primary-color, etc.)
   - Check the header/nav for primary brand color
   - Look at buttons and links for accent colors
   - Check text on different backgrounds for text color hierarchy
   - Note any gradient usage

4. **Provide Next Steps**
   Tell me:
   - Where to place logo files (`brands/{name}/assets/`)
   - How to use the brand in a project (`--brand {name}`)
   - How to configure ElevenLabs voice ID (environment variable or voice.json)
   - Link to docs: `docs/creating-brands.md`

## Output Location
`brands/{brand-name-lowercase}/`

## Output Files to Create
- `brands/{name}/brand.json` - Visual identity config
- `brands/{name}/voice.json` - Voice settings
- `brands/{name}/assets/` - Logo and asset directory

## Color Generation Helper
If the user only provides a primary color, generate a cohesive palette:
- Primary: user's color
- Primary Light: lighten by 15-20%
- Text Dark: dark neutral (#1e293b or similar)
- Text Medium: medium neutral (#475569 or similar)
- Text Light: light neutral (#94a3b8 or similar)
- Bg Light: #ffffff
- Bg Dark: dark version of primary or neutral dark
