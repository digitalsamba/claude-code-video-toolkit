# Toolkit Changelog

All notable changes to claude-code-video-toolkit.

> Releases are automated via GitHub Actions. See `.github/workflows/release.yml`.

---

## Unreleased

### Added

- Sentence-aware transcript highlights with repetition and internal-pause removal.
- OpenCV subject/face tracking and smoothed reframing of existing footage.
- Speech-driven music ducking and timestamped sound effect placement.
- Diarized speaker import, stable TTS voice mapping, timed dubbing and background stems.
- Locked character references with FLUX still and dependent LTX motion plans.
- Reusable CSV/JSON chart videos, data-derived narration and provenance.
- Locked source checks for script and displayed project claims, with explicit review warnings.
- Chroma key, supplied-alpha and optional recurrent ONNX human matting with tracked graphics.
- An eleven-task offline capability example, media assertions and checkpoint resume verification.

- Runtime-validated project manifests and compatibility-preserving migration.
- Checkpointed build graphs, content-aware selective regeneration, cost forecasts,
  estimate caps, and named draft/final execution profiles.
- Local visual storyboard review with notes and content-bound approvals.
- Immutable searchable asset versions and checksum-verified restore.
- Named narration cues for cursor/camera timing and a runnable synthetic demo.
- Adaptive storyboard compositions for landscape, portrait and square output.
- Reusable browser action recording/replay with secret references and target events.
- Language/campaign variants, audio retiming plans and SRT/WebVTT export.
- Validated local provider/template contracts and an offline template scaffold.
- Windows/Linux workflow verification with actual video and browser checks.

### Fixed

- **product-demo template text is readable on a phone and passes WCAG AA**: every slide's text is at least 28px at 1080p (it was down to 12px), the problem badges use dark text on the coral (2.8:1 → 6.5:1), and the CTA tagline uses `textMedium` instead of `textLight` (2.6:1 → 5.4:1). The demo window is marked `data-qc="product"`. `qc.py --layout` on the template now reports no text findings.

---

## 2026-10-28 (v0.23.0)

QC learns to look at the picture and the text, not just the sound: `qc.py --render` finds frozen
stretches, black gaps and an empty first frame, and `qc.py --layout` measures every piece of text for
size, overflow and contrast. Alongside: a motion-craft reference for Remotion scenes, clone prosody
tools for cloned voices that come out flat, and the `ds-release-notes` template.

### Added

- **`qc.py --layout` measures the text in the video**: it bundles a Remotion project, opens it in Playwright and seeks a frame a second with Remotion's own hooks (no render), then reports text off the canvas or clipped by its box, text that never reaches 28px at 1080p, and contrast under WCAG AA, grouped by colour pair with a passing colour suggested. Text inside `data-qc="product"` (now set on `Stage`'s browser window) is the product's own UI and is listed apart, as info. On the product-demo template it finds white on the coral badges at 2.78:1 and the grey CTA tagline at 2.58:1. `Stage`'s "AI-generated" note goes from 2.9:1 to 5.3:1. Adapted from HyperFrames' `check` (HeyGen, Apache-2.0). (#136)
- **`qc.py --render` checks the picture**, not just the loudness. It flags stretches over 3s where nothing on screen changes (`--max-still`), black gaps of 0.5s or more mid-video, and an empty first frame (a warning on vertical and square renders, whose first frame is the Shorts/Reels cover). The still check compares 20px tiles of a 320px-wide copy at 4fps, so a moving cursor or a talking-head PiP counts as change, where ffmpeg's `freezedetect` misses both. It takes about 8s on a 4-minute render; `--skip-frames` turns it off. A render with no audio track, or a silent one, now says which instead of "could not measure loudness". Thresholds follow HyperFrames' `check` craft warnings (HeyGen, Apache-2.0). (#135)
- **Clone prosody tools**: `tools/voice_prosody.py` measures pitch spread (F0 semitone sd, p5-p95) of any audio and writes a livelier clone reference by expanding the pitch contour around the median and slowing it (Praat PSOLA, `uv sync --extra prosody`). `tools/pick_takes.py` chooses the best take per phrase from several voiceover runs by pace, then pitch spread, stretching only a straggler. Why: a Qwen3 clone copies its reference clip's pace and intonation, and the raw Lugh clip is flat and fast, so every line was time-stretched 0.8x and still monotone. The expanded clip (`assets/voices/lugh-ref-expressive.wav`) gives clones at 3-4 st and 145-165 wpm with no stretch. CLAUDE.md "Clone Prosody" has the workflow. (#133)
- **`ds-release-notes` template**: the synthetic release-notes video as a reusable Remotion project (room and account-center kits, Stage, cards, SFX, the sentence-level voice pipeline), shipped as the finished Digital Samba release 650 build so it renders as is. Its release number and month live in `src/config.ts`, and `/video` offers it. (#133, #137)
- **Motion craft reference** (`.claude/skills/remotion/motion-craft.md`): durations and staggers in frames at 30fps, choosing an ease by what the move does, springs named by damping ratio (with Remotion's default `spring()` worked out as ζ = 0.5, i.e. playful), read time for on-screen text, a beat grid that doesn't drift, transitions with a cause, and a list of defaults that make a video look generated, with the toolkit components that produce them. `/design` and `/scene-review` now read it alongside frontend-design. Adapted from HyperFrames (HeyGen, Apache-2.0). (#134)

---

## 2026-10-27 (v0.22.0)

Synthetic UI demos become a toolkit feature: shared `Stage` and `Cursor` components and a
`synthetic-ui-demo` skill. This release also fixes `glitch` and `checkerboard` transitions that
distorted scenes even when idle, and syncs the Remotion skills to 4.0.539.

### Added

- **Synthetic UI demos as a toolkit feature**: shared `Stage` and `Cursor` components in `lib/components` (a browser window with camera zoom/pan keyframes and a scripted cursor), and a `synthetic-ui-demo` skill. The skill covers rebuilding an app's screens from its source, a frame-based timeline anchored to the voice lines, measuring click targets from stills, and labelling AI-generated people. `examples/ds-release-649` and `examples/ds-free-schedule-demo` are the reference builds. (#122)

### Fixed

- **Seams across whole scenes with the `glitch` transition**: `TransitionSeries` keeps a scene inside its presentation for the entire sequence, and `glitch` sliced the scene into 8 strips even when idle. The strips were also mis-sized (each was 0.5% too tall, so content drifted a few pixels per strip), which clipped text and image rows every ~135 px for the whole scene before the transition and after it. The idle scene now passes through untouched, and the active slices use exact `clip-path` bands. `checkerboard` had the same idle problem on the entering side (64 clipped copies of the scene after the transition) and now shows the scene whole. `lib/transitions/README.md` notes that presentations must pass idle scenes through, and that `<Audio>` belongs outside the `TransitionSeries` because transitions render a scene several times. (#125)

### Changed

- **Remotion skills synced to upstream 4.0.539** (`remotion-dev/skills` `30d4553`). The markup and interactivity guides are reorganised, with a new Arabic text page and longer colour correction, connected compositions and video editing pages. The toolkit's templates stay on `<OffthreadVideo>`. (#126)
- **README front page** shows the two synthetic demo videos first under "See What It Makes". (#123, #124)

### Notes

- CommentKeyword (Instagram comment-to-DM for a video's "Comment KEYWORD" call to action) is listed under the README's Community add-ons (#128, #129).

---

## 2026-10-21 (v0.21.1)

Two showcase examples made without screen recording (product UIs rebuilt from source and driven by a
scripted cursor and camera), two TTS/SFX fixes, and the Remotion skills sync to 4.0.536.

### Added

- **Two synthetic demo examples**: `examples/ds-release-649` (an 80s release-notes video) and `examples/ds-free-schedule-demo` (a 26s product demo). Nothing is screen-recorded. The product screens are rebuilt from their source code and driven by a scripted cursor and camera, narrated by Lugh. (#119)

### Fixed

- **`sfx.py --preset pop` works.** Its 0.3s duration was under ElevenLabs' 0.5s minimum, so every call was rejected. (#118)
- **`--min-tempo` for the pace clamp** (`voiceover.py`, `qwen3_tts.py`). The 0.85x stretch floor can now go as low as 0.7x, and a take that is still over `--max-wpm` after the clamp is reported as `floored`, with a hint. Before this, voices that rush hard needed a manual atempo afterwards. (#118)

### Changed

- **Remotion skills synced to upstream 4.0.536** (`remotion-dev/skills` `1441a11`). New pages cover colour correction (`effects` on `@remotion/media` `<Video>`) and timeline structure, and the transitions page is rewritten. The toolkit's templates stay on `<OffthreadVideo>`. (#117)

---

## 2026-10-12 (v0.21.0)

Five new production steps: pre-render QC with loudness (`tools/qc.py`), `/thumbnail`,
one-command Modal setup (`tools/modal_setup.py`), burned captions for the Remotion
templates (`tools/captions.py` + `<Captions>`) and social cut-downs (`tools/cutdown.py`).

Also a model refresh, checked against each upstream on 2026-10-02. The entries below move
several defaults to newer releases and add newer models as opt-ins (a deploy-time
variable for the Modal apps, `--model 4.5` for Ideogram); models not listed keep their
current version. Endpoint URLs are unchanged, and RunPod images
are untouched.

### Changed
- **Remotion skills synced to upstream v4.0.535** (via 4.0.533; #105, #113).
- **concept-explainer-short captions pick Whisper `turbo` on CUDA.** `whisperModel: "auto"`
  (the new template default) uses turbo with a GPU and `base` on CPU; an explicit model
  name still wins. Needs `openai-whisper>=20240930`.
- **Release notes skip merge commits.** `release.yml` builds the GitHub Release body
  from `git log --no-merges`, so "Merge pull request" and branch-sync merges no longer
  appear as bullets; the commits they bring in still do.
- **`modal-flux2`: FLUX.2-small-decoder** (Apache-2.0, ~1.4x faster decode) and the
  `diffusers==0.41.0` release in place of the git commit pin. klein-4B stays the
  default. Modal FLUX edits now default to 4 steps / guidance 1.0 (was 50 / 4.0), as
  generation already did, since klein is step-distilled; `--steps`/`--guidance`
  still override.
- **`modal-image-edit`: `diffusers==0.41.0`** in place of unpinned git main, with
  transformers 5.17. Both Modal image apps now use the diffusers 0.41.0 release on
  torch 2.6.0. Qwen-Image-Edit-2511 stays the default, at 8 steps.
- **`image_edit.py` steps on Modal:** the client still sends 8 steps by default and
  marks them `steps_is_client_default`, so the app runs its deployed model's default
  (8 on 2511, 40 on the Qwen-Image-2.1 opt-in). An app deployed from an older version
  ignores the flag and runs the 8 it is sent; an explicit `--steps` is always honoured.
  The image cap also follows the deployed model, and the tool prints the `Model:` and
  `Steps:` that ran.
- **ACE-Step 1.5 → 1.5 XL Turbo (4B)** (`modal-music-gen`). ACE-Step-1.5 pinned to
  v0.1.8, weights baked into `checkpoints/` (v0.1.8 no longer reads the HF cache),
  torch 2.10, `shift=3.0` default. Still fits the A10G.
- **ElevenLabs:** `scribe_v2` is the STT default in `redub` and `align_captions`.
  `eleven_v4` added to `voiceover` and `redub` via the Text to Dialogue API
  (stability + similarity only, chunked above 2,000 characters); `redub --sync` now
  sends stability + similarity on v4 too. Deprecated ids are still accepted, each
  with a one-line warning: `eleven_turbo_v2_5` is sent as `eleven_flash_v2_5`,
  `scribe_v1_experimental` as `scribe_v2`, and `scribe_v1` as is. Default stays
  `eleven_multilingual_v2`. `elevenlabs>=2.21.0`.
- **Captions:** `concept-explainer-short` `whisperModel` now defaults to `"auto"` —
  Whisper `turbo` on CUDA, `base` on CPU. `openai-whisper>=20240930`.
- **Cost estimates** use Modal's current rates (A100-80GB $2.50/h, was $4.68; H100
  $3.95/h; L40S added) and the GPU each Modal app actually runs on.

### Added
- **`tools/qc.py` — pre-render QC + loudness.** One command run before `npm run render`
  (or `build.py`): missing media referenced by the config, narration longer than its scene
  (cut off) or >3s shorter (dead air), wpm pacing per scene, and per-scene loudness
  consistency. `--render out/video.mp4` checks the finished video against YouTube's
  -14 LUFS / -1 dBTP; nothing in the templates normalises loudness, and the first project
  checked had shipped at -22.5 LUFS. `--fix` normalises loudness only: scene VO to
  -16 LUFS in place (originals in `.qc-backup/`), the render to a new `<name>.loudnorm.mp4`
  with the video stream copied. Reuses `sync_timing.py`'s config parsing and `pacing.py`'s
  labels; reads Remotion projects and concept-explainer-short `scenes.json`.
- **`/thumbnail` command.** Generates a 1280×720 YouTube thumbnail for a project.
  Claude reads the project, script and brand, proposes a 1–3 word hook phrase, writes
  an Ideogram 4 JSON caption (text on the left, nothing under YouTube's duration badge),
  renders it with `ideogram4.py --resolution 1280x720`, reads the image back to check the
  spelling, and records it as `publish.thumbnail` so `/publish` uploads it. ~$0.10 per
  image.
- **`tools/modal_setup.py`: one-command Modal setup.** `modal_setup.py qwen3_tts flux2`
  (or `--all`, which leaves out `ltx2`) deploys each app in turn and writes its endpoint
  URL into `.env`. It runs SoulX's `populate_weights` first, and reads every URL from
  Modal's API, not the deploy log, which wraps and truncates long URLs. `--list` shows
  what's deployed and whether `.env` agrees. `--sync-env` fills `.env` from apps you
  already deployed, without redeploying. `--recreate` covers variant switches. Failed
  deploys print a hint (payment method, rate limit, gated Hugging Face access).
  `qwen3_tts.py --setup --cloud modal` now uses it.
- **`HF_SECRET_NAME` deploy-time override** for the gated `ltx2` and `flux2` (klein-9B)
  apps. They still default to the `huggingface-token` secret. Modal's Hugging Face template
  names the secret `huggingface-secret`, which made first deploys fail. `modal_setup.py`
  now checks that the secret exists before deploying, and picks up `huggingface-secret`
  automatically; `--hf-secret NAME` overrides it.
- **Burned captions for the Remotion templates.** `tools/captions.py` transcribes narration
  (ElevenLabs Scribe or local whisper) and writes a word-level `<name>.captions.json` beside each
  audio file, aligned to the scene's `.txt` script so names are spelled as written. A new shared
  `<Captions>` component (`lib/components`) renders it with karaoke highlighting; sprint-review,
  sprint-review-v2 and product-demo show captions automatically once the JSON exists
  (`captions: { enabled: false }` turns them off). No new npm dependency.
- **Social cut-downs** (`tools/cutdown.py`): reframe a finished 16:9 video to 9:16, 1:1
  or 4:5 for Shorts, Reels, TikTok and feeds. Three modes: `blur` (whole frame over a
  blurred copy, the default), `crop` (fill the frame, with `--focus` choosing which part
  to keep) and `pad` (solid colour, or a brand's `bgDark`). `--start/--end` keeps one
  range; repeated `--segment` joins several into a highlight reel, with a 30ms audio
  fade at each join and an optional crop focus per segment. Several `--format`s render
  in one call. `--platform` warns past the 3-minute Shorts/Reels limit, and fitted
  modes warn when the source shrinks enough to make burned captions small. Pure ffmpeg,
  works on any MP4, no project needed.
- **Opt-in non-commercial models, chosen at deploy time.** Licences differ, so the
  Apache-2.0 models stay the default and the newer ones need an explicit switch:
  - `FLUX2_MODEL=klein-9b` deploys FLUX.2-klein-9B (gated, FLUX Non-Commercial
    licence, outputs need review) on an L40S.
  - `IMAGE_EDIT_MODEL=qwen-image-2.1` deploys Qwen-Image-2.1 (`QwenImage21Pipeline`,
    up to 10 input images, RGBA; Qwen Research licence, non-commercial).
  The endpoint URL is unchanged either way, and the tools print the `Model:` that ran.
- **SoulX-FlashHead Lite** (`soulx.py --model lite`): a drafts-only alternative to
  SadTalker; long-take drift unmeasured (tested on ~2.6s clips). ~8x faster per chunk
  than Pro at ~$0.0004 per second of output, no torch.compile. Sizes snap to a 32px
  grid. Lite is unloaded before any Pro render.
- **Ideogram 4.5 as an opt-in** (`ideogram4.py --model 4.5`), posted as multipart to
  `/v2/image/generate/ideogram-4-5`. `--speed` maps to its quality tiers: TURBO,
  DEFAULT and QUALITY → low $0.03, medium $0.06 and high $0.10, the same prices as on
  Ideogram 4; FLASH, which has no Ideogram 4 price, → low. `--resolution` is checked against 4.5's size rules; the six
  Ideogram 4 presets that break them (720x1280, 1280x720, 720x1440, 1440x720,
  1296x3168, 3168x1296) exit 2 with a suggested size. Ideogram 4 stays the default and
  sends the same requests as before; its cost log now shows QUALITY at $0.10 (was
  $0.09) and warns that FLASH currently returns HTTP 400. Run live, 4.5 renders JSON
  captions near-perfectly but adds garbled filler text to plain prompts, so use it with
  `--json`. 4.0 stays the default because 1280x720 (YouTube, `/thumbnail`) is off 4.5's grid.
- **LTX-2.5 as a deploy-time opt-in** (`LTX2_VERSION=2.5` on `modal-ltx2`). LTX-2.3
  stays the default and its image is unchanged, so redeploying 2.3 rebuilds nothing.
  2.5 runs LTX-2 v1.4.2 on torch 2.13 / CUDA 13.0 with the gated `Lightricks/LTX-2.5`
  weights (Gemma 4 text encoder bundled), pinned to one HF revision. LTX-2.x Community
  licence: free under $10M annual revenue, commercial use included; watermark and
  provenance features must not be removed. Same endpoint and request format; responses
  gain a `model` field, which `ltx2.py` prints as `Model:`. The `crt-terminal` LoRA was
  trained on 2.3 and doesn't keep its look on 2.5. Benchmarked against 2.3 at the
  default size, for image-to-video and with the LoRA (table in `docs/ltx2.md`): 2.5 is
  about 25-45% slower per clip, so 2.3 stays the default.

### Fixed
- **`modal-flux2` and `modal-image-edit` crash-looped on fresh builds** (seen as an
  endless cold start): open-ended `transformers` now needs torch 2.6, so both images
  move to torch 2.6.0 (#106). The RunPod flux2 and qwen-edit Dockerfiles get the same
  bump, and their pip specifiers are now quoted, since the shell had been reading `>=`
  as a redirect and dropping every version floor. RunPod images are not rebuilt yet.
- **Modal setup docs:** every GPU app needs a payment method on a new workspace, not
  just the A100s (A10G deploys fail too). Also documented: Modal's Hugging Face secret
  template defaults to the name `huggingface-secret`, `ltx2` needs access to Google's
  gated Gemma 3 repo, deploy output wraps long endpoint URLs across lines, and
  switching an opt-in should use `modal deploy --strategy recreate`.

### Notes
- Upload-Post (multi-platform publishing) is listed under the README's Community
  add-ons (#103, #107).
- `FLUX2_MODEL=klein-9b` needs the `huggingface-token` Modal secret too (gated
  weights), like `modal-ltx2`.
- `qwen-tts` pinned to 0.1.1 in `modal-qwen3-tts`; no newer Qwen TTS model exists.
  SoulX-FlashHead, Real-ESRGAN and ProPainter are already on their latest releases.

---

## 2026-10-02 (v0.20.2)

Patch release. Fixes two first-deploy failures on Modal (LTX-2, SoulX) and the
Windows cp1252 crashes, moves Remotion to 4.0.529, and rewrites the contribution
policy for hosted services.

### Added
- **`concept-explainer-short`: `voice.brand`** in `config.json` now drives the
  narrator from a brand profile. `gen_vo.py` never passed `--brand`, so a brand
  could set the palette but not the voice. Empty or absent adds no flag. Picked
  up from #84. (#88)
- **Community add-on: arkiv** — natural-language search over your own footage
  archive, listed in the README. (#100)

### Changed
- **Remotion 4.0.518 → 4.0.529** across all templates, examples, showcase and
  the render-baseline harness, via the first grouped Dependabot bump. The
  baseline A/B rendered identical on every frame, and all nine Remotion
  projects smoke-render on the new version. (#97)
- **Official Remotion skills synced to 4.0.529**, so the skills and the pinned
  version match again. New upstream material: connected compositions, motion
  blur, editable SVG paths. (#83, #92)
- **Contribution policy for hosted services** rewritten. Integrations that need
  a hosted third-party service default to Community add-ons; the providers
  in-tree are there because a maintainer uses and can test each one, and the
  route in is an issue before any code. The old wording contradicted the tree
  and invited the PRs it then declined. (#99)
- **`/contribute`** asks what actually differs before copying a template, and
  routes colors, fonts, voice and timing to `/brand`. Two PRs had arrived as
  near-identical template forks because the command told them to. (#89)
- The Remotion skills sync now updates the registry's `upstreamVersion`, which
  had been left at 4.0.518 through two syncs.

### Fixed
- **Windows: tools and Modal setup died on cp1252.** Python picks the locale
  codec for redirected output, so printing `→` or `⚠` raised
  `UnicodeEncodeError`, and `modal deploy` aborted mid-build on its own
  progress output. New `tools/win_encoding.py` guards the affected tools, and
  `/setup` documents `PYTHONIOENCODING=utf-8`. Thanks @randomdill1 (#91).
- **`qwen3_tts.py --setup --cloud modal`** reported "could not parse endpoint
  URL" after a successful deploy, because Modal wraps long URLs across lines.
  (#91, hardened in #98 for URLs on a non-final branch of the output tree.)
- **LTX-2 Modal app** (`docker/modal-ltx2/app.py`) cloned upstream unpinned, so
  any deploy after upstream's August 2026 releases crashed on import with
  `module 'torch.compiler' has no attribute 'nested_compile_region'` (upstream
  now needs torch 2.13; the app installs 2.7). The clone is pinned to the commit
  the pipeline code was written against. Existing deployments are unaffected
  until they are redeployed. Thanks @nevesen (#94).
- **SoulX-FlashHead** deployed before `populate_weights` ran crash-looped with a
  bare `FileNotFoundError` while the client hung until its timeout. The app now
  starts anyway and every request returns a readable error naming the command to
  run; a warm container picks the weights up without a redeploy. `/setup` now
  includes SoulX (populate, then deploy) and `verify_setup.py` checks both the
  endpoint and the weights volume. Thanks @nevesen (#95).
- **SoulX GPU tier** is a deploy-time setting (`SOULX_GPU=L40S modal deploy …`)
  instead of a hard-coded A10G, which OOMs at 1280x720 (#95).
- **Modal timeouts and 5xx errors** now point at `modal app logs`, since a
  container crashing on startup looks the same as a slow cold start from the
  client (#94, #95).

---

## 2026-09-08 (v0.20.1)

Patch release. v0.20.0 shipped with two tools that aborted with a `NameError`
before doing any work; both are fixed here.

### Fixed
- **Codex and Kiro migrations** (`scripts/migrate_to_codex.py`,
  `scripts/migrate_to_kiro.py`) — `write_text()` was dropped by the
  `_migrate_common` extraction without being moved to the shared module, so
  `--force` aborted with `NameError: name 'write_text' is not defined` on the
  first real write. `--dry-run` returned before that path, which is why it went
  unnoticed. Restored in `_migrate_common`, and the two orphaned function bodies
  the same refactor left stranded are gone. Thanks @Anai-Guo (#85).
- **`tools/dewatermark.py --setup`** — `setup_runpod()` called
  `get_runpod_config()`, which is defined nowhere in the repo, so the documented
  one-time setup crashed immediately. Now uses the `load_dotenv()` +
  `os.getenv("RUNPOD_API_KEY")` pattern `tools/upscale.py` already uses (#86).
- **`_migrate_common.load_mapping`** returned raw JSON instead of the normalized
  shape, so any caller would `KeyError` on `mapping["skip_commands"]`. It was
  unreachable only because both scripts shadowed it with a local copy. The real
  version is now shared and the duplicates (plus a duplicate `find_repo_root`
  and an unused `yaml_quote` import) are gone — verified byte-identical
  migration output before and after (#86).

### Added
- **`Lint Python` CI workflow** — gates on undefined names
  (`ruff --select F821,F811,F822,F823`) across `scripts/` and `tools/`.
  Deliberately narrow: only rules where a hit is a real defect, not style. Run
  against the v0.20.0 tree it reports 26 errors, including both `write_text`
  sites. There was previously no Python lint in CI (#86).

---

## 2026-08-31 (v0.20.0)

### Added
- **SoulX-FlashHead talking head** (`tools/soulx.py` + `docker/modal-soulx/`) — the toolkit's
  default talking head generator (Soul AI Lab, Apache 2.0, 1.3B), Modal-only. Preserves the
  input aspect ratio, so 16:9 presenter images come back 16:9 with no `--preprocess`
  workaround, and `--size` snaps to the model's latent grid while keeping the aspect. It is
  the default because identity holds over a long take: measured at **97% of frame-zero
  sharpness at 70s, flat across 72 segments**, so per-scene generation is a choice rather
  than a workaround. ~$0.0024 per second of output against SadTalker's ~$0.0014. Decision
  table in CLAUDE.md, full detail in `docs/soulx.md`. (#81)

### Changed
- **`docker/modal-soulx/` keeps its weights in a Modal Volume**, unlike the other Modal apps
  which bake them into the image. Measured: redeploy after a code change is ~2.3s against
  79-385s baked, while cold start and generation speed are unchanged. Needs a one-off
  `modal run …::populate_weights` (14.7 GB). The settled apps stay baked.
- Upstream repo ref and both model revisions in `modal-soulx` pinned by SHA — weights in a
  Volume aren't tied to the image, so nothing else prevents drift. (same lesson as #71/#74)
- **`NarratorPiP`** honours its `objectPosition` prop, which was declared and documented but
  silently ignored by a hardcoded value, and gains an `objectFit` prop (default `contain`,
  unchanged behaviour).
- `tools/cloud_gpu.py` learned `soulx`, including its GPU tier, so its jobs report a cost
  estimate.

### Removed
- **EchoMimicV3** (`tools/echomimic3.py`, `docker/modal-echomimic3/`, `docs/echomimic3.md`),
  added earlier in this unreleased cycle and never part of a tagged release. Identity drift
  made it unusable for real narration: on a controlled A/B — same photo, same 80s audio,
  same 544x736 — it fell to 50% of frame-zero sharpness by 70s, collapsing to a featureless
  smear, where SoulX-FlashHead held 97%. The failure is absorbing rather than gradual, since
  each segment re-anchors on the previous one's output, so it could not be tuned out; it
  implied a ~30s render ceiling that shaped the surrounding design. SoulX supersedes it on
  every axis that mattered — drift, cost (~3.7x cheaper), wall clock (~3-6x faster) — while
  keeping aspect-ratio preservation. (#81, closes #80)

---

## 2026-08-27 (v0.19.0)

### Added
- **uv is the install path** — `pyproject.toml` + `uv.lock`; `uv sync` (extras: `whisper`, `modal`, `youtube`) and every doc/tool now says `uv run tools/…`. `tools/requirements.txt` stays as the pip fallback. Contributed by @AsharibAli (#33, rebased as #59).
- **Kiro CLI support** — `scripts/migrate_to_kiro.py` installs the toolkit's skills/commands for Kiro with steering-file generation; see `docs/kiro.md`. Contributed by @homeo26 (#55). Follow-up refactor tracked in #58.
- **Version maintenance (closes #23)** — `scripts/check_versions.py` (toolkit-wide Remotion/Python/toolkit staleness, wired into `/versions`); `tests/render-baseline/` + `scripts/render-baseline.mjs` + `render-baseline.yml` (same-runner before/after render diff that gates Remotion bumps in CI); `scripts/bump_remotion.py` as `/versions --bump-toolkit`; grouped monthly Dependabot for the `remotion` group. (#63, #64, #65)
- **`sync_timing.py` generic mode** — pairs any `*AudioFile` with `*DurationSeconds` so ticket-driven and future config layouts work without tool changes. (#62)
- **Community add-ons** README section for integrations that live in their own repos (first entry: VOICEPEAK offline Japanese TTS, #42), plus CONTRIBUTING guidance for single-vendor integrations and automated/AI-generated PRs. (#60, #61)

### Changed
- **Remotion 4.0.425 → 4.0.518** across all templates, examples, showcases and the baseline harness — first bump gated by the render-baseline check (all frames identical). (#67)
- All Remotion projects now pin the same exact version (four stragglers were on 4.0.381/4.0.382/`^4.0.0`).

### Housekeeping
- External PR queue triaged: MiniMax video/music and Atlas Cloud bot PRs closed under the new policy; Edge-TTS (#46) has review feedback pending.
- Stale branches removed; unmerged experiments preserved as `archive/*` tags.

---

## 2026-08-26 (v0.18.0)

### Added
- **YouTube publishing** — `tools/youtube_upload.py` + `/publish` command. OAuth login (`--auth`), private-by-default upload, scheduled go-live (`--publish-at`), thumbnails, `--dry-run` validation, 403s classified by API reason. Setup guide in `docs/youtube-upload.md`; google-* deps are opt-in in `tools/requirements.txt`. (#29, #37, #38, #39, #40)
- **60db TTS provider** — third `voiceover.py --provider 60db` backend plus standalone `tools/sixtydb_tts.py` (synthesize / stream / websocket transports, `--list-voices`). Unified 0-1 voice settings auto-convert to 60db's 0-100 scale; `redub.py --tts-provider 60db`. Brands carry a `sixtydb` block in `voice.json`. (#27)
- **`_internal/narrative-patterns.md`** — Discovery Story structure for sprint reviews (reorder by narrative arc, turning-point slides, honest bridge framing).
- **`docs/ltx2.md` Windows gotchas** + LTX-2 Modal config wired into `verify_setup.py`. (#51, #53)

### Changed
- **Remotion skills sync** now tracks upstream's new per-topic layout via `skills/remotion-best-practices/` (router skill embedding markup/maps/captions/render/saas/studio/docs/upgrade). Workflow fails loudly on the next re-layout and carries the upstream skill version in PR titles. First sync from the new layout: 39 → 137 files. (#56)
- **Modal image-edit** requests A100-80GB (OOMed on 40GB).
- `.gitignore` covers `scratch/`, Playwright run artifacts, recursive `assets/voices`; duplicate project copies under `projects/` untracked.

### Fixed
- Weekly Remotion skills sync had failed every run since 2026-07-13 (upstream directory moved).
- `verify_setup.py` accepts both `modal` CLI JSON key casings when detecting deployed apps.

### Registry
- Added missing entries: `align_captions` tool, `quick-spot` / `data-viz-chart` / `ds-crt-stinger` / `sky-blue-short` examples. `the-space-between` marked showcase-only (no source directory).

---

## 2026-06-10 (v0.17.0)

### Added
- **concept-explainer-short template** — 9:16 vertical TikTok/Reels/Shorts explainers. First pure Python/moviepy template: `scenes.json` → `gen_vo.py` (per-scene TTS) → `gen_captions.py` (whisper forced-align) → `build.py` (Ken Burns stills, boomerang clips, karaoke caption pills, ducked music). (#31)
- **`examples/sky-blue-short`** — 52s showcase of the template with all assets committed.
- **TTS pacing QC** — `tools/pacing.py`; `voiceover.py` and `qwen3_tts.py` report per-take `wpm` + `pacing` label and accept `--max-wpm` for a pitch-preserving atempo clamp. Fixes cloned voices inheriting a rushed pace from their reference. (#30)

### Docs
- README restructured to lead with finished videos; "Using with Codex" moved to `docs/codex.md`.

---

## 2026-06-08 (v0.16.0)

### Added
- **Ideogram 4 text-to-image** — `tools/ideogram4.py` + `ideogram4` skill. Structured JSON captions for legible in-image text and exact brand colours; hosted paid API. FLUX-vs-Ideogram decision rule added to CLAUDE.md. (#28)
- **Qwen3-TTS v0.2** — VoiceDesign mode (`--design-instruct`), shared-prompt cloning for consistent multi-scene narration, `clone.design` brand schema with self-caching reference; `tools/align_captions.py` (Scribe-based caption timing alignment).
- **Codex migration flow** for toolkit skills/commands (#16, thanks @kimhoontae-gogo); thin `AGENTS.md`.

### Changed
- Remotion pinned to exact versions in lab examples; CLAUDE.md clarifies the `<OffthreadVideo>` choice vs upstream `@remotion/media`.
- Official Remotion skills synced (upstream 277510e). (#17)
- GitHub Actions bumped to Node 24-compatible versions; docker-build workflow handles multi-commit pushes and skips `modal-*` dirs.

### Fixed
- Python 3.9 compatibility restored across `tools/` via `from __future__ import annotations`. (#22)
- `modal-upscale` basicsr build conflict. (#19, thanks @ATECHPCS)
- Codex migration name read from registry; no-op skip entry dropped. (closes #20)

---

## 2026-04-21 (v0.15.0)

### Added
- **LTX-2 style LoRA support** with `crt-terminal` preset.
- **`examples/ds-crt-stinger`** — 6s CRT LoRA + grunged-logo brand stinger.
- Animated synthwave README banner (`showcase/banner`).

### Changed
- Official Remotion skills synced (upstream 41a3b06); "Stsrt" → "Start" typo fixed. (#14)

---

## 2026-04-09 (v0.14.2)

### Added
- **moviepy skill** (`.claude/skills/moviepy/`) — Python video composition for overlaying deterministic text on AI-generated video (LTX-2, SadTalker). Leads with the "trustworthy text" framing: news, trailers, lower thirds, and social captions all need deterministic overlay because AI models can't guarantee readable text. Covers the PIL workaround for moviepy 2.x's TextClip ascender-clipping bug, audio-anchored timelines, common recipes (labels on LTX-2 clips, lower thirds on SadTalker heads, tinted overlays for contrast, split-screen composites, music + VO mixing), 2.x API gotchas, and a genres-where-this-shines table.
- **`examples/quick-spot`** — runnable 15-second ad-style moviepy example. Audio-anchored timeline, PIL text rendering with cross-platform font loading, optional per-scene VO + ducked music, solid-colour backgrounds. Renders with `python3 build.py` and zero external assets.
- **`examples/data-viz-chart`** — runnable animated time-series chart (real GitHub star history) demonstrating the "matplotlib for data + moviepy for trustworthy text" split that mirrors real news-graphics pipelines. Cache-aware matplotlib step.
- **`examples/hello-world`** — minimal Remotion sprint-review example finally committed. The root README's quick-start callout has been pointing at `cd examples/hello-world && npm install && npm run render` since Feb 23, but the example had zero git history on any branch. Fixes the broken link for anyone cloning the repo.
- **LTX-2 skill: Stylized Character Cameo** — new use case in `.claude/skills/ltx2/SKILL.md` documenting LTX-2 image-to-video as a SadTalker alternative for non-photoreal faces (fantasy characters, heavy beards, masks, helmets) where lip-sync precision matters less than motion + atmosphere.
- **CLAUDE.md: Audio-Anchored Timelines** — new subsection in the Video Timing guide complementing the existing reactive `sync_timing.py` flow with a prevention-first pattern (generate audio first, anchor visuals to absolute timestamps). Especially useful for single-file moviepy `build.py` projects.

### Changed
- **`tools/requirements.txt`** adds `Pillow>=10.0`, `moviepy>=2.0`, `matplotlib>=3.7`. One install command (`python3 -m pip install -r tools/requirements.txt`) now covers every Python feature in the toolkit: AI voiceover, image gen, music gen, and the new moviepy examples.
- **Root README quick-start** switches to `python3 -m pip install` — the [PyPA-recommended invocation](https://pip.pypa.io/en/stable/user_guide/#running-pip) that works identically on macOS, Linux, and Windows and guarantees `pip` and `python3` resolve to the same environment.
- **`docs/sadtalker.md`** adds a "When NOT to Use SadTalker" section pointing at LTX-2 image-to-video for stylized characters, heavy facial hair, masks, and non-frontal angles.
- **`examples/README.md`** splits "Using Examples" into Remotion (`npm install` flow) and moviepy (`python3 build.py` flow) sections and adds both new examples to the listing table.

### Fixed
- **Hardcoded macOS font paths in `examples/quick-spot/build.py` and `examples/data-viz-chart/build.py`** — both examples used `/System/Library/Fonts/Supplemental/Arial Bold.ttf` directly, which would crash with `FileNotFoundError` on Linux and Windows. Replaced with a `platform.system()`-keyed fallback chain that tries OS-appropriate sans-serif fonts (Arial on macOS and Windows, DejaVu Sans on Linux, with multiple candidate paths each) and falls back to `ImageFont.load_default()` if nothing matches, so the examples never crash on unusual setups. Both examples now genuinely run cross-platform as their READMEs promise.
- **Latent Pillow gap in `tools/flux2.py` and `tools/image_edit.py`** — both tools imported PIL inside a `try/except ImportError` guard with a friendly error message, but Pillow was never actually declared in `tools/requirements.txt`. Users had to install Pillow manually after hitting the runtime error. Free-rider fix via the moviepy declaration.
- **Guarded imports** added to both moviepy example `build.py` files — matches the existing `flux2.py` / `image_edit.py` friendly-error pattern. Prevents bare `ModuleNotFoundError` tracebacks for users who run the examples before installing dependencies.

### Docs
- **README examples table** adds `q2-townhall-longarm-ad` (Super Bowl-style launch ad with LTX-2 animated Lugh cameo) and `q2-townhall-stars` (GitHub star history time-lapse) as finished demo entries dated 2026-04-08.
- **moviepy skill** added to the README's skills table and `_internal/toolkit-registry.json`.

---

## 2026-04-04 (v0.14.1)

### Added
- **`tools/chain_video.py`** — Chain LTX-2 clips with visual continuity (each scene uses previous clip's last frame as input). Supports per-scene prompts via JSON, auto-resume of interrupted runs, and structured progress reporting.
- **Structured progress reporting** (`--progress json`) for all cloud GPU tools
- **OpenClaw skill: yieldMs polling** — agent stays in poll loop for live progress instead of going silent with `background:true`
- **OpenClaw skill: style drift prevention** — rules to prevent anime/Asian style drift in chained LTX-2 sequences (per-scene prompts + negative prompts)

### Changed
- **acemusic cloud API** is now default music generation provider (official API, XL Turbo 4B model with 5Hz LM thinking)
- **SadTalker timeouts** increased: 20x multiplier + 300s buffer for size=512+gfpgan on A10G
- Synced official Remotion skills (upstream d5d3955)

### Fixed
- chain_video: glob matching for manually-renamed clips (e.g. `chain-05-brigid.mp4`)
- chain_video: 20-min subprocess timeout, stderr in error messages, JSON validation, temp file cleanup
- chain_video: registered in toolkit-registry.json
- R2 recovery note for SadTalker client timeouts

---

## 2026-03-26 (v0.13.2)

### Changed
- **Default cloud provider switched to Modal** — All 6 cloud GPU tools (flux2, image_edit, music_gen, qwen3_tts, sadtalker, voiceover) now default to `--cloud modal` instead of `--cloud runpod`. RunPod remains available via `--cloud runpod`.

### Fixed
- Remove unused imports in `tools/ltx2.py` (`json`, `requests`) — flagged by code quality bot

---

## 2026-03-25 (v0.13.1)

### Added
- **`tools/ltx2.py`** — AI video generation using LTX-2.3 22B DiT model
  - Text-to-video generation from prompts (~5s clips at up to 1024x1536)
  - Image-to-video — animate still images with motion prompts
  - Joint audio+video generation (synchronized ambient audio)
  - Quality presets: `standard` (30 steps) and `fast` (15 steps)
  - Valid frame counts: 25-193 frames ((n-1)%8==0), 24fps default
  - Modal deployment on A100-80GB with baked weights (~55GB)
  - Estimated cost: ~$0.23 per 5-second clip
- **LTX-2 skill** (`.claude/skills/ltx2/`) — prompting guide, parameters, video production use cases
- **Openclaw skill updated** — LTX-2 added as section 4d (video clips), setup instructions, cost table

### Technical Notes
- `torch.inference_mode()` required around pipeline call — without it, Gemma text encoder retains ~37GB of autograd activations causing OOM (upstream bug: Lightricks/LTX-2#152)
- Weights hosted on `Lightricks/LTX-2.3` HuggingFace repo (separate from the code repo `Lightricks/LTX-2`)
- `transformers` pinned to 4.57.6 (5.x breaks `Gemma3TextConfig.rope_local_base_freq`)
- HuggingFace token required for both LTX-2 (rate limiting) and Gemma 3 (gated model)

---

## 2026-03-22 (v0.12.0)

### Added
- **`tools/music_gen.py`** — AI music generation using ACE-Step 1.5
  - Text-to-music with precise BPM, key, time signature, and duration control
  - Vocal music with lyrics in 50+ languages
  - Cover / style transfer from reference audio (`--cover`)
  - Stem extraction — isolate vocals, drums, bass, etc. (`--extract`)
  - **8 scene presets** for video production: `corporate-bg`, `upbeat-tech`, `ambient`, `dramatic`, `tension`, `hopeful`, `cta`, `lofi`
  - **Brand-aware generation** — `--brand` loads style hints from brand.json
  - `--setup` creates RunPod template + endpoint via GraphQL
  - Docker image: `ghcr.io/conalmullan/video-toolkit-acestep:latest` (CUDA 12.8, baked model weights)
  - MIT licensed model — free alternative to ElevenLabs Music with more control
  - ~2-3s inference on GPU (turbo mode, 8 steps)
- **ACE-Step skill** (`.claude/skills/acestep/`) — prompt engineering patterns, lyrics formatting, scene preset guide, video production integration

---

## 2026-03-22 (v0.11.1)

### Fixed
- **Queue timeout safeguard** — All 6 RunPod tools now auto-cancel jobs stuck in queue for >5 min, preventing runaway billing from GPU unavailability
- **Cancel on exit** — Jobs are cancelled on RunPod when client polling times out (previously left orphaned jobs in queue indefinitely)
- **R2 upload support** — flux2.py and image_edit.py now pass R2 config to handlers (was already supported server-side but never triggered)

### Changed
- **flux2 default GPU** — Changed from ADA_24 (RTX 4090, frequently throttled) to AMPERE_24,ADA_24 (fallback chain)
- **flux2 setup** — `--setup-gpu` now accepts comma-separated GPU types for fallback
- **RunPod skill** — Added API reference (GraphQL queries/mutations, REST endpoints, GPU IDs, R2 CLI patterns)
- **R2 docs** — Added operations guide (listing, cleanup, `--region auto` gotcha) to runpod-setup.md

---

## 2026-03-15 (v0.11.0)

### Added
- **`tools/flux2.py`** — AI image generation and editing using FLUX.2 Klein 4B
  - Text-to-image generation from prompts (~2.5s fast mode, ~8s quality mode)
  - Image editing with reference images (`--input photo.jpg --prompt "..."`)
  - Multi-image compositing (up to 3 reference images)
  - **8 scene presets** for video production: `title-bg`, `problem`, `solution`, `demo-bg`, `stats-bg`, `cta`, `thumbnail`, `portrait-bg`
  - **Brand-aware generation** — `--brand digital-samba` reads brand.json and injects color palette into prompts
  - Preset + prompt layering — preset provides style/mood, `--prompt` adds subject context
  - `--setup` creates RunPod template + endpoint via GraphQL
  - `--list-presets` shows all available scene presets
  - Docker image: `ghcr.io/conalmullan/video-toolkit-flux2:latest` (baked model weights, ~15GB)
  - Apache 2.0 licensed model (commercial OK)
- **"The Space Between"** — [showcase video](https://demos.digitalsamba.com/video/the-space-between.mp4) demonstrating end-to-end AI video creation
  - Avatar generated with flux2, voiced with Qwen3-TTS, animated with SadTalker, composed in Remotion
  - Concept, script, voice, visuals — all AI-generated
  - New video format: video essay (beyond sprint reviews and product demos)

### Changed
- **Baked Docker images rebuilt** — both `flux2` and `qwen3-tts` images now include model weights baked in, eliminating cold-start model downloads (~30s cold start vs minutes)
- Updated `_internal/toolkit-registry.json` with flux2 tool entry, cloud endpoint, scene presets, and showcase example
- Updated README with flux2 tool, Docker image, demo table entry, and Cloud GPU tools table

---

## 2026-02-25 (v0.10.1)

### Added
- **`tools/sync_timing.py`** — Audio-to-config timing sync tool
  - Measures actual per-scene audio durations via ffprobe
  - Auto-detects config file and template type (sprint-review v1/v2, product-demo)
  - 3-pass audio-to-scene matching: `audioFile` field → index → name
  - Comparison table with delta indicators; skips changes < 0.3s
  - `--apply` updates `durationSeconds` in-place (creates `.bak` backup)
  - `--voiceover-json` accepts voiceover.py output directly
  - Suggests `playbackRate` adjustments for demo scenes

### Changed
- **CLAUDE.md slimmed 44%** (861 → 480 lines)
  - Removed catalog data duplicated in `toolkit-registry.json` (skills, commands, components, transitions, presets, Docker images, duplicate CLI examples)
  - Added cross-references to registry for structured data
  - All workflow guidance, timing knowledge, code patterns, and tool-specific gotchas retained
- Integrated `sync_timing.py` into Video Production Workflow (step 7) and TTS drift feedback loop
- Updated toolkit-registry.json with `sync_timing` tool entry

---

## 2026-02-24 (v0.10.0)

### Added
- **Qwen3-TTS integration** — Self-hosted TTS via RunPod as free alternative to ElevenLabs
  - `tools/qwen3_tts.py` standalone CLI with 9 speakers, tone presets, voice cloning
  - `voiceover.py --provider qwen3` for per-scene generation
  - Docker image: `ghcr.io/conalmullan/video-toolkit-qwen3-tts:latest`
  - Temperature/top_p params for expressiveness control
- **`/voice-clone` command** — Record, test, and save a cloned voice to a brand profile
- **`sprint-review-v2` template** — Composable scene-based architecture for sprint reviews
- **`FilmGrain` component** — SVG noise overlay for cinematic film texture
- **`hello-world` example** — Minimal 25s video, zero config, renders in 2 minutes
- **`runpod` skill** — Cloud GPU setup, Docker images, endpoint management, troubleshooting
- **Remotion skills sync update** — 6 new upstream rules (audio-visualization, ffmpeg, light-leaks, subtitles, transparent-videos, voiceover)

### Improved
- **Onboarding & developer experience** — All API keys now optional, videos render with just Node.js
  - `.env.example` reorganized with sections, all keys commented out by default
  - README and getting-started.md restructured with "Try It Now" quick start
  - All Python tools show actionable guidance when API keys are missing (add key, use alternative, or skip)
  - `voiceover.py` catches missing `elevenlabs` pip package with helpful message pointing to Qwen3-TTS

### Changed
- Updated README with prerequisites table, Qwen3-TTS, Docker images, voice-clone command
- Updated CLAUDE.md with FilmGrain component and checkerboard transition
- Updated roadmap metrics and skill status table
- Fixed cho-oyu demo link; added cortina sprint to demos table

---

## 2026-02-19 (v0.9.3)

### Added
- **Official Remotion skills** — Synced 33 rule files from [remotion-dev/skills](https://github.com/remotion-dev/skills) into `.claude/skills/remotion-official/`
- **Weekly sync workflow** — GitHub Actions checks upstream every Monday and opens a PR if files changed
- **Sync documentation** — `docs/remotion-skills-sync.md` explaining the split and sync process

### Changed
- **Remotion skill split** — Custom `remotion` skill trimmed from ~470 to ~160 lines, now covers only toolkit-specific patterns (transitions, components, conventions). Core framework knowledge deferred to `remotion-official`
- Updated CLAUDE.md skills table with both remotion skills

---

## 2026-01-25 (v0.9.2)

### Added
- **Per-scene voiceover generation** - New recommended workflow for audio
  - `voiceover.py --scene-dir` processes all `.txt` files in a directory
  - `voiceover.py --concat` joins scene audio for SadTalker narrator
  - Each scene's `<Audio>` starts at frame 0 within its `Series.Sequence`
  - Regenerate individual scenes without re-doing the whole video
  - Scene durations match audio naturally (no manual offset calculations)

- **Sprint-review template per-scene audio support**
  - `audioFile` field on `DemoConfig`, `info`, `overview`, `summary`
  - SprintReview.tsx renders per-scene `<Audio>` elements
  - Backward compatible: global voiceover track still works

- **Updated `/generate-voiceover` command**
  - Detects `public/audio/scenes/*.txt` and offers per-scene mode
  - Per-scene is now the default when scene scripts exist
  - Concat option for SadTalker integration

### Changed
- Updated documentation (CLAUDE.md, README.md, getting-started.md)

---

## 2025-12-30 (v0.7.0)

### Added
- **`tools/dewatermark.py`** - AI-powered watermark removal using ProPainter
  - Cloud GPU processing via RunPod serverless (~$0.05-0.30/video)
  - Local processing for NVIDIA GPU users (8GB+ VRAM)
  - Auto resize-ratio: 1.0 for <30s, 0.75 for <1min, 0.5 for longer
  - Chunked processing for long videos with overlap stitching
  - Cloudflare R2 for reliable file transfer
  - Automated `--setup` for one-command RunPod configuration
  - Presets: notebooklm, tiktok, sora, stock-br, stock-bl, stock-center

- **`tools/locate_watermark.py`** - Helper tool for finding watermark coordinates
  - Coordinate grid overlay for visual identification
  - Region verification across multiple frames
  - Preset support for common watermarks
  - Requires ImageMagick (`brew install imagemagick`)

- **`tools/notebooklm_brand.py`** - Post-processing for NotebookLM videos
  - Trims NotebookLM visual outro while preserving full audio
  - Adds custom branded outro with logo and URL
  - Handles freeze-frame bridging when audio exceeds video

- **Docker infrastructure** for RunPod serverless (`docker/runpod-propainter/`)
  - Dockerfile, handler.py, README
  - Public image available for quick deployment

- **Documentation**
  - `docs/optional-components.md` - ML component installation guide
  - `docs/runpod-setup.md` - Cloud GPU setup guide

### Changed
- ElevenLabs TTS: Added `eleven_v3` (alpha) model option
- Updated `CLAUDE.md` with dewatermark and locate_watermark documentation
- Updated `.env.example` with RunPod and R2 configuration

---

## 2025-12-28 (v0.6.0)

### Added
- **`redub.py --sync`** - Word-level time remapping for TTS synchronization
  - Uses ElevenLabs `convert_with_timestamps` API for character-level alignment
  - Uses Scribe word-level timestamps from original audio
  - Builds segment mapping (configurable via `--segment-size`, default: 15 words)
  - Applies variable speed per segment via FFmpeg filtergraph
  - Fixes audio drift when TTS voice speaks at different pace than original
  - Solves: TTS often starts fast and ends slow, causing 3-4+ second drift

### Changed
- Updated `CLAUDE.md` with Redub Sync Mode documentation
- TTS duration now measured via ffprobe (more accurate than timestamp data)

---

## 2025-12-28 (v0.5.0)

### Added
- **`tools/addmusic.py`** - Add background music to existing videos
  - Generate music via ElevenLabs or use existing audio file
  - Options: `--music-volume`, `--fade-in`, `--fade-out`
  - Works on any video file without requiring a project structure

### Changed
- Updated `CLAUDE.md` with addmusic documentation

---

## 2025-12-28 (v0.4.0)

### Added
- **`/redub` command** - Redub existing videos with a different voice
  - Guided workflow for voice selection and transcript handling
  - Supports transcript review/editing before TTS generation
  - Works on any video file without requiring a project structure

- **`tools/redub.py`** - Voice replacement utility tool
  - Extracts audio from video (FFmpeg)
  - Transcribes audio to text (ElevenLabs Scribe STT)
  - Generates new audio with target voice (ElevenLabs TTS)
  - Replaces audio track in video (FFmpeg)
  - Options: `--save-transcript`, `--transcript`, `--keep-temp`, `--speed`
  - Warns about duration mismatches between original and new audio

- **Utility tools concept** - Documented distinction between project tools and utility tools
  - Project tools (voiceover, music, sfx): Used during video creation workflow
  - Utility tools (redub): Quick transformations on existing videos, no project needed

### Changed
- Updated `CLAUDE.md` with utility tools documentation
- Updated Python Tools section to include redub

---

## 2025-12-20

### Added
- **GitHub Actions release automation** (`.github/workflows/release.yml`)
  - Tag-triggered releases (`v*` tags)
  - Manual dispatch option with version input
  - Auto-generates changelog from commits since last tag
  - Reads version from `toolkit-registry.json`

- **`/versions` command** - Check dependency versions and toolkit updates
  - Detects Remotion package version mismatches in projects
  - Compares local toolkit version against GitHub releases
  - Offers to fix version mismatches by pinning and reinstalling
  - Documents common issues and prevention strategies

### Fixed
- **Remotion version mismatch** in digital-samba-free-intro project
  - Pinned all Remotion packages to 4.0.387 (was mixing 4.0.383 and 4.0.387)
  - Removed `^` prefix to prevent future drift

---

## 2025-12-10

### Added
- **`/scene-review` command** - Dedicated scene-by-scene review with Remotion Studio
  - Starts Remotion Studio for visual verification
  - Walks through scenes one by one (not summary tables)
  - Generic - works with any template's config
  - `/video` now delegates to `/scene-review` when phase is `review`
  - `/generate-voiceover` warns if review not complete
  - Fixes: Review kept getting skipped because `/video` command was too long

### Changed
- **Consolidated tracking files** - Simplified from 4 files to 3:
  - `ROADMAP.md` - What we're building (removed duplicate "Next Actions" and "In Progress" sections)
  - `BACKLOG.md` - What we might build (removed all implemented items, now only future ideas)
  - `CHANGELOG.md` - What we built (historical record)
  - Removed `FEEDBACK.md` - Evolution principles moved to `docs/contributing.md`
- Created `docs/contributing.md` with evolution principles and contribution workflow

### Fixed
- **Slash commands not loading** - Renamed `/skill` to `/skills` to avoid conflict with built-in `Skill` tool. The naming collision was silently preventing ALL custom commands from loading. Bug reported to Anthropic.

### Removed
- **`/review` command** - Clashed with Claude Code's built-in PR review command. Replaced by `/scene-review`.

### Added
- **Animation components** (`lib/components/`)
  - Envelope - 3D envelope with opening flap animation, configurable message
  - PointingHand - Animated hand emoji with directional slide-in and pulse effect
- **`/contribute` command** - Guided contribution workflow
  - Report issues via `gh issue create`
  - Submit PRs for improvements
  - Share skills and templates
  - Safety checks to exclude private project work
- **Evolution narrative** across all commands
  - Consistent "## Evolution" section in each command
  - Local improvement workflow
  - Remote contribution links (GitHub issues + PRs)
  - Command history tracking
- **Template evolution guidance** in `/template` command
  - How to add features to existing templates
  - Template maturity indicators
  - Pattern extraction to shared lib/
- **Product demo template README** (`templates/product-demo/README.md`)
  - Quick start guide, configuration, project structure
  - All 7 scene types documented with examples
  - Narrator PiP and demo chrome options

### Changed
- **`projects/` now fully gitignored** - User video work stays completely private
  - Only `.gitkeep` is tracked to preserve directory structure
  - Safe to contribute without exposing project content
- **New `examples/` directory** for shareable showcase projects
  - Configs, scripts, and docs are tracked
  - Large media files (mp4, mp3) are gitignored
  - Each example includes `ASSETS-NEEDED.md` documenting required media
- **`/contribute` now supports example projects** (Option 5)
  - Guides copying from projects/ to examples/
  - Auto-generates ASSETS-NEEDED.md
  - Creates README with quick start instructions
  - **Contributor recognition** with backlinks to website/org
- **CONTRIBUTORS.md** - Recognition for organizations and individuals who share examples
- **Documentation updates**
  - `docs/getting-started.md` - Updated for `/video` command, added multi-session workflow
  - `docs/creating-brands.md` - Updated for `/brand` command integration
- **Renamed `skills-registry.json` → `toolkit-registry.json`**
  - Consistent format across all entries (path, description, status, created, updated)
  - Added `components` section for shared lib/components
  - Synced with actual commands (7), templates (2), components (9)

---

## 2025-12-09

### Added
- **Multi-session project system** (`lib/project/`)
  - `types.ts` - TypeScript schema for project.json
  - `README.md` - Documentation for project lifecycle and phases
  - Projects now track: phase, scenes, assets, audio, session history
  - Filesystem reconciliation (compares intent vs reality)
  - Auto-generated CLAUDE.md per project for instant context

- **Unified commands** - Context-aware entry points that list existing items or create new:
  - `/video` - Replaces `/new-video`. Scans projects, offers resume or new
  - `/brand` - Replaces `/new-brand`. Lists brands, edit or create new
  - `/template` - Lists templates or creates new ones (copy, minimal, from project)
  - `/skill` - Lists installed skills or creates new ones

### Changed
- **Command pattern unified** - All domain commands now scan first, then offer actions
- Commands integrate with project.json for state tracking
- README.md updated with new commands and multi-session workflow

### Removed
- `/new-video` - Replaced by `/video`
- `/new-brand` - Replaced by `/brand`

### Notes
- After creating/modifying commands or skills, restart Claude Code to load changes

---

## 2025-12-09

### Added
- **Shared component library** (`lib/`)
  - `lib/theme/` - ThemeProvider, useTheme, createStyles, type definitions
  - `lib/components/` - Reusable video components:
    - AnimatedBackground (supports subtle, tech, warm, dark variants)
    - SlideTransition (fade, zoom, slide-up, blur-fade)
    - Label (floating badge with optional JIRA ref)
    - Vignette (cinematic edge darkening)
    - LogoWatermark (corner branding)
    - SplitScreen (side-by-side video layout)
    - NarratorPiP (picture-in-picture presenter - needs refinement)
  - `lib/index.ts` - Unified exports for templates

### Changed
- **Templates now import from shared library**
  - sprint-review: Uses lib components via re-exports
  - product-demo: Uses lib components (keeps local NarratorPiP for different API)
  - Both templates use shared ThemeProvider from lib/theme
  - Deleted duplicate component files from templates
- Updated ROADMAP.md - Shared component library marked complete
- Updated BACKLOG.md - Added NarratorPiP API refinement task

---

## 2025-12-09 (earlier)

### Added
- **Product demo template** (`templates/product-demo/`)
  - Scene-based composition (title, problem, solution, demo, stats, CTA)
  - Config-driven content via `demo-config.ts`
  - Dark tech aesthetic with animated background
  - Narrator PiP (picture-in-picture presenter)
  - Browser/terminal chrome for demo videos
  - Stats cards with spring animations
- **`/new-brand` command** - guided brand profile creation
  - Extract colors from website URL
  - Manual color entry with palette generation
  - Logo and voice configuration guidance
- **Digital Samba brand profile** (`brands/digital-samba/`)
- **Template-brand integration**
  - Brand loader utility (`lib/brand.ts`)
  - `brand.ts` in each template (generated at project creation)
  - `project.json` for tracking project metadata
  - Brand generator script (`lib/generate-brand-ts.ts`)
- **`/new-video` command** - unified project creation wizard
  - Choose template (sprint-review, product-demo)
  - Choose brand from available brands
  - **Scene-centric workflow:**
    - Content gathering (URLs, notes, paste)
    - Claude proposes scene breakdown
    - Interactive scene refinement
    - Scene types: title, overview, demo, split-demo, stats, credits, problem, solution, feature, cta
    - Visual types: `[DEMO]`, `[SCREENSHOT]`, `[EXTERNAL VIDEO]`, `[SLIDE]`
  - Generates VOICEOVER-SCRIPT.md with narration and asset checklist
  - Creates project.json with scene tracking
  - Guides through asset creation phase
  - Ongoing project support (resume any project)

### Changed
- **Replaced `/new-sprint-video` with `/new-video`** - single entry point for all templates
- Templates now load theme from `brand.ts` instead of hardcoded values
- Updated CLAUDE.md with new workflow and commands
- Updated ROADMAP.md - Phase 3 template-brand integration complete
- **Added Video Timing section to CLAUDE.md** - pacing rules, scene durations, timing calculations
- **Removed `/convert-asset` from backlog** - FFmpeg skill handles this conversationally
- **Removed `/sync-timing` from backlog** - timing knowledge now in CLAUDE.md

---

## 2025-12-08

### Added
- **Open source release** - Published to GitHub at digitalsamba/claude-code-video-toolkit
- **Brand profiles system** (`brands/`)
  - `brand.json` for colors, fonts, typography
  - `voice.json` for ElevenLabs voice settings
  - `assets/` for logos and backgrounds
  - Default brand profile included
- **Documentation** (`docs/`)
  - `getting-started.md` - First video walkthrough
  - `creating-brands.md` - Brand profile guide
  - `creating-templates.md` - Template creation guide
- **Environment variable support**
  - `ELEVENLABS_VOICE_ID` - Set voice ID via env var
  - Falls back to `_internal/skills-registry.json` if not set
- `/generate-voiceover` command - guided ElevenLabs TTS generation
- `/record-demo` command - guided Playwright browser recording
- Interactive recording stop controls (Escape key, Stop button)
- Window scaling for laptop screens (`--scale` option, default 0.75)
- FFmpeg skill (beta) - common video/audio conversion commands
- Playwright recording skill (beta) - browser demo capture
- Playwright infrastructure (`playwright/`) with recording scripts
- Python tools: `voiceover.py`, `music.py`, `sfx.py`
- Skills registry for centralized config
- README.md, LICENSE (MIT), CONTRIBUTING.md
- `.env.example` template

### Changed
- **Directory restructure for open source:**
  - `templates/` - Video templates (moved from root)
  - `projects/` - User video projects (moved from root)
  - `brands/` - Brand profiles (new)
  - `assets/` - Shared assets (consolidated)
  - `_internal/` - Toolkit metadata (renamed from `_toolkit/`)
- Updated `/new-sprint-video` command paths
- `tools/config.py` reads from `_internal/` and supports env vars
- Playwright recordings output at 30fps (matches Remotion)

### Fixed
- Removed hardcoded voice ID from committed files
- Proper `.gitignore` for secrets and build artifacts
- FFmpeg trim command syntax (use `-to` not `-t` for end time)
- Playwright double navigation issue
- Recording frame rate mismatch (was 25fps, now 30fps)

---

## 2025-12-04

### Added
- Sprint review template (`templates/sprint-review/`)
  - Theme system with colors, fonts, spacing
  - Config-driven content via `sprint-config.ts`
  - Slide components: Title, Overview, Summary, EndCredits
  - Demo components: DemoSection, SplitScreen
  - NarratorPiP component for picture-in-picture narrator
  - Audio integration (voiceover, background music, SFX)
- `/new-sprint-video` slash command for guided project creation
- Initial workspace setup
- Remotion skill documentation
- ElevenLabs skill documentation
- First video project: sprint-review-cho-oyu
- Voice cloning workflow with ElevenLabs
