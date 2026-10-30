export type StoryScene = {
  id: string; type: string; title: string; text: string; durationSeconds: number;
  asset?: {kind: 'image' | 'video'; file: string}; audioFile?: string;
  captionsFile?: string; narratorFile?: string;
};
export type StoryProject = {
  name: string; scenes: StoryScene[];
  direction: 'ltr' | 'rtl';
  palette: {background: string; foreground: string; accent: string};
};
function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function text(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label} must be text`);
  return value;
}
function file(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  const path = text(value, 'Asset path');
  if (!path || path.startsWith('/') || /[:\\]/.test(path) || path.split('/').some(part => part === '..')) {
    throw new Error('Assets must be public-relative paths');
  }
  return path;
}
function colour(value: unknown, fallback: string): string {
  if (value === undefined) return fallback;
  if (typeof value !== 'string' || !/^#[0-9a-f]{6}$/i.test(value)) throw new Error('Palette colours must be six-digit hex values');
  return value;
}
export function parseStoryProject(value: unknown): StoryProject {
  if (!object(value) || !Array.isArray(value.scenes) || value.scenes.length === 0) throw new Error('A project needs at least one scene');
  const ids = new Set<string>();
  const scenes = value.scenes.map((scene: unknown): StoryScene => {
    if (!object(scene)) throw new Error('Scene must be an object');
    const id = text(scene.id, 'Scene id');
    if (!id || ids.has(id)) throw new Error(`Duplicate or empty scene ID: ${id}`);
    ids.add(id);
    const duration = scene.durationSeconds;
    if (typeof duration !== 'number' || !Number.isFinite(duration) || duration <= 0) throw new Error(`${id}: positive durationSeconds required`);
    const title = text(scene.title ?? id, 'Title');
    const body = text(scene.onScreenText ?? '', 'onScreenText');
    if (title.length > 120 || body.length > 320) throw new Error(`${id}: keep titles within 120 and onScreenText within 320 characters, or split the scene`);
    const visual = object(scene.visual) ? scene.visual : {};
    const path = file(visual.asset);
    let asset: StoryScene['asset'];
    if (path) {
      if (/\.(mp4|webm|mov)$/i.test(path)) asset = {kind: 'video', file: path};
      else if (/\.(png|jpe?g|webp|gif|svg)$/i.test(path)) asset = {kind: 'image', file: path};
      else throw new Error(`${id}: unsupported visual asset`);
    }
    const audio = object(scene.audio) ? scene.audio : {};
    return {id, type: text(scene.type, 'Scene type'), title,
      text: body, durationSeconds: duration, asset,
      audioFile: file(scene.audioFile ?? audio.file), captionsFile: file(scene.captionsFile ?? audio.captions),
      narratorFile: file(scene.narratorFile)};
  });
  const palette = object(value.palette) ? value.palette : {};
  const direction = typeof value.locale === 'string' && /^(ar|fa|he|ur)(-|$)/i.test(value.locale) ? 'rtl' : 'ltr';
  return {name: text(value.name, 'Project name'), scenes, direction, palette: {
    background: colour(palette.background, '#13251F'), foreground: colour(palette.foreground, '#FFF8E9'),
    accent: colour(palette.accent, '#E1C16D')}};
}
