import { interpolate, spring, useVideoConfig } from 'remotion';
import { story } from '../config/story-config';
import { theme } from '../config/theme';
import { scenes, scenesFrames, useScene } from '../timeline';

const chapterSpan = (c: number) => {
  const own = scenes.filter((s) => s.chapter === c && !s.hook);
  return { from: own[0]?.from ?? 0, to: own.length ? own[own.length - 1].from + own[own.length - 1].frames : 0 };
};

/**
 * Story-style progress segments (one per chapter) plus the chapter title.
 * The title swaps with a per-word rise whenever the chapter changes.
 */
export const ChapterHeader: React.FC = () => {
  const { fps } = useVideoConfig();
  const { frame, scene } = useScene();
  const { gutter, headerTop } = theme.layout;
  const chapter = scene.chapter;
  const span = chapterSpan(chapter);
  const sinceChapter = frame - span.from;
  const enter = spring({ frame: sinceChapter, fps, config: { damping: 18, stiffness: 140 } });
  const outro = interpolate(frame, [scenesFrames, scenesFrames + 12], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  // Hidden through the cold open, then the bars sweep in with chapter 1
  if (scene.hook) return null;
  const shown = interpolate(frame, [span.from, span.from + 10], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const words = story.chapters[chapter].title.split(' ');
  const n = story.chapters.length;
  const gap = 12;
  const segW = (1080 - gutter * 2 - gap * (n - 1)) / n;

  return (
    <div style={{ position: 'absolute', left: gutter, right: gutter, top: headerTop, opacity: outro * (chapter === 0 ? shown : 1) }}>
      <div style={{ display: 'flex', gap }}>
        {story.chapters.map((_, i) => {
          const s = chapterSpan(i);
          const p = interpolate(frame, [s.from, s.to], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
          return (
            <div key={i} style={{ width: segW, height: 6, borderRadius: 3, background: theme.hairline, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${p * 100}%`,
                  height: '100%',
                  borderRadius: 3,
                  background: i === chapter ? theme.primary : theme.cream,
                  opacity: i === chapter ? 1 : 0.55,
                  boxShadow: i === chapter ? `0 0 12px ${theme.primary}` : 'none',
                }}
              />
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 44 }}>
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontWeight: 700,
            fontSize: 30,
            color: theme.cream,
            background: `linear-gradient(135deg, ${theme.primary}, ${theme.primaryDark})`,
            borderRadius: 14,
            padding: '8px 16px',
            letterSpacing: 1,
            boxShadow: '0 8px 24px rgba(193, 18, 31, 0.35)',
            transform: `scale(${interpolate(enter, [0, 1], [0.6, 1])}) rotate(${interpolate(enter, [0, 1], [-8, 0])}deg)`,
            opacity: enter,
          }}
        >
          {String(chapter + 1).padStart(2, '0')}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', columnGap: 16, overflow: 'hidden', paddingBottom: 6 }}>
          {words.map((w, i) => {
            const s = spring({ frame: sinceChapter - 4 - i * 3, fps, config: { damping: 16, stiffness: 160 } });
            return (
              <span
                key={`${chapter}-${i}`}
                style={{
                  fontFamily: theme.fonts.sans,
                  fontWeight: 800,
                  fontSize: 58,
                  letterSpacing: -1.2,
                  color: theme.text,
                  display: 'inline-block',
                  transform: `translateY(${interpolate(s, [0, 1], [70, 0])}px)`,
                  opacity: s,
                }}
              >
                {w}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
};
