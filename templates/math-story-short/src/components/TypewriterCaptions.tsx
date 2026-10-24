import { useCallback, useEffect, useMemo, useState } from 'react';
import { getStaticFiles, interpolate, staticFile, useCurrentFrame, useDelayRender, useVideoConfig } from 'remotion';
import { theme } from '../config/theme';

interface Word {
  text: string;
  startMs: number;
  endMs: number;
}

export const captionsFor = (audioPath: string) => audioPath.replace(/\.[^./]+$/, '') + '.captions.json';

/** No captions JSON yet: spread the script's words over the scene, weighted by length. */
const estimate = (narration: string, durationMs: number): Word[] => {
  const words = narration.split(/\s+/).filter(Boolean);
  const start = 250;
  const span = Math.max(500, durationMs - 900);
  const weights = words.map((w) => w.length + 2 + (/[.!?]$/.test(w) ? 6 : /[,:;]$/.test(w) ? 3 : 0));
  const total = weights.reduce((a, b) => a + b, 0);
  let t = start;
  return words.map((w, i) => {
    const d = (weights[i] / total) * span;
    const word = { text: ` ${w}`, startMs: t, endMs: t + d * 0.8 };
    t += d;
    return word;
  });
};

const MAX_CHARS = 54;
const paginate = (words: Word[]) => {
  const pages: Word[][] = [];
  let cur: Word[] = [];
  for (const w of words) {
    const len = cur.map((c) => c.text).join('').length + w.text.length;
    const prev = cur[cur.length - 1];
    if (prev && (len > MAX_CHARS || /[.!?]["')]*$/.test(prev.text.trim()))) {
      pages.push(cur);
      cur = [];
    }
    cur.push(w);
  }
  if (cur.length) pages.push(cur);
  return pages;
};

/**
 * Terminal-style narration line (the brand's DNA): a crimson prompt and a block
 * cursor, with whole words landing in sync with the voice (readable with the
 * sound off). The word being spoken is bright; the rest of the line settles back.
 * Uses word timing from tools/captions.py when <audio>.captions.json exists,
 * otherwise estimates it from the script.
 */
export const TypewriterCaptions: React.FC<{ audioFile: string; narration: string; frames: number }> = ({ audioFile, narration, frames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const src = captionsFor(audioFile);
  const exists = useMemo(() => getStaticFiles().some((f) => f.name === src), [src]);
  const [loaded, setLoaded] = useState<Word[] | null>(null);
  const { delayRender, continueRender, cancelRender } = useDelayRender();
  const [handle] = useState(() => (exists ? delayRender(`Loading ${src}`) : null));

  const load = useCallback(async () => {
    if (handle === null) return;
    try {
      const res = await fetch(staticFile(src));
      setLoaded((await res.json()) as Word[]);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [src, handle, continueRender, cancelRender]);
  useEffect(() => {
    load();
  }, [load]);

  const words = useMemo(
    () => (loaded && loaded.length ? loaded : estimate(narration, (frames / fps) * 1000)),
    [loaded, narration, frames, fps],
  );
  const pages = useMemo(() => paginate(words), [words]);
  const now = (frame / fps) * 1000;

  let pi = 0;
  pages.forEach((p, i) => {
    if (now >= p[0].startMs - 80) pi = i;
  });
  const page = pages[pi];
  if (!page) return null;

  const shown = page.filter((w) => now >= w.startMs - 40);
  const active = shown.length - 1;
  const speaking = shown.length > 0 && now < shown[active].endMs + 120;
  const cursorOn = speaking || Math.floor(frame / 15) % 2 === 0;
  const pageStart = (page[0].startMs / 1000) * fps;
  const enter = interpolate(frame, [pageStart - 3, pageStart + 3], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  const exit = interpolate(frame, [frames - 8, frames], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

  return (
    <div
      style={{
        position: 'absolute',
        left: theme.layout.gutter,
        right: theme.layout.gutter,
        top: theme.layout.captionsTop,
        display: 'flex',
        gap: 22,
        alignItems: 'flex-start',
        opacity: Math.min(enter, exit),
      }}
    >
      <svg width="32" height="70" viewBox="0 0 32 70" style={{ flexShrink: 0 }}>
        <path d="M6,22 L24,35 L6,48" fill="none" stroke={theme.primary} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div
        style={{
          fontFamily: theme.fonts.sans,
          fontWeight: 700,
          fontSize: 56,
          lineHeight: 1.25,
          letterSpacing: -0.6,
          textShadow: '0 2px 18px rgba(0, 20, 34, 0.9)',
        }}
      >
        {shown.map((w, i) => {
          const inF = frame - ((w.startMs - 40) / 1000) * fps;
          const k = interpolate(inF, [0, 4], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
          return (
            <span
              key={i}
              style={{
                display: 'inline-block',
                whiteSpace: 'pre',
                color: theme.text,
                opacity: k * (i === active ? 1 : 0.62),
                transform: `translateY(${(1 - k) * 12}px)`,
              }}
            >
              {i === 0 ? w.text.trimStart() : w.text}
            </span>
          );
        })}
        <span
          style={{
            display: 'inline-block',
            width: 24,
            height: 54,
            marginLeft: 8,
            verticalAlign: '-8px',
            borderRadius: 3,
            background: theme.primary,
            opacity: cursorOn ? 0.95 : 0,
          }}
        />
      </div>
    </div>
  );
};
