/**
 * Captions - Burned-in narration captions with optional karaoke highlight
 *
 * Renders a word-level `<name>.captions.json` written by `tools/captions.py`
 * (the @remotion/captions `Caption[]` shape). Place it inside the same
 * Sequence as the audio it captions, so word times line up with the scene:
 *
 *   <Series.Sequence durationInFrames={...}>
 *     <Audio src={staticFile('audio/scenes/01-title.mp3')} />
 *     <Captions src={captionsFor('audio/scenes/01-title.mp3')} />
 *   </Series.Sequence>
 *
 * Renders nothing until the JSON exists, so templates can include it
 * unconditionally. Words are paged into short lines: a page ends at sentence
 * punctuation, at a pause, or when it hits maxWords / maxChars.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AbsoluteFill,
  getStaticFiles,
  interpolate,
  staticFile,
  useCurrentFrame,
  useDelayRender,
  useVideoConfig,
} from 'remotion';

/** One word, in the @remotion/captions `Caption` shape. */
export interface CaptionWord {
  text: string;
  startMs: number;
  endMs: number;
  timestampMs: number | null;
  confidence: number | null;
}

export interface CaptionsProps {
  /** Captions JSON path relative to public/ (see captionsFor) */
  src: string;
  /** Highlight the word being spoken (default true) */
  karaoke?: boolean;
  /** Max words per caption page (default 7) */
  maxWords?: number;
  /** Max characters per caption page (default 42 — one line at 1080p) */
  maxChars?: number;
  /** A silence longer than this starts a new page (default 600) */
  pauseMs?: number;
  /** How long the last page lingers after its final word (default 400) */
  holdMs?: number;
  /** Break pages at phrases (default false): after a comma, colon or semicolon once a page has three words, and
   * when a page is full, at its last such break rather than mid-phrase */
  phrases?: boolean;
  /** Distance from the bottom of the frame in px (default 80) */
  bottom?: number;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: number;
  color?: string;
  /** Active-word colour when karaoke is on */
  highlightColor?: string;
  /** Pill background behind the text (pass 'transparent' to drop it) */
  background?: string;
  maxWidth?: string;
}

/** Template-config shape: an on/off switch plus any style prop */
export type CaptionsConfig = { enabled?: boolean } & Omit<CaptionsProps, 'src'>;

interface Page {
  words: CaptionWord[];
  startMs: number;
  endMs: number;
}

/** Sibling captions path for an audio file: audio/x.mp3 -> audio/x.captions.json */
export const captionsFor = (audioPath: string): string =>
  audioPath.replace(/\.[^./]+$/, '') + '.captions.json';

const SENTENCE_END = /[.!?…]["')\]]*$/;

const paginate = (
  words: CaptionWord[],
  maxWords: number,
  maxChars: number,
  pauseMs: number,
  holdMs: number,
  phrases = false,
): Page[] => {
  const pages: Page[] = [];
  let cur: CaptionWord[] = [];
  const flush = () => {
    if (cur.length) {
      pages.push({ words: cur, startMs: cur[0].startMs, endMs: cur[cur.length - 1].endMs });
    }
    cur = [];
  };
  const PHRASE_END = /[,;:]["')\]]*$/;
  for (const w of words) {
    const prev = cur[cur.length - 1];
    const chars = cur.map((c) => c.text).join('').trim().length + w.text.length;
    const full = cur.length >= maxWords || chars > maxChars;
    if (prev && phrases && full && !SENTENCE_END.test(prev.text.trim())) {
      // full mid-phrase: end the page at its last phrase break and carry the rest over
      let j = cur.length - 1;
      while (j > 0 && !PHRASE_END.test(cur[j - 1].text.trim())) j--;
      if (j > 0) {
        const rest = cur.slice(j);
        cur = cur.slice(0, j);
        flush();
        cur = rest;
      } else {
        flush();
      }
    } else if (
      prev &&
      (full ||
        SENTENCE_END.test(prev.text.trim()) ||
        (phrases && cur.length >= 3 && PHRASE_END.test(prev.text.trim())) ||
        w.startMs - prev.endMs > pauseMs)
    ) {
      flush();
    }
    cur.push(w);
  }
  flush();
  // Bridge short gaps so captions don't flicker off between pages
  pages.forEach((p, i) => {
    const next = pages[i + 1];
    p.endMs = next && next.startMs - p.endMs <= pauseMs ? next.startMs : p.endMs + holdMs;
  });
  return pages;
};

export const Captions: React.FC<CaptionsProps> = ({
  src,
  karaoke = true,
  maxWords = 7,
  maxChars = 42,
  pauseMs = 600,
  holdMs = 400,
  phrases = false,
  bottom = 80,
  fontFamily = 'Inter, system-ui, sans-serif',
  fontSize = 46,
  fontWeight = 600,
  color = '#ffffff',
  highlightColor = '#fbbf24',
  background = 'rgba(0, 0, 0, 0.7)',
  maxWidth = '72%',
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const exists = useMemo(() => getStaticFiles().some((f) => f.name === src), [src]);
  const [words, setWords] = useState<CaptionWord[] | null>(null);
  const { delayRender, continueRender, cancelRender } = useDelayRender();
  const [handle] = useState(() => (exists ? delayRender(`Loading captions ${src}`) : null));

  const load = useCallback(async () => {
    if (handle === null) return;
    try {
      const res = await fetch(staticFile(src));
      setWords((await res.json()) as CaptionWord[]);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [src, handle, continueRender, cancelRender]);

  useEffect(() => {
    load();
  }, [load]);

  const pages = useMemo(
    () => (words ? paginate(words, maxWords, maxChars, pauseMs, holdMs, phrases) : []),
    [words, maxWords, maxChars, pauseMs, holdMs, phrases],
  );

  const nowMs = (frame / fps) * 1000;
  const page = pages.find((p) => nowMs >= p.startMs && nowMs < p.endMs);
  if (!page) return null;

  // Highlight stays on the latest word through the gaps between words
  const activeIndex = karaoke ? page.words.filter((w) => w.startMs <= nowMs).length - 1 : -1;
  const pageFrame = frame - (page.startMs / 1000) * fps;
  const opacity = interpolate(pageFrame, [0, 3], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: bottom, pointerEvents: 'none' }}
    >
      <div
        style={{
          opacity,
          maxWidth,
          padding: '14px 32px',
          borderRadius: 14,
          background,
          textAlign: 'center',
          fontFamily,
          fontSize,
          fontWeight,
          lineHeight: 1.3,
          color,
          whiteSpace: 'pre-wrap',
          textShadow: background === 'transparent' ? '0 2px 8px rgba(0,0,0,0.9)' : 'none',
        }}
      >
        {page.words.map((w, i) => (
          <span key={i} style={{ color: i === activeIndex ? highlightColor : color }}>
            {i === 0 ? w.text.trimStart() : w.text}
          </span>
        ))}
      </div>
    </AbsoluteFill>
  );
};
