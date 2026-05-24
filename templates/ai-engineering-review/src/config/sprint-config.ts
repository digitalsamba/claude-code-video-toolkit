import type { SprintConfig, VideoConfig, CaptionChunk } from './types';

export const videoConfig: VideoConfig = {
  fps: 30,
  width: 1920,
  height: 1080,
};

const fps = videoConfig.fps;
export const seconds = (s: number) => Math.round(s * fps);

// Phrase-level captions. Each chunk is timed in frames RELATIVE to the scene's start.
// Phrase = 3–6 words, held 1.2–2s, with ~6–12 frames of gap between chunks for breath.
const cap = (text: string, fromSec: number, durSec: number): CaptionChunk => ({
  text,
  fromFrame: Math.round(fromSec * fps),
  durationInFrames: Math.round(durSec * fps),
});

export const sprintConfig: SprintConfig = {
  info: {
    team: 'AI Engineering',
    sprintName: 'Blue Moon',
    dateRange: '20 April – 4 May 2026',
    tagline: 'Four tickets. Four fixes. One sprint.',
  },

  lugh: {
    portraitFile: 'images/lugh-woodshop.png',
    introDurationSeconds: 20,
    outroDurationSeconds: 9,
    introAudioFile: 'audio/scenes/00-intro.mp3',
    outroAudioFile: 'audio/scenes/99-outro.mp3',
    // Intro voiceover (target ~21s). Captions start ~1.5s in so the nameplate
    // has time to establish first.
    // "Good morning. I'm Lugh — the first agent running on Longarm. It's a
    //  runtime AI Engineering built to keep bespoke, long-lived agents alive as
    //  Digital Samba's support and devops load grows. They say I'm a work in
    //  progress — Conal's words. My job is to surface the work he does that
    //  nobody else sees. Here's April."
    introCaptions: [
      cap('Good morning.', 0.46, 0.8),
      cap("I'm Lugh.", 1.42, 0.44),
      cap('First agent running on Longarm.', 2.42, 1.88),
      cap('A runtime AI Engineering built...', 4.78, 2.14),
      cap('...to keep bespoke, long-lived agents alive.', 6.68, 2.36),
      cap("They say I'm a work in progress.", 12.08, 1.64),
      cap("Conal's words.", 13.92, 0.58),
      cap('I surface the work he does...', 15.66, 1.3),
      cap('...that nobody else sees.', 16.7, 1.4),
      cap("Here's April.", 18.42, 0.8),
    ],
    // Outro voiceover (target ~10s):
    // "That's April. Most of it, nobody sees — that's the idea. See you at the
    //  end of May. Conal says I talk too much."
    outroCaptions: [
      cap("That's April.", 0.32, 0.92),
      cap('Most of it, nobody sees.', 1.4, 1.64),
      cap("That's the idea.", 3.12, 1.12),
      cap('See you at the end of May.', 4.62, 1.28),
      cap('Conal says I talk too much.', 6.14, 1.56),
    ],
  },

  titleCard: {
    durationSeconds: 6,
    title: 'April Review',
    subtitle: 'Seven tickets · Two sprints · Five subsystems',
    // Silent — let the text breathe, no narration. Captions omitted deliberately.
    captions: [],
  },

  chapters: [
    {
      label: 'Release 636',
      sprintName: 'Forgotten Customers',
      dateRange: '6 – 19 April 2026',
      durationSeconds: 3,
      startTicketIndex: 0,
    },
    {
      label: 'Sprint Blue Moon',
      sprintName: 'Before Anyone Noticed',
      dateRange: '20 April – 4 May 2026',
      durationSeconds: 3,
      startTicketIndex: 3,
    },
  ],

  tickets: [
    // ===== Release 636 (6 – 19 April) =====
    {
      key: 'SAMBA-8623',
      title: 'Microsoft Graph mail as an alternative to SMTP',
      subsystem: 'integrations',
      whatBroke: "OTTO Chemie (Netucate customer) couldn't send mail — policy blocked SMTP basic auth.",
      whatShipped: 'Microsoft Graph client-credentials flow. Mail flows. And a path to wean D-Link off.',
      durationSeconds: 16,
      audioFile: 'audio/scenes/samba-8623.mp3',
      // "OTTO Chemie — a Netucate customer — couldn't send mail because their
      //  policy blocked SMTP basic auth. Microsoft Graph unlocks them. And it's
      //  a path to wean D-Link off the old SambaLive transport too. Quiet strategic
      //  win."
      captions: [
        cap('OTTO Chemie — a Netucate customer —', 1.36, 1.42),
        cap("couldn't send mail.", 2.9, 1.1),
        cap('SMTP basic auth blocked by policy.', 4.66, 2.12),
        cap('Microsoft Graph unlocks them.', 6.98, 1.84),
        cap('And a path to wean D-Link off too.', 8.9, 2.04),
        cap('Quiet strategic win.', 13.24, 1.52),
      ],
    },
    {
      key: 'SAMBA-8644',
      title: 'Invitation time limits in Account Center',
      subsystem: 'account-center',
      whatBroke: 'Invitation time limits were API-only. Netucate had built their own workaround.',
      whatShipped: 'Native toggle in Account Center. Workaround retired.',
      durationSeconds: 11,
      audioFile: 'audio/scenes/samba-8644.mp3',
      // "Invitation time limits used to be API-only — Netucate had built a workaround.
      //  Now it's a native toggle in Account Center. Workaround retired."
      captions: [
        cap('Invitation time limits...', 0.14, 1.6),
        cap('...were API-only.', 2.04, 1.44),
        cap('Netucate built a workaround.', 3.28, 2.42),
        cap('Now a native toggle in Account Center.', 5.54, 2.36),
        cap('Workaround retired.', 8.04, 1.44),
      ],
    },
    {
      key: 'SAMBA-8757',
      title: 'German: Du → Sie across systemcheck pages',
      subsystem: 'locales',
      whatBroke: 'Consularia — among our heaviest users — saw informal "Du" on their German pages.',
      whatShipped: 'Formal "Sie" across every German string. Other locales untouched.',
      durationSeconds: 12,
      audioFile: 'audio/scenes/samba-8757.mp3',
      // "Consularia — among our heaviest users — had German systemcheck pages
      //  using informal Du. Hotfix swapped every string to formal Sie.
      //  Other locales untouched."
      // NOTE for voice gen: Qwen3 should pronounce "Du" as "doo" and "Sie" as "zee".
      // Use SSML phonemes or spell-out hints when running voiceover.
      captions: [
        cap('Consularia —', 0.28, 1.26),
        cap('...among our heaviest users...', 1.26, 1.64),
        cap('...had German pages using informal "Du".', 3.08, 2.68),
        cap('Hotfix to formal "Sie".', 7.8, 0.82),
        cap('Other locales untouched.', 9.4, 1.6),
      ],
    },

    // ===== Sprint Blue Moon (20 April – 4 May) =====
    {
      key: 'SAMBA-8782',
      title: 'Custom attendee roles missing Q&A permission',
      subsystem: 'sambalive',
      whatBroke: 'Keymeeting reported attendees saw the Q&A panel but had no input box.',
      whatShipped: 'Twenty roles patched on Keymeeting — and the same broken pattern on Netucate, quietly.',
      durationSeconds: 11,
      audioFile: 'audio/scenes/samba-8782.mp3',
      // "Keymeeting reported custom attendee roles missing their Q&A input box.
      //  We patched twenty roles — and the same broken pattern on Netucate,
      //  before they ran into it."
      captions: [
        cap('Keymeeting reported Q&A input...', 2.78, 1.3),
        cap('...missing from custom attendee roles.', 2.9, 2.6),
        cap('Twenty roles patched.', 5.02, 0.6),
        cap('Same broken pattern on Netucate.', 6.34, 1.94),
        cap('Patched — before they hit it.', 8.02, 1.3),
      ],
    },
    {
      key: 'SAMBA-8822',
      title: 'Q&A questions leaking — and anonymous',
      subsystem: 'sambalive',
      whatBroke: 'Keymeeting again — every participant saw every question. All marked anonymous.',
      whatShipped: 'Host gate restored. Names reattached. Pushed across the other OEMs before they hit it.',
      durationSeconds: 12,
      // note: actual audio 10.6s — left at 12 for a ~1.4s breathing room at scene end
      audioFile: 'audio/scenes/samba-8822.mp3',
      // "Keymeeting again. Q&A questions leaking to every participant, all
      //  anonymous. Host gate restored, names reattached. Pushed across the
      //  other OEMs before they surfaced it."
      captions: [
        cap('Keymeeting again.', 0.88, 0.66),
        cap('Q&A leaking to every participant.', 2.76, 1.64),
        cap('All of them anonymous.', 4.32, 1.06),
        cap('Host gate restored. Names reattached.', 5.52, 2.88),
        cap('Pushed across the other OEMs.', 8.14, 1.66),
      ],
    },
    {
      key: 'SAMBA-8823',
      title: 'Usage attribution inflated 7x',
      subsystem: 'sambalive',
      whatBroke: 'A 4-hour Keymeeting session reported as 7+. Usage attribution, platform-wide.',
      whatShipped: 'Rejoin handler fixed across every OEM. Dedupe query for legacy data — before Netucate saw the same numbers.',
      durationSeconds: 15,
      // note: actual audio 13s — 2s of dense scene padding
      audioFile: 'audio/scenes/samba-8823.mp3',
      // "A four-hour session reported as seven. Usage attribution — platform-wide.
      //  Keymeeting caught it. Rejoin handler fixed across every OEM. Dedupe
      //  query for legacy data — before Netucate saw the same numbers."
      captions: [
        cap('A 4-hour session reported as 7.', 0.3, 1.9),
        cap('Usage attribution — platform-wide.', 2.76, 2.3),
        cap('Keymeeting caught it.', 5.68, 0.62),
        cap('Rejoin handler fixed across every OEM.', 6.5, 2.72),
        cap('Dedupe query for the legacy data.', 9.92, 1.46),
        cap('Before Netucate saw it.', 11.12, 1.4),
      ],
    },
    {
      key: 'SAMBA-8829',
      title: 'Portrait MP4 stretched to 16:9',
      subsystem: 'scaleway',
      whatBroke: "Civi reported portrait MP4s stretched to 16:9 — our HLS serverless function forcing the wrong aspect.",
      whatShipped: 'Fixed and deployed across Civi and DS Embedded before it spread.',
      durationSeconds: 12,
      // note: actual audio 11.1s
      audioFile: 'audio/scenes/samba-8829.mp3',
      // "Civi reported portrait MP4s stretched to sixteen-by-nine. AI Engineering's
      //  HLS serverless function was forcing the wrong aspect ratio. Fixed and
      //  deployed across Civi and DS Embedded."
      captions: [
        cap('Civi reported portrait MP4s...', 0.28, 2.08),
        cap('...stretched to 16:9.', 2.12, 1.1),
        cap('Our HLS serverless function...', 5.06, 1.44),
        cap('...was forcing the wrong aspect ratio.', 6.28, 2.14),
        cap('Fixed across Civi and DS Embedded.', 9.48, 1.84),
      ],
    },
  ],

  audio: {
    backgroundMusicFile: 'audio/background-music.mp3',
    backgroundMusicVolume: 0.08,
  },

  captions: {
    enabled: true,
    style: 'inscribed',
  },
};

export const totalDurationSeconds =
  sprintConfig.lugh.introDurationSeconds +
  sprintConfig.titleCard.durationSeconds +
  (sprintConfig.chapters?.reduce((sum, c) => sum + c.durationSeconds, 0) ?? 0) +
  sprintConfig.tickets.reduce((sum, t) => sum + t.durationSeconds, 0) +
  sprintConfig.lugh.outroDurationSeconds;
