import { AbsoluteFill, Audio, Series, staticFile, getStaticFiles } from 'remotion';
import { ThemeProvider, defaultTheme } from './config/theme';
import { sprintConfig, seconds, totalDurationSeconds } from './config/sprint-config';
import { LughIntroScene, TitleCardScene, TicketScene, ChapterCard } from './components/scenes';

export const AIEngineeringReview: React.FC = () => {
  const { info, lugh, titleCard, chapters, tickets, audio } = sprintConfig;
  const staticFiles = getStaticFiles();

  const audioExists = (path: string | undefined) =>
    !!path && staticFiles.some((f) => f.name === path);

  const hasBackgroundMusic = audioExists(audio.backgroundMusicFile);

  return (
    <ThemeProvider theme={defaultTheme}>
      <AbsoluteFill style={{ backgroundColor: '#0f0d0b' }}>
        <Series>
          {/* Cold open — Lugh introduces himself */}
          <Series.Sequence durationInFrames={seconds(lugh.introDurationSeconds)}>
            {audioExists(lugh.introAudioFile) && (
              <Audio src={staticFile(lugh.introAudioFile!)} />
            )}
            <LughIntroScene
              portraitFile={lugh.portraitFile}
              nameplate="LUGH"
              tagline="Long arm of AI Engineering"
              captions={lugh.introCaptions}
              durationInFrames={seconds(lugh.introDurationSeconds)}
            />
          </Series.Sequence>

          {/* Sprint title card */}
          <Series.Sequence durationInFrames={seconds(titleCard.durationSeconds)}>
            {audioExists(titleCard.audioFile) && (
              <Audio src={staticFile(titleCard.audioFile!)} />
            )}
            <TitleCardScene
              team={info.team}
              sprintName={titleCard.title ?? `Sprint ${info.sprintName}`}
              subtitle={titleCard.subtitle}
              captions={titleCard.captions}
              durationInFrames={seconds(titleCard.durationSeconds)}
            />
          </Series.Sequence>

          {/* Ticket scenes — interleaved with chapter cards at their startTicketIndex */}
          {tickets.map((ticket, idx) => {
            const chapter = chapters?.find((c) => c.startTicketIndex === idx);
            return (
              <>
                {chapter && (
                  <Series.Sequence
                    key={`chapter-${chapter.label}`}
                    durationInFrames={seconds(chapter.durationSeconds)}
                  >
                    <ChapterCard
                      label={chapter.label}
                      sprintName={chapter.sprintName}
                      dateRange={chapter.dateRange}
                      durationInFrames={seconds(chapter.durationSeconds)}
                    />
                  </Series.Sequence>
                )}
                <Series.Sequence
                  key={ticket.key}
                  durationInFrames={seconds(ticket.durationSeconds)}
                >
                  {audioExists(ticket.audioFile) && (
                    <Audio src={staticFile(ticket.audioFile!)} />
                  )}
                  <TicketScene
                    ticket={ticket}
                    portraitFile={lugh.portraitFile}
                    index={idx}
                    total={tickets.length}
                    durationInFrames={seconds(ticket.durationSeconds)}
                  />
                </Series.Sequence>
              </>
            );
          })}

          {/* Outro — Lugh signs off */}
          <Series.Sequence durationInFrames={seconds(lugh.outroDurationSeconds)}>
            {audioExists(lugh.outroAudioFile) && (
              <Audio src={staticFile(lugh.outroAudioFile!)} />
            )}
            <LughIntroScene
              portraitFile={lugh.portraitFile}
              nameplate="LUGH"
              tagline="Back in a month"
              captions={lugh.outroCaptions}
              durationInFrames={seconds(lugh.outroDurationSeconds)}
            />
          </Series.Sequence>
        </Series>

        {/* Background music — low volume throughout */}
        {hasBackgroundMusic && (
          <Audio
            src={staticFile(audio.backgroundMusicFile!)}
            volume={audio.backgroundMusicVolume ?? 0.08}
          />
        )}
      </AbsoluteFill>
    </ThemeProvider>
  );
};

// Total expected duration — used by Root.tsx to size the Composition.
export const totalDurationInFrames = seconds(totalDurationSeconds);
