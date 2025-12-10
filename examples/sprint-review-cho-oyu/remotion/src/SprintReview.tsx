import { AbsoluteFill, Audio, Series, Sequence, staticFile } from "remotion";
import { TitleSlide } from "./components/TitleSlide";
import { OverviewSlide } from "./components/OverviewSlide";
import { SummarySlide } from "./components/SummarySlide";
import { AppRenameSlide } from "./components/AppRenameSlide";
import { DemoSection } from "./components/DemoSection";
import { SplitScreen } from "./components/SplitScreen";
import { AnimatedBackground } from "./components/AnimatedBackground";
import { SlideTransition } from "./components/SlideTransition";
import { MakingOf } from "./components/MakingOf";
import { EndCredits } from "./components/EndCredits";
import { TimelapseDemo } from "./components/TimelapseDemo";
import { TimelapseDemoWithSkip } from "./components/TimelapseDemoWithSkip";

// Frame calculations at 30fps
const FPS = 30;
const seconds = (s: number) => s * FPS;

// Timeline - Sprint Review Cut (~2:27 / 147s)
// Section                  | Duration | Start  | Notes
// -------------------------|----------|--------|------------------
// Title                    | 5s       | 0:00   |
// Overview                 | 15s      | 0:05   |
// App Rename/Migration     | 15s      | 0:20   | SAMBA-8117
// Sleep/Lock Fix           | 21s      | 0:35   | 1.4x speed
// Background Screenshare   | 15s      | 0:56   | Skip middle (0-1min, 6-6:20)
// Screenshare Tile Timing  | 12.5s    | 1:11   | 375 frames
// Button State             | 24s      | 1:24   | 1.4x speed
// Recording Indicator      | 12s      | 1:48   | 1.4x speed
// Deep Links               | 16s      | 2:00   | 1.4x speed
// Summary                  | 12s      | 2:16   | Stats + screenshot
// [Making Of]              | [80s]    | [2:27] | Director's cut (commented out)
// Total: 147s (~2:27) without Making Of

export const SprintReview: React.FC = () => {
  return (
    <AbsoluteFill>
      {/* Persistent animated background */}
      <AnimatedBackground variant="subtle" />

      <Series>
        {/* 0:00 - Title Card */}
        <Series.Sequence durationInFrames={seconds(5)}>
          <SlideTransition durationInFrames={seconds(5)} style="zoom">
            <TitleSlide />
          </SlideTransition>
        </Series.Sequence>

        {/* 0:05 - Overview */}
        <Series.Sequence durationInFrames={seconds(15)}>
          <SlideTransition durationInFrames={seconds(15)} style="zoom">
            <OverviewSlide />
          </SlideTransition>
        </Series.Sequence>

        {/* 0:20 - App Rename / Legacy Migration */}
        <Series.Sequence durationInFrames={seconds(18)}>
          <SlideTransition durationInFrames={seconds(18)} style="zoom">
            <AppRenameSlide />
          </SlideTransition>
        </Series.Sequence>

        {/* 0:38 - Sleep/Lock Connection Fix (29s video at 2.5x = ~12s) */}
        <Series.Sequence durationInFrames={seconds(12)}>
          <SlideTransition durationInFrames={seconds(12)} style="blur-fade">
            <DemoSection
              videoFile="sleep-lock.mp4"
              label="Sleep/Lock Fix"
              jiraRef="SAMBA-8059"
              playbackRate={2.5}
            />
          </SlideTransition>
        </Series.Sequence>

        {/* 0:56 - Background Screenshare Connection Fix (with skip: 0-1min, skip, 6-6:20) */}
        <Series.Sequence durationInFrames={seconds(15)}>
          <SlideTransition durationInFrames={seconds(15)} style="blur-fade">
            <TimelapseDemoWithSkip
              label="Background Screenshare Fix"
              jiraRef="SAMBA-8094"
            />
          </SlideTransition>
        </Series.Sequence>

        {/* 1:32 - Screenshare Tile Timing (split screen, 375 frames = 12.5s) */}
        <Series.Sequence durationInFrames={375}>
          <SlideTransition durationInFrames={375} style="blur-fade">
            <SplitScreen
              leftVideo="tile-timing-phone.mp4"
              rightVideo="tile-timing-browser.mp4"
              leftLabel="iPhone"
              rightLabel="Remote Participant"
              bottomLabel="Screenshare Tile Timing"
              jiraRef="SAMBA-8053/8057"
              leftStartFrom={0}
              rightStartFrom={300}
            />
          </SlideTransition>
        </Series.Sequence>

        {/* 2:02 - Button State After Reconnection (33s at 2x = ~17s) */}
        <Series.Sequence durationInFrames={seconds(17)}>
          <SlideTransition durationInFrames={seconds(17)} style="blur-fade">
            <DemoSection
              videoFile="button-state.mp4"
              label="Button State Sync"
              jiraRef="SAMBA-8054"
              playbackRate={2}
            />
          </SlideTransition>
        </Series.Sequence>

        {/* 2:26 - Recording Indicator Cleanup (17s at 1.4x = ~12s) */}
        <Series.Sequence durationInFrames={seconds(12)}>
          <SlideTransition durationInFrames={seconds(12)} style="blur-fade">
            <DemoSection
              videoFile="recording-indicator.mp4"
              label="Recording Indicator"
              jiraRef="SAMBA-8056"
              playbackRate={1.4}
            />
          </SlideTransition>
        </Series.Sequence>

        {/* 2:38 - Deep Links (23s at 1.4x = ~16s) */}
        <Series.Sequence durationInFrames={seconds(16)}>
          <SlideTransition durationInFrames={seconds(16)} style="blur-fade">
            <DemoSection
              videoFile="deep-links.mp4"
              label="Deep Links"
              jiraRef="SAMBA-8112"
              playbackRate={1.4}
            />
          </SlideTransition>
        </Series.Sequence>

        {/* Summary (stats then screenshot overlay) */}
        <Series.Sequence durationInFrames={seconds(15)}>
          <SlideTransition durationInFrames={seconds(15)} style="zoom" transitionDuration={20}>
            <SummarySlide />
          </SlideTransition>
        </Series.Sequence>

        {/* End Credits - 32.5 seconds (ends at 2:50) */}
        <Series.Sequence durationInFrames={975}>
          <EndCredits />
        </Series.Sequence>

        {/* COMMENTED OUT - Director's Cut: The Making Of (80s)
        <Series.Sequence durationInFrames={seconds(80)}>
          <MakingOf />
        </Series.Sequence>
        */}
      </Series>

      {/* Voiceover audio track - starts at frame 120 (4 seconds in) */}
      <Sequence from={120}>
        <Audio src={staticFile("audio/voiceover.mp3")} />
      </Sequence>

      {/* Background music - low volume under the whole video */}
      <Audio src={staticFile("audio/background-music.mp3")} volume={0.15} />

      {/* Success chime on summary slide (starts at frame 3675 = 122.5s) */}
      <Sequence from={3675}>
        <Audio src={staticFile("audio/sfx-chime.mp3")} volume={0.5} />
      </Sequence>
    </AbsoluteFill>
  );
};
