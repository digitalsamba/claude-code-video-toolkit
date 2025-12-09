import { AbsoluteFill, Series, Audio, staticFile, Sequence } from 'remotion';
import { ThemeProvider, defaultTheme } from './config/theme';
import { demoConfig, videoConfig, calculateTotalFrames } from './config/demo-config';
import { AnimatedBackground, Vignette, LogoWatermark, NarratorPiP } from './components/core';
import {
  TitleSlide,
  ProblemSlide,
  SolutionSlide,
  DemoSlide,
  StatsSlide,
  CTASlide,
} from './components/slides';
import type {
  TitleContent,
  ProblemContent,
  SolutionContent,
  DemoContent,
  StatsContent,
  CTAContent,
} from './config/types';

export const ProductDemo: React.FC = () => {
  const { fps } = videoConfig;
  const totalFrames = calculateTotalFrames(demoConfig, fps);

  // Calculate scene frame offsets for narrator timing
  let currentFrame = 0;
  const sceneOffsets = demoConfig.scenes.map((scene) => {
    const offset = currentFrame;
    currentFrame += scene.durationSeconds * fps;
    return offset;
  });

  return (
    <ThemeProvider theme={defaultTheme}>
      <AbsoluteFill
        style={{
          backgroundColor: defaultTheme.colors.bgLight,
          fontFamily: defaultTheme.fonts.primary,
        }}
      >
        <AnimatedBackground />

        {/* Background Music */}
        {demoConfig.audio?.backgroundMusicFile && (
          <Audio
            src={staticFile(demoConfig.audio.backgroundMusicFile)}
            volume={demoConfig.audio.backgroundMusicVolume || 0.12}
            startFrom={0}
          />
        )}

        {/* Voiceover */}
        {demoConfig.audio?.voiceoverFile && (
          <Sequence from={demoConfig.audio.voiceoverStartFrame || 0}>
            <Audio
              src={staticFile(demoConfig.audio.voiceoverFile)}
              volume={1}
            />
          </Sequence>
        )}

        {/* Narrator PiP */}
        {demoConfig.narrator?.enabled && (
          <Sequence from={demoConfig.narrator.startFrame || demoConfig.audio?.voiceoverStartFrame || 0}>
            <NarratorPiP
              config={demoConfig.narrator}
              totalFrames={totalFrames - (demoConfig.narrator.startFrame || 0)}
            />
          </Sequence>
        )}

        {/* Professional overlays */}
        <Vignette />
        {demoConfig.product.logo && (
          <LogoWatermark
            logoSrc={demoConfig.product.logo}
            label={demoConfig.product.name}
            fadeInFrame={demoConfig.scenes[0].durationSeconds * fps}
          />
        )}

        {/* Scenes */}
        <Series>
          {demoConfig.scenes.map((scene, i) => {
            const durationInFrames = scene.durationSeconds * fps;

            return (
              <Series.Sequence key={i} durationInFrames={durationInFrames}>
                {scene.type === 'title' && (
                  <TitleSlide content={scene.content as TitleContent} />
                )}
                {scene.type === 'problem' && (
                  <ProblemSlide content={scene.content as ProblemContent} />
                )}
                {scene.type === 'solution' && (
                  <SolutionSlide content={scene.content as SolutionContent} />
                )}
                {scene.type === 'demo' && (
                  <DemoSlide content={scene.content as DemoContent} />
                )}
                {scene.type === 'stats' && (
                  <StatsSlide content={scene.content as StatsContent} />
                )}
                {scene.type === 'cta' && (
                  <CTASlide
                    content={scene.content as CTAContent}
                    logoSrc={demoConfig.product.logo}
                  />
                )}
              </Series.Sequence>
            );
          })}
        </Series>
      </AbsoluteFill>
    </ThemeProvider>
  );
};
