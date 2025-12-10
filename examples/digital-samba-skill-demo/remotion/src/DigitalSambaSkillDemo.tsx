import { AbsoluteFill, Series, Audio, staticFile, Sequence, OffthreadVideo, interpolate, useCurrentFrame, Img } from "remotion";
import { TitleSlide } from "./components/TitleSlide";
import { ProblemSlide } from "./components/ProblemSlide";
import { SolutionIntro } from "./components/SolutionIntro";
import { SkillInstallDemo } from "./components/SkillInstallDemo";
import { BuildingApp } from "./components/BuildingApp";
import { AppWalkthrough } from "./components/AppWalkthrough";
import { SummaryStats } from "./components/SummaryStats";
import { CTASlide } from "./components/CTASlide";
import { AnimatedBackground } from "./components/AnimatedBackground";

// Timeline based on VIDEO-SPEC.md (30fps)
// Extended CTA to accommodate voiceover (~173s total)
const SCENE_DURATIONS = {
  title: 240,        // 0:00 - 0:08 (8s)
  problem: 300,      // 0:08 - 0:18 (10s)
  solution: 210,     // 0:18 - 0:25 (7s)
  install: 900,      // 0:25 - 0:55 (30s) - extended +10s
  build: 900,        // 0:55 - 1:25 (30s)
  walkthrough: 1500, // 1:25 - 2:15 (50s)
  summary: 210,      // 2:15 - 2:22 (7s)
  cta: 930,          // 2:22 - 2:53 (31s) - reduced to maintain total
};

// Total duration in frames
const TOTAL_DURATION = Object.values(SCENE_DURATIONS).reduce((a, b) => a + b, 0);

// Vignette overlay - subtle cinematic darkening around edges
const Vignette: React.FC = () => {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: `radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.4) 100%)`,
        pointerEvents: "none",
      }}
    />
  );
};

// Logo watermark - top-left corner
const LogoWatermark: React.FC = () => {
  const frame = useCurrentFrame();

  // Fade in after title slide (frame 240)
  const opacity = interpolate(
    frame,
    [240, 270],
    [0, 0.6],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        position: "absolute",
        top: 24,
        left: 24,
        display: "flex",
        alignItems: "center",
        gap: 10,
        opacity,
      }}
    >
      <Img
        src={staticFile("images/ds-logo.png")}
        style={{
          width: 32,
          height: 32,
          objectFit: "contain",
        }}
      />
      <span
        style={{
          fontSize: 14,
          fontWeight: 500,
          color: "rgba(255,255,255,0.5)",
          letterSpacing: "0.5px",
        }}
      >
        Digital Samba
      </span>
    </div>
  );
};

// Narrator PiP component - rounded rectangle in corner
const NarratorPiP: React.FC = () => {
  const frame = useCurrentFrame();

  // Fade in over first 15 frames, fade out over last 15 frames
  const opacity = interpolate(
    frame,
    [0, 15, 4140, 4160], // ~138s = 4140 frames at 30fps
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        position: "absolute",
        bottom: 40,
        right: 40,
        width: 320,
        height: 180,
        borderRadius: 16,
        overflow: "hidden",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        border: "2px solid rgba(255,255,255,0.1)",
        opacity,
      }}
    >
      <OffthreadVideo
        src={staticFile("narrator.mp4")}
        style={{
          width: "100%",
          height: "130%",
          objectFit: "cover",
          objectPosition: "center top",
        }}
        muted
      />
      {/* Gradient fade at bottom to soften any remaining hand visibility */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 40,
          background: "linear-gradient(to bottom, transparent, rgba(0,0,0,0.8))",
          pointerEvents: "none",
        }}
      />
    </div>
  );
};

export const DigitalSambaSkillDemo: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0a0a0a",
        fontFamily: "Inter, system-ui, sans-serif",
      }}
    >
      <AnimatedBackground />

      {/* Background Music - plays throughout at low volume */}
      <Audio
        src={staticFile("audio/background-music.mp3")}
        volume={0.12}
        startFrom={0}
      />

      {/* Voiceover - starts at frame 120 (4 seconds in) */}
      <Sequence from={120}>
        <Audio
          src={staticFile("audio/voiceover.mp3")}
          volume={1}
        />
      </Sequence>

      {/* Narrator PiP - synced with voiceover start */}
      <Sequence from={120}>
        <NarratorPiP />
      </Sequence>

      {/* Professional overlays */}
      <Vignette />
      <LogoWatermark />

      <Series>
        <Series.Sequence durationInFrames={SCENE_DURATIONS.title}>
          <TitleSlide />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.problem}>
          <ProblemSlide />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.solution}>
          <SolutionIntro />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.install}>
          <SkillInstallDemo />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.build}>
          <BuildingApp />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.walkthrough}>
          <AppWalkthrough />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.summary}>
          <SummaryStats />
        </Series.Sequence>

        <Series.Sequence durationInFrames={SCENE_DURATIONS.cta}>
          <CTASlide />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
