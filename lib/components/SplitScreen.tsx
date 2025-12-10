/**
 * SplitScreen - Side-by-side video comparison layout
 *
 * Displays two videos side by side with labels. Useful for showing
 * mobile vs desktop, before/after, or parallel features.
 */

import { AbsoluteFill, OffthreadVideo, staticFile, getStaticFiles } from 'remotion';
import { useTheme } from '../theme';
import { Label } from './Label';

export interface SplitScreenProps {
  leftVideo: string;
  rightVideo: string;
  leftLabel?: string;
  rightLabel?: string;
  bottomLabel?: string;
  jiraRef?: string;
  /** Shared start offset for both videos (frames) */
  startFrom?: number;
  /** Individual start offset for left video (frames) */
  leftStartFrom?: number;
  /** Individual start offset for right video (frames) */
  rightStartFrom?: number;
  /** Playback speed multiplier */
  playbackRate?: number;
  /** Path prefix for video files. Default: 'demos/' */
  videoPath?: string;
}

interface VideoOrPlaceholderProps {
  videoFile: string;
  videoPath: string;
  startFrom: number;
  playbackRate: number;
}

const VideoOrPlaceholder: React.FC<VideoOrPlaceholderProps> = ({
  videoFile,
  videoPath,
  startFrom,
  playbackRate,
}) => {
  const theme = useTheme();
  const staticFiles = getStaticFiles();
  const fullPath = `${videoPath}${videoFile}`;
  const hasVideo = staticFiles.some((f) => f.name === fullPath);

  if (hasVideo) {
    return (
      <OffthreadVideo
        src={staticFile(fullPath)}
        startFrom={startFrom}
        playbackRate={playbackRate}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
        }}
      />
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        fontFamily: theme.fonts.primary,
      }}
    >
      <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.5 }}>🎬</div>
      <div style={{ color: theme.colors.textLight, fontSize: 16 }}>Video placeholder</div>
      <div
        style={{
          color: theme.colors.primary,
          fontSize: 14,
          marginTop: 8,
          fontFamily: theme.fonts.mono,
        }}
      >
        {videoFile}
      </div>
    </div>
  );
};

export const SplitScreen: React.FC<SplitScreenProps> = ({
  leftVideo,
  rightVideo,
  leftLabel = 'Left',
  rightLabel = 'Right',
  bottomLabel,
  jiraRef,
  startFrom = 0,
  leftStartFrom,
  rightStartFrom,
  playbackRate = 1,
  videoPath = 'demos/',
}) => {
  const theme = useTheme();

  // Allow individual overrides, fall back to shared startFrom
  const leftOffset = leftStartFrom ?? startFrom;
  const rightOffset = rightStartFrom ?? startFrom;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: 'transparent',
        display: 'flex',
        flexDirection: 'row',
      }}
    >
      {/* Left panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
      >
        <VideoOrPlaceholder
          videoFile={leftVideo}
          videoPath={videoPath}
          startFrom={leftOffset}
          playbackRate={playbackRate}
        />
        <div
          style={{
            position: 'absolute',
            top: 24,
            left: 24,
            backgroundColor: theme.colors.bgOverlay,
            color: theme.colors.textDark,
            padding: '14px 24px',
            borderRadius: theme.borderRadius.md,
            fontFamily: theme.fonts.primary,
            fontSize: 32,
            fontWeight: 500,
            boxShadow: `0 4px 12px ${theme.colors.shadow}`,
          }}
        >
          {leftLabel}
        </div>
      </div>

      {/* Divider */}
      <div
        style={{
          width: 4,
          backgroundColor: theme.colors.divider,
        }}
      />

      {/* Right panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
      >
        <VideoOrPlaceholder
          videoFile={rightVideo}
          videoPath={videoPath}
          startFrom={rightOffset}
          playbackRate={playbackRate}
        />
        <div
          style={{
            position: 'absolute',
            top: 24,
            left: 24,
            backgroundColor: theme.colors.bgOverlay,
            color: theme.colors.textDark,
            padding: '14px 24px',
            borderRadius: theme.borderRadius.md,
            fontFamily: theme.fonts.primary,
            fontSize: 32,
            fontWeight: 500,
            boxShadow: `0 4px 12px ${theme.colors.shadow}`,
          }}
        >
          {rightLabel}
        </div>
      </div>

      {/* Bottom label */}
      {bottomLabel && <Label text={bottomLabel} jiraRef={jiraRef} position="bottom-left" />}
    </AbsoluteFill>
  );
};
