import { AbsoluteFill, OffthreadVideo, staticFile, getStaticFiles } from 'remotion';
import { useTheme } from '../../config/theme';
import { Label } from '../../../../../lib/components';

interface DemoSectionProps {
  videoFile: string;
  label?: string;
  jiraRef?: string;
  startFrom?: number;
  playbackRate?: number;
}

export const DemoSection: React.FC<DemoSectionProps> = ({
  videoFile,
  label,
  jiraRef,
  startFrom = 0,
  playbackRate = 1,
}) => {
  const theme = useTheme();

  // Check if video file exists
  const staticFiles = getStaticFiles();
  const videoPath = `demos/${videoFile}`;
  const hasVideo = staticFiles.some((f) => f.name === videoPath);

  const styles = {
    container: {
      backgroundColor: 'transparent',
    },
    placeholder: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      fontFamily: theme.fonts.primary,
    },
    placeholderIcon: {
      fontSize: 80,
      marginBottom: 24,
      opacity: 0.5,
    },
    placeholderText: {
      color: theme.colors.textLight,
      fontSize: 24,
    },
    placeholderFile: {
      color: theme.colors.primary,
      fontSize: 20,
      marginTop: 8,
      fontFamily: theme.fonts.mono,
    },
  };

  return (
    <AbsoluteFill style={styles.container}>
      {hasVideo ? (
        <OffthreadVideo
          src={staticFile(videoPath)}
          startFrom={startFrom}
          playbackRate={playbackRate}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
          }}
        />
      ) : (
        <div style={styles.placeholder}>
          <div style={styles.placeholderIcon}>🎬</div>
          <div style={styles.placeholderText}>Video placeholder</div>
          <div style={styles.placeholderFile}>public/demos/{videoFile}</div>
        </div>
      )}

      {label && <Label text={label} jiraRef={jiraRef} position="bottom-left" />}
    </AbsoluteFill>
  );
};
