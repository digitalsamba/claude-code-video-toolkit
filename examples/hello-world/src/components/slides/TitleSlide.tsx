import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, getStaticFiles } from 'remotion';
import { useTheme } from '../../config/theme';
import { sprintConfig } from '../../config/sprint-config';

interface TitleSlideProps {
  logoFile?: string;
}

export const TitleSlide: React.FC<TitleSlideProps> = ({
  logoFile = 'images/logo.png',
}) => {
  const frame = useCurrentFrame();
  const theme = useTheme();
  const { info } = sprintConfig;

  // Scale up animation (0.95 -> 1.0)
  const scale = interpolate(frame, [0, 30], [0.95, 1], {
    extrapolateRight: 'clamp',
  });

  // Staggered fade timings (each element starts 6 frames after previous)
  const logoOpacity = interpolate(frame, [0, 18], [0, 1], { extrapolateRight: 'clamp' });
  const titleOpacity = interpolate(frame, [6, 24], [0, 1], { extrapolateRight: 'clamp' });
  const sprintOpacity = interpolate(frame, [12, 30], [0, 1], { extrapolateRight: 'clamp' });
  const dateOpacity = interpolate(frame, [18, 36], [0, 1], { extrapolateRight: 'clamp' });
  const sectionOpacity = interpolate(frame, [24, 42], [0, 1], { extrapolateRight: 'clamp' });
  const versionOpacity = interpolate(frame, [30, 48], [0, 1], { extrapolateRight: 'clamp' });

  const staticFiles = getStaticFiles();
  const hasLogo = staticFiles.some((f) => f.name === logoFile);

  const styles = {
    container: {
      backgroundColor: 'transparent',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: theme.fonts.primary,
    },
    title: {
      color: theme.colors.textDark,
      fontSize: 88,
      fontWeight: 700,
      margin: 0,
      textAlign: 'center' as const,
    },
    sprintLine: {
      color: theme.colors.textDark,
      fontSize: 52,
      fontWeight: 500,
      margin: '24px 0 10px 0',
    },
    date: {
      color: theme.colors.textLight,
      fontSize: 34,
      margin: '0 0 48px 0',
    },
    sprintName: {
      color: theme.colors.primary,
      fontWeight: 600,
    },
    sectionTitle: {
      color: theme.colors.textDark,
      fontSize: 48,
      fontWeight: 700,
      margin: '0 0 10px 0',
    },
    version: {
      color: theme.colors.textLight,
      fontSize: 38,
      fontWeight: 500,
      margin: 0,
    },
    logo: {
      width: 200,
      height: 200,
      marginBottom: 48,
    },
    logoPlaceholder: {
      width: 200,
      height: 200,
      marginBottom: 48,
      backgroundColor: theme.colors.primary,
      borderRadius: 32,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#fff',
      fontSize: 48,
      fontWeight: 700,
    },
  };

  const versionString = info.build
    ? `Version ${info.version} (BUILD ${info.build})`
    : `Version ${info.version}`;

  return (
    <AbsoluteFill style={styles.container}>
      <div style={{ transform: `scale(${scale})`, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        {hasLogo ? (
          <Img src={staticFile(logoFile)} style={{ ...styles.logo, opacity: logoOpacity }} />
        ) : (
          <div style={{ ...styles.logoPlaceholder, opacity: logoOpacity }}>
            {info.product.charAt(0)}
          </div>
        )}
        <h1 style={{ ...styles.title, opacity: titleOpacity }}>{info.product}</h1>
        <p style={{ ...styles.sprintLine, opacity: sprintOpacity }}>
          Sprint Review : <span style={styles.sprintName}>{info.name}</span>
        </p>
        <p style={{ ...styles.date, opacity: dateOpacity }}>{info.dateRange}</p>
        <p style={{ ...styles.sectionTitle, opacity: sectionOpacity }}>{info.platform}</p>
        <p style={{ ...styles.version, opacity: versionOpacity }}>{versionString}</p>
      </div>
    </AbsoluteFill>
  );
};
