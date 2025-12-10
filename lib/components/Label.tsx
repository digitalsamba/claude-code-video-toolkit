/**
 * Label - Floating label badge for demos and scenes
 *
 * Displays a label with optional JIRA/ticket reference.
 * Typically positioned in a corner of the video frame.
 */

import { useTheme } from '../theme';

export type LabelPosition = 'bottom-left' | 'bottom-right' | 'top-left' | 'top-right';
export type LabelSize = 'sm' | 'md' | 'lg';

export interface LabelProps {
  text: string;
  jiraRef?: string;
  position?: LabelPosition;
  size?: LabelSize;
}

const SIZE_STYLES = {
  sm: { fontSize: 24, padding: '12px 20px', jiraSize: 18, gap: 12 },
  md: { fontSize: 32, padding: '16px 32px', jiraSize: 24, gap: 16 },
  lg: { fontSize: 40, padding: '20px 40px', jiraSize: 28, gap: 20 },
};

const POSITION_STYLES: Record<LabelPosition, React.CSSProperties> = {
  'top-left': { top: 24, left: 24 },
  'top-right': { top: 24, right: 24 },
  'bottom-left': { bottom: 48, left: 48 },
  'bottom-right': { bottom: 48, right: 48 },
};

export const Label: React.FC<LabelProps> = ({
  text,
  jiraRef,
  position = 'bottom-left',
  size = 'md',
}) => {
  const theme = useTheme();
  const s = SIZE_STYLES[size];

  return (
    <div
      style={{
        position: 'absolute',
        ...POSITION_STYLES[position],
        backgroundColor: theme.colors.bgOverlay,
        color: theme.colors.textDark,
        padding: s.padding,
        borderRadius: theme.borderRadius.md,
        fontFamily: theme.fonts.primary,
        fontSize: s.fontSize,
        fontWeight: 500,
        boxShadow: `0 4px 12px ${theme.colors.shadow}`,
        display: 'flex',
        alignItems: 'center',
        gap: s.gap,
      }}
    >
      {text}
      {jiraRef && (
        <span
          style={{
            color: theme.colors.primary,
            fontSize: s.jiraSize,
          }}
        >
          {jiraRef}
        </span>
      )}
    </div>
  );
};
