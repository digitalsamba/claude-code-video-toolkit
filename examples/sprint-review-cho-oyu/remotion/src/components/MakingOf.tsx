import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

const styles = {
  container: {
    backgroundColor: "#0f172a",
    overflow: "hidden",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  scrollContainer: {
    position: "absolute" as const,
    left: "50%",
    transform: "translateX(-50%)",
    width: 1400,
    padding: "120px 100px",
  },
  section: {
    marginBottom: 120,
  },
  label: {
    color: "#ea580c",
    fontSize: 20,
    fontWeight: 600,
    letterSpacing: 3,
    textTransform: "uppercase" as const,
    marginBottom: 16,
  },
  heading: {
    color: "#f8fafc",
    fontSize: 52,
    fontWeight: 700,
    marginBottom: 24,
    lineHeight: 1.2,
  },
  paragraph: {
    color: "#94a3b8",
    fontSize: 32,
    lineHeight: 1.7,
    textAlign: "left" as const,
  },
  highlight: {
    color: "#f8fafc",
    fontWeight: 500,
  },
  accent: {
    color: "#ea580c",
    fontWeight: 600,
  },
  divider: {
    width: 80,
    height: 3,
    backgroundColor: "#ea580c",
    marginTop: 60,
    opacity: 0.6,
  },
  closing: {
    textAlign: "center" as const,
    marginTop: 40,
  },
  closingText: {
    color: "#64748b",
    fontSize: 28,
    fontStyle: "italic" as const,
    marginBottom: 30,
  },
  cta: {
    color: "#ea580c",
    fontSize: 36,
    fontWeight: 600,
  },
};

export const MakingOf: React.FC = () => {
  const frame = useCurrentFrame();

  // Director's cut - 80 seconds = 2400 frames
  // Hold at start, slow scroll, hold at end
  const scrollY = interpolate(
    frame,
    [0, 90, 2200, 2400],  // hold 3s at start, scroll, hold 6.7s at end
    [250, 250, -1950, -1950],
    { extrapolateRight: "clamp" }
  );

  // Fade in at start
  const fadeIn = interpolate(frame, [0, 60], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Slow fade out at end - 5 seconds
  const fadeOut = interpolate(frame, [2200, 2400], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={styles.container}>
      {/* Fade overlay at top - text disappears as it scrolls up */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: "25%",
          background: "linear-gradient(to bottom, #0f172a 0%, #0f172a 30%, transparent 100%)",
          zIndex: 10,
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          ...styles.scrollContainer,
          top: scrollY,
          opacity: fadeIn * fadeOut,
        }}
      >
        {/* Section 1: The Sprint */}
        <div style={styles.section}>
          <p style={styles.label}>The Sprint</p>
          <h2 style={styles.heading}>These fixes came from JIRA</h2>
          <p style={styles.paragraph}>
            Every bug fix in this review was pulled directly from our backlog via{" "}
            <span style={styles.accent}>JIRA MCP</span>. The AI reads the tickets,
            understands the context, and helps implement the solutions. The same
            workflow that created the fixes also created this video.
          </p>
          <div style={styles.divider} />
        </div>

        {/* Section 2: The Scaffolding */}
        <div style={styles.section}>
          <p style={styles.label}>The Scaffolding</p>
          <h2 style={styles.heading}>We learned how iOS apps are built</h2>
          <p style={styles.paragraph}>
            Before asking AI to help, we invested time understanding the{" "}
            <span style={styles.highlight}>fundamental structure</span> — how React Native
            bridges to Swift, how screenshare works at the native level, how the
            App Store pipeline flows. This scaffolding knowledge is what makes
            AI collaboration effective.
          </p>
          <div style={styles.divider} />
        </div>

        {/* Section 3: The Shift */}
        <div style={styles.section}>
          <p style={styles.label}>The Shift</p>
          <h2 style={styles.heading}>You don't need to write code</h2>
          <p style={styles.paragraph}>
            These mobile fixes, this video, this entire presentation — none of it
            required traditional coding. What it required was{" "}
            <span style={styles.highlight}>patience</span> and{" "}
            <span style={styles.highlight}>clear communication</span>. The AI writes
            the code. Your job is to understand the problem and describe the solution.
          </p>
          <div style={styles.divider} />
        </div>

        {/* Section 4: The Mindset */}
        <div style={styles.section}>
          <p style={styles.label}>The Mindset</p>
          <h2 style={styles.heading}>Patience over bravery</h2>
          <p style={styles.paragraph}>
            The instinct is to dive in and be bold. But with AI-assisted development,
            the real skill is <span style={styles.accent}>restraint</span>. Take time to
            understand the scaffolding. Build context before asking for changes. Let the
            AI work through problems step by step. The results speak for themselves.
          </p>
          <div style={styles.divider} />
        </div>

        {/* Section 5: The Invitation */}
        <div style={styles.section}>
          <p style={styles.label}>The Invitation</p>
          <h2 style={styles.heading}>This capability is yours too</h2>
          <p style={styles.paragraph}>
            <span style={styles.accent}>Claude Code</span>,{" "}
            <span style={styles.accent}>JIRA MCP</span>,{" "}
            <span style={styles.accent}>Remotion skills</span> — these tools are ready
            for any team that wants to work this way. We're not gatekeeping. We're
            sharing what we've learned.
          </p>
        </div>

        {/* Closing */}
        <div style={styles.closing}>
          <p style={styles.closingText}>
            The future of development isn't about writing more code.
            <br />
            It's about understanding problems deeply enough to describe them clearly.
          </p>
          <p style={styles.cta}>Ask us how to get started.</p>
        </div>
      </div>
    </AbsoluteFill>
  );
};
