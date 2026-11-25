import React from 'react';
import { PANEL, R, RFONT } from './theme';
import { RIcon, IconName } from './icons';

// Side panel shell (src/features/side-panels/components/styled.tsx, SidePanelHeader.tsx): paper, 1px divider, r8;
// header in h3 with pin and close on the right.
export const SidePanel: React.FC<{ title: string; children?: React.ReactNode }> = ({ title, children }) => (
  <div
    style={{
      position: 'absolute',
      left: PANEL.x,
      top: PANEL.y,
      width: PANEL.w,
      height: PANEL.h,
      background: R.paper,
      border: `1px solid ${R.divider}`,
      borderRadius: 8,
      overflow: 'hidden',
      boxSizing: 'border-box',
      fontFamily: RFONT,
      color: R.text,
    }}
  >
    <div style={{ position: 'absolute', left: 20, top: 14, fontSize: 24, lineHeight: 1.4 }}>{title}</div>
    <div style={{ position: 'absolute', right: 12, top: 18, display: 'flex', gap: 10 }}>
      <RIcon name="pin" size={26} color="rgba(0,0,0,0.6)" />
      <RIcon name="close" size={26} color="rgba(0,0,0,0.6)" />
    </div>
    {children}
  </div>
);

// Text RoundedButton with a start icon ("Create new", "Add content"): radius 25, 1rem, weight 400
export const RoundedButton: React.FC<{ x: number; y: number; icon: IconName; label: string; hover?: boolean; pressed?: number }> = ({ x, y, icon, label, hover, pressed = 0 }) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      height: 40,
      padding: '0 16px 0 10px',
      borderRadius: 25,
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      color: R.primary,
      fontSize: 16,
      background: hover ? 'rgba(55,113,224,0.1)' : 'transparent',
      transform: `scale(${1 - 0.04 * pressed})`,
    }}
  >
    <RIcon name={icon} size={26} color={R.primary} />
    {label}
  </div>
);

// Room switch (src/components/Switch.tsx): 36x22, thumb 16. Off: white track, #c5c5c5 border and thumb. On: primary.
export const RSwitch: React.FC<{ on: number; hover?: boolean }> = ({ on, hover }) => {
  const lit = on > 0.5;
  return (
    <div
      style={{
        position: 'relative',
        width: 36,
        height: 22,
        borderRadius: 36,
        boxSizing: 'border-box',
        background: lit ? R.primary : '#fff',
        border: `1px solid ${lit ? R.primary : R.divider}`,
        boxShadow: hover ? '0 0 0 8px rgba(55,113,224,0.12)' : 'none',
      }}
    >
      <div style={{ position: 'absolute', left: 2 + 14 * on, top: 2, width: 16, height: 16, borderRadius: 8, background: lit ? '#fff' : R.divider }} />
    </div>
  );
};

// Text input as the room styles it: r4, 1px divider border, padding 7px 12px, 1rem
export const Input: React.FC<{ w: number | string; value?: string; placeholder?: string; focused?: boolean; end?: React.ReactNode; style?: React.CSSProperties }> = ({ w, value, placeholder, focused, end, style }) => (
  <div
    style={{
      width: w,
      height: 38,
      boxSizing: 'border-box',
      border: focused ? `2px solid ${R.primary}` : `1px solid ${R.divider}`,
      borderRadius: 4,
      padding: focused ? '0 11px' : '0 12px',
      display: 'flex',
      alignItems: 'center',
      fontSize: 16,
      color: value ? R.text : 'rgba(0,0,0,0.42)',
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      gap: 8,
      ...style,
    }}
  >
    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>{value || placeholder}</span>
    {end}
  </div>
);

// MUI dialog shell at a set position; actions render UPPERCASE
export const RDialog: React.FC<{ x: number; y: number; w: number; title: string; opacity: number; actions: { label: string; primary?: boolean; disabled?: boolean; hover?: boolean; pressed?: number }[]; children: React.ReactNode }> = ({ x, y, w, title, opacity, actions, children }) =>
  opacity <= 0 ? null : (
    <>
      <div style={{ position: 'absolute', inset: 0, background: `rgba(0,0,0,${0.5 * opacity})`, zIndex: 30 }} />
      <div
        style={{
          position: 'absolute',
          left: x,
          top: y,
          width: w,
          background: R.paper,
          borderRadius: 4,
          boxShadow: '0 11px 15px -7px rgba(0,0,0,0.2), 0 24px 38px 3px rgba(0,0,0,0.14)',
          opacity,
          transform: `scale(${0.94 + 0.06 * opacity})`,
          fontFamily: RFONT,
          color: R.text,
          zIndex: 31,
        }}
      >
        <div style={{ padding: '24px 24px 8px', fontSize: 20, fontWeight: 500 }}>{title}</div>
        <div style={{ position: 'relative', padding: '8px 24px 12px' }}>{children}</div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '8px 16px 16px' }}>
          {actions.map((a) => (
            <div
              key={a.label}
              style={{
                padding: '6px 10px',
                borderRadius: 4,
                fontSize: 14,
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.02857em',
                color: a.disabled ? 'rgba(0,0,0,0.26)' : a.primary ? R.primary : R.grey600,
                background: a.hover ? 'rgba(55,113,224,0.1)' : 'transparent',
                transform: `scale(${1 - 0.05 * (a.pressed ?? 0)})`,
              }}
            >
              {a.label}
            </div>
          ))}
        </div>
      </div>
    </>
  );
