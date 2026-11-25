import React from 'react';
import { Img, staticFile } from 'remotion';
import ICONS from './icons.json';

// The Digital Samba dashboard as static React, from react-dashboard origin/master 0177aab4 (2026-11-11):
// components/Layouts (Header, SideMenu), pages/SessionDetails (SessionDetails, DetailsTable, ParticipantsTable,
// ExportButton) and components/ThemeProvider. Strings are from src/i18n/locales/en/common.js (traced for release 651).
export const D = {
  text: '#323E66',
  text2: '#808699',
  primary: '#3771e0',
  head: '#EBF1FC',
  divider: '#c5c5c5',
  paper: '#fff',
};
export const DFONT = 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
export const DW = 1440;
export const DH = 810;
export const HEADER_H = 72;
export const SIDE_W = 250;

type IconName = keyof typeof ICONS;
export const DIcon: React.FC<{ name: IconName; size?: number; color?: string }> = ({ name, size = 24, color = 'currentColor' }) => (
  <svg viewBox={ICONS[name].viewBox} width={size} height={size} fill={color} style={{ flexShrink: 0 }}>
    {ICONS[name].paths.map((d, i) => (
      <path key={i} d={d} />
    ))}
  </svg>
);

const NAV: [string, IconName][] = [
  ['Rooms', 'Room'],
  ['Sessions', 'Session'],
  ['Team', 'Team'],
  ['Roles', 'Role'],
  ['Libraries', 'Library'],
  ['Recordings', 'Recording'],
  ['Webhooks', 'Webhook'],
  ['Tokens', 'Token'],
  ['Statistics', 'BarChart'],
  ['Documentation', 'Documentation'],
  ['System check', 'System'],
];

const Pill: React.FC<{ label: string; filled?: boolean; w?: number }> = ({ label, filled, w }) => (
  <div
    style={{
      width: w,
      height: 38,
      padding: w ? 0 : '0 22px',
      borderRadius: 50,
      boxSizing: 'border-box',
      border: `1px solid ${D.primary}`,
      background: filled ? D.primary : 'transparent',
      color: filled ? '#fff' : D.primary,
      fontWeight: 600,
      fontSize: 16,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}
  >
    {label}
  </div>
);

const Caret: React.FC<{ color?: string }> = ({ color = D.text }) => (
  <svg viewBox="0 0 24 24" width={24} height={24} fill={color}>
    <path d="m7 10 5 5 5-5z" />
  </svg>
);

export const DashShell: React.FC<{ active: string; children?: React.ReactNode }> = ({ active, children }) => (
  <div style={{ position: 'relative', width: DW, height: DH, background: D.paper, color: D.text, fontFamily: DFONT, fontSize: 16, lineHeight: 1.4, overflow: 'hidden' }}>
    {/* Header: logo left; Meet now, the team menu and the account menu right */}
    <div style={{ position: 'absolute', left: 0, top: 0, width: DW, height: HEADER_H, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 28px', boxSizing: 'border-box' }}>
      <Img src={staticFile('logo-color.svg')} style={{ height: 30 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 44 }}>
        <Pill label="Meet now" />
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          Acme <Caret />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div style={{ width: 36, height: 36, borderRadius: 18, background: D.head, color: D.primary, fontWeight: 600, fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>LH</div>
          <Caret />
        </div>
      </div>
    </div>
    {/* Side menu: vertical tabs, the selected one in the primary colour with a 2px right border */}
    <div style={{ position: 'absolute', left: 0, top: HEADER_H, width: SIDE_W, height: DH - HEADER_H, borderRight: `1px solid ${D.divider}`, boxSizing: 'border-box' }}>
      {NAV.map(([label, icon]) => {
        const on = label === active;
        return (
          <div key={label} style={{ height: 42, display: 'flex', alignItems: 'center', gap: 16, paddingLeft: 28, color: on ? D.primary : D.text, borderRight: on ? `2px solid ${D.primary}` : 'none', marginRight: -1 }}>
            <DIcon name={icon} size={22} />
            {label}
          </div>
        );
      })}
      <div style={{ position: 'absolute', left: 28, bottom: 24, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Pill label="Upgrade plan" filled w={194} />
        <Pill label="Need help?" w={194} />
        <Pill label="Online demo" w={194} />
      </div>
    </div>
    <div style={{ position: 'absolute', left: SIDE_W, top: HEADER_H, width: DW - SIDE_W, height: DH - HEADER_H }}>{children}</div>
  </div>
);

// ---------------------------------------------------------------------------
// Session details. Positions are in the shell's canvas (1440x810), exported for the cursor.
export const SESSION_ID = '4be1c6f0-93a7-4d0e-8f52-7a1d2c9e6b34';
const CX = SIDE_W + 48; // content left edge
const CR = DW - 48; // content right edge
export const EXPORT_BTN = { x: CR - 60, y: HEADER_H + 104, w: 120, h: 40 };

const Btn: React.FC<{ x: number; y: number; w: number; label: string; icon?: React.ReactNode; hover?: boolean; pressed?: number }> = ({ x, y, w, label, icon, hover, pressed = 0 }) => (
  <div
    style={{
      position: 'absolute',
      left: x - w / 2,
      top: y - 20,
      width: w,
      height: 40,
      borderRadius: 50,
      boxSizing: 'border-box',
      border: `1px solid ${D.primary}`,
      color: D.primary,
      background: hover ? 'rgba(55,113,224,0.08)' : 'transparent',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 8,
      fontWeight: 600,
      transform: `scale(${1 - 0.04 * pressed})`,
    }}
  >
    {icon}
    {label}
  </div>
);

const DETAILS: [string, string][] = [
  ['Status', 'Ended'],
  ['Room name', 'Wochenplanung Produkt'],
  ['Started (UTC)', '2026-11-05 09:00:12'],
  ['Ended (UTC)', '2026-11-05 09:42:51'],
  ['Duration', '42m 39s'],
  ['Participants (max)', '4'],
  ['Participants (total unique)', '4'],
];
const PEOPLE: [string, string, string, string, string][] = [
  ['a41f09c2', 'Lena Hoffmann', 'Desktop', '2026-11-05 09:00:12', '42m 39s'],
  ['c7e2b518', 'Marco Rossi', 'Desktop', '2026-11-05 09:00:40', '42m 11s'],
  ['0d93fa6e', 'Sofia Marino', 'Mobile', '2026-11-05 09:01:03', '41m 48s'],
  ['5b28e7d1', 'Jonas Becker', 'Desktop', '2026-11-05 09:02:17', '40m 34s'],
];
const COLS = [0, 190, 430, 610, 870];

export const SessionDetails: React.FC<{ exportHover?: boolean; exportPressed?: number }> = ({ exportHover, exportPressed }) => (
  <div style={{ position: 'absolute', left: -SIDE_W, top: -HEADER_H, width: DW, height: DH }}>
    <div style={{ position: 'absolute', left: CX, top: HEADER_H + 28, display: 'flex', alignItems: 'center', gap: 6, color: D.primary }}>
      <svg viewBox="0 0 24 24" width={16} height={16} fill={D.primary}>
        <path d="M11.67 3.87 9.9 2.1 0 12l9.9 9.9 1.77-1.77L3.54 12z" transform="translate(6 0)" />
      </svg>
      Back
    </div>
    <div style={{ position: 'absolute', left: CX, top: HEADER_H + 86, fontSize: 24 }}>Session ID: {SESSION_ID}</div>
    <Btn x={EXPORT_BTN.x - 136} y={EXPORT_BTN.y} w={100} label="Join" />
    <Btn x={EXPORT_BTN.x} y={EXPORT_BTN.y} w={EXPORT_BTN.w} label="Export" icon={<DIcon name="Export" size={20} />} hover={exportHover} pressed={exportPressed} />
    {DETAILS.map(([k, v], i) => (
      <div key={k} style={{ position: 'absolute', left: CX, top: HEADER_H + 152 + i * 34, width: 700, display: 'flex' }}>
        <div style={{ width: 280, color: D.text2 }}>{k}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>{v}</div>
      </div>
    ))}
    <div style={{ position: 'absolute', left: CX, top: HEADER_H + 410, fontSize: 24 }}>Participants</div>
    <div style={{ position: 'absolute', left: CX, top: HEADER_H + 466, width: CR - CX }}>
      <div style={{ position: 'relative', height: 48, background: D.head, borderRadius: 5 }}>
        {['Participant ID', 'Name', 'Device', 'Joined (UTC)', 'Duration'].map((h, i) => (
          <div key={h} style={{ position: 'absolute', left: 16 + COLS[i], top: 13, fontWeight: 600 }}>
            {h}
          </div>
        ))}
      </div>
      {PEOPLE.map((row, r) => (
        <div key={row[0]} style={{ position: 'relative', height: 48, borderBottom: `1px solid ${D.head}` }}>
          {row.map((c, i) => (
            <div key={i} style={{ position: 'absolute', left: 16 + COLS[i], top: 13, color: i === 0 ? D.primary : D.text }}>
              {c}
            </div>
          ))}
        </div>
      ))}
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Export dialog (ExportButton.js): maxWidth xs (444px), two small full-width selects, Cancel and Download.
export const DATASETS = ['Session transcript', 'Session summary', 'Public chat log', 'Q&A log', 'Polls log', 'Quizzes results', 'Logs', 'Recorder logs'];
const DLG = { w: 444, h: 318 };
export const DLG_X = (DW - DLG.w) / 2;
export const DLG_Y = (DH - DLG.h) / 2;
export const DATASET_SELECT = { x: DW / 2, y: DLG_Y + 148 };
export const DATASET_OPTION = (i: number) => ({ x: DW / 2 - 100, y: DLG_Y + 148 + 30 + 18 + i * 36 });
export const DOWNLOAD_BTN = { x: DLG_X + DLG.w - 24 - 56, y: DLG_Y + DLG.h - 36 };

const Select: React.FC<{ y: number; value?: string; placeholder: string; focused?: boolean; disabled?: boolean }> = ({ y, value, placeholder, focused, disabled }) => (
  <div
    style={{
      position: 'absolute',
      left: 24,
      top: y,
      width: DLG.w - 48,
      height: 40,
      boxSizing: 'border-box',
      borderRadius: 4,
      border: focused ? `2px solid ${D.primary}` : `1px solid ${D.divider}`,
      padding: focused ? '0 13px' : '0 14px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      color: value ? (disabled ? 'rgba(0,0,0,0.38)' : '#000') : 'rgba(0,0,0,0.6)',
    }}
  >
    {value ?? placeholder}
    <Caret color={disabled ? 'rgba(0,0,0,0.26)' : 'rgba(0,0,0,0.54)'} />
  </div>
);

export const ExportDialog: React.FC<{
  open: number; // 0-1
  dataset?: string;
  format?: string;
  menu?: number; // 0-1: the dataset menu
  hoverOption?: number;
  downloadHover?: boolean;
  downloadPressed?: number;
}> = ({ open, dataset, format, menu = 0, hoverOption, downloadHover, downloadPressed = 0 }) =>
  open <= 0 ? null : (
    <div style={{ position: 'absolute', left: -SIDE_W, top: -HEADER_H, width: DW, height: DH, fontFamily: DFONT, color: '#000' }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)', opacity: open }} />
      <div
        style={{
          position: 'absolute',
          left: DLG_X,
          top: DLG_Y,
          width: DLG.w,
          height: DLG.h,
          background: '#fff',
          borderRadius: 4,
          boxShadow: '0 11px 15px -7px rgba(0,0,0,0.2), 0 24px 38px 3px rgba(0,0,0,0.14)',
          opacity: open,
          transform: `scale(${0.96 + 0.04 * open})`,
        }}
      >
        <div style={{ position: 'absolute', left: 24, top: 20, fontSize: 24 }}>Export</div>
        <div style={{ position: 'absolute', left: 24, top: 68, width: DLG.w - 48 }}>Select the dataset and format you wish to export for this session.</div>
        <Select y={128} value={dataset} placeholder="Select dataset" focused={menu > 0} />
        <Select y={192} value={format} placeholder="Select format" disabled={!!format} />
        <div style={{ position: 'absolute', right: 24 + 112 + 8, top: DLG.h - 56, height: 40, padding: '0 12px', display: 'flex', alignItems: 'center', color: D.primary }}>Cancel</div>
        <div
          style={{
            position: 'absolute',
            right: 24,
            top: DLG.h - 56,
            width: 112,
            height: 40,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: dataset ? D.primary : 'rgba(0,0,0,0.26)',
            background: downloadHover ? 'rgba(55,113,224,0.08)' : 'transparent',
            fontWeight: 600,
            textTransform: 'uppercase',
            transform: `scale(${1 - 0.04 * downloadPressed})`,
          }}
        >
          Download
        </div>
        {menu > 0 && (
          <div
            style={{
              position: 'absolute',
              left: 24,
              top: 128 + 40,
              width: DLG.w - 48,
              padding: '8px 0',
              background: '#fff',
              borderRadius: 4,
              boxShadow: '0 5px 5px -3px rgba(0,0,0,0.2), 0 8px 10px 1px rgba(0,0,0,0.14)',
              opacity: menu,
              zIndex: 2,
            }}
          >
            {DATASETS.map((d, i) => (
              <div key={d} style={{ height: 36, padding: '0 16px', display: 'flex', alignItems: 'center', background: i === hoverOption ? 'rgba(0,0,0,0.04)' : 'transparent' }}>
                {d}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
