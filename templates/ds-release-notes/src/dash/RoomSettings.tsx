import React from 'react';
import { D, DFONT, DW, DH, HEADER_H, SIDE_W } from './Dashboard';

// Edit room, General section, from react-dashboard origin/master f312763f (SAMBA-9700):
// pages/Settings/Components/Form.js + Sections/General.js. One long form scrolled by the left navigation; the view
// is scrolled to the rows around the new switch. Strings from src/i18n/locales/en/common.js (traced for release 652).
const CX = SIDE_W + 32; // content left
const CR = DW - 32; // content right
const NAV_W = 188;
const FX = CX + NAV_W + 20; // form left
const LABEL_W = 357;
const CTRL_X = FX + LABEL_W + 32; // control column
const TOP = HEADER_H + 108; // first row under the sticky header

const NAV = ['General', 'Branding', 'Components', 'Content Library', 'Recordings', 'Restreaming', 'Chat', 'Q&A', 'Breakout rooms', 'Polls & Quizzes', 'Transcription', 'Join settings', 'Telephony', 'External stream'];

type Row = { label: string; desc: string; on: boolean; top: number };
export const ROWS: Row[] = [
  { label: 'Connection message', desc: 'When enabled, the "Connection message" status notification will be displayed in the session if a weak connection occurs.', on: true, top: TOP },
  { label: 'Local connection quality indicator', desc: 'When enabled, participants can see their local connection quality indicator in the room.', on: true, top: TOP + 121 },
  { label: 'Never crop stream', desc: 'When enabled, participant video streams display in full without cropping. Participants can override this from the in-room settings menu, but only for their own view and only for that session.', on: false, top: TOP + 225 },
  { label: 'Always show names on video tiles', desc: 'When enabled, names of participants will be always shown on their tiles and not only on hover.', on: false, top: TOP + 363 },
  { label: 'Screen watermark', desc: 'When enabled, a repeated text watermark is displayed across the screen to discourage unauthorized recording or sharing.', on: false, top: TOP + 467 },
];
export const NAMES_ROW = 3;
/** Centre of the new switch, in the 1440x810 canvas */
export const NAMES_SWITCH = { x: CTRL_X + 19, y: ROWS[NAMES_ROW].top + 8 + 11 };
export const UPDATE_BTN = { x: CR - 72, y: HEADER_H + 44 };

// MUI Switch as themed in the dashboard: 38x22, white 16px thumb, track #323E66 off, primary on
export const DSwitch: React.FC<{ on: number; hover?: boolean }> = ({ on, hover }) => (
  <div style={{ position: 'relative', width: 38, height: 22, borderRadius: 11, background: on > 0.5 ? D.primary : D.text, transition: 'none', boxShadow: hover ? `0 0 0 8px rgba(55,113,224,0.12)` : 'none' }}>
    <div style={{ position: 'absolute', left: 3 + 16 * on, top: 3, width: 16, height: 16, borderRadius: 8, background: '#fff' }} />
  </div>
);

const Btn: React.FC<{ label: string; filled?: boolean; hover?: boolean; pressed?: number }> = ({ label, filled, hover, pressed = 0 }) => (
  <div
    style={{
      height: 40,
      padding: '0 22px',
      borderRadius: 50,
      border: `1px solid ${D.primary}`,
      background: filled ? (hover ? '#2f62c4' : D.primary) : hover ? 'rgba(55,113,224,0.08)' : 'transparent',
      color: filled ? '#fff' : D.primary,
      fontSize: 16,
      display: 'flex',
      alignItems: 'center',
      boxSizing: 'border-box',
      transform: `scale(${1 - 0.04 * pressed})`,
    }}
  >
    {label}
  </div>
);

export const EditRoom: React.FC<{ namesOn: number; switchHover?: boolean; updateHover?: boolean; updatePressed?: number; highlight?: number }> = ({ namesOn, switchHover, updateHover, updatePressed, highlight = 0 }) => (
  <div style={{ position: 'absolute', left: -SIDE_W, top: -HEADER_H, width: DW, height: DH, fontFamily: DFONT, color: D.text, pointerEvents: 'none' }}>
    {/* Sticky header: title left, actions right */}
    <div style={{ position: 'absolute', left: CX, top: HEADER_H + 20, fontSize: 28 }}>Edit room</div>
    <div style={{ position: 'absolute', right: DW - CR, top: HEADER_H + 24, display: 'flex', gap: 12 }}>
      <Btn label="Join" />
      <Btn label="Cancel" />
      <Btn label="Update room" filled hover={updateHover} pressed={updatePressed} />
    </div>
    {/* Section navigation */}
    <div style={{ position: 'absolute', left: CX, top: TOP - 4, width: NAV_W }}>
      {NAV.map((n) => (
        <div key={n} style={{ height: 38, fontSize: 15, color: n === 'General' ? D.primary : D.text }}>
          {n}
        </div>
      ))}
    </div>
    {/* Rows */}
    {ROWS.map((r, i) => {
      const hl = i === NAMES_ROW ? highlight : 0;
      return (
        <div key={r.label} style={{ position: 'absolute', left: FX, top: r.top, width: CR - FX }}>
          {hl > 0 && <div style={{ position: 'absolute', left: -16, top: -12, right: 0, height: 96, borderRadius: 8, background: `rgba(55,113,224,${0.08 * hl})` }} />}
          <div style={{ position: 'relative', width: LABEL_W }}>
            <div style={{ fontWeight: 600, fontSize: 16, lineHeight: '24px', marginBottom: 8 }}>{r.label}</div>
            <div style={{ fontSize: 14, lineHeight: 1.2, color: D.text2 }}>{r.desc}</div>
          </div>
          <div style={{ position: 'absolute', left: CTRL_X - FX, top: 8 }}>
            <DSwitch on={i === NAMES_ROW ? namesOn : r.on ? 1 : 0} hover={i === NAMES_ROW && switchHover} />
          </div>
        </div>
      );
    })}
  </div>
);

// notistack success snackbar, bottom left
export const Snackbar: React.FC<{ text: string; opacity: number }> = ({ text, opacity }) =>
  opacity <= 0 ? null : (
    <div
      style={{
        position: 'absolute',
        left: 20 - SIDE_W,
        bottom: 20,
        padding: '14px 20px',
        borderRadius: 4,
        background: '#43a047',
        color: '#fff',
        fontFamily: DFONT,
        fontSize: 15,
        boxShadow: '0 3px 5px -1px rgba(0,0,0,0.2), 0 6px 10px 0 rgba(0,0,0,0.14)',
        opacity,
        transform: `translateY(${(1 - opacity) * 16}px)`,
      }}
    >
      {text}
    </div>
  );
