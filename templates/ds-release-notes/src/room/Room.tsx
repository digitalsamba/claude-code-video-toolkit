import React from 'react';
import { Img, staticFile } from 'remotion';
import { APP_H, APP_W, PANEL, R, RFONT, TOOLBAR_PLACEMENT, TOOLBAR_W, TOPBAR_H } from './theme';
import { RIcon, IconName } from './icons';
import { Person, initials } from './people';

// ---------------------------------------------------------------------------
// Tiles (src/features/layout/tiles/Tile.tsx, streaming/components/*)
// ---------------------------------------------------------------------------
// badge: opacity of the name chip (0 = hover-only mode, hidden); menu: opacity of the mic + tile-menu buttons, which
// show with the names when 'Always show names on video tiles' is on (StreamBadge.tsx, TileControlButtons.tsx)
export type TileState = { person: Person; micOff?: boolean; camOff?: boolean; speaking?: boolean; zoom?: number; badge?: number; menu?: number };

export const Tile: React.FC<TileState & { w: number; h: number }> = ({ person, micOff, camOff, speaking, w, h, zoom = 1, badge = 1, menu }) => {
  const showVideo = person.img && !camOff;
  const avatar = Math.min(120, Math.max(30, h * 0.42));
  return (
    <div style={{ position: 'relative', width: w, height: h, background: R.tile, borderRadius: 8, overflow: 'hidden' }}>
      {showVideo ? (
        <Img
          src={staticFile(`people/${person.img}.jpg`)}
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${zoom})` }}
        />
      ) : (
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            width: avatar,
            height: avatar,
            marginLeft: -avatar / 2,
            marginTop: -avatar / 2,
            borderRadius: '50%',
            background: person.color,
            color: '#fff',
            fontFamily: RFONT,
            fontSize: avatar * 0.38,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {initials(person.name)}
        </div>
      )}
      <div
        style={{
          position: 'absolute',
          top: 8,
          left: '50%',
          transform: 'translateX(-50%)',
          height: 26,
          padding: '0 14px',
          borderRadius: 13,
          background: 'rgba(0,0,0,0.7)',
          color: '#fff',
          fontFamily: RFONT,
          fontSize: 14,
          display: 'flex',
          alignItems: 'center',
          whiteSpace: 'nowrap',
          opacity: badge,
        }}
      >
        {person.you ? 'You' : person.name}
      </div>
      {[{ icon: (micOff ? 'mute' : 'mic') as IconName, color: speaking && !micOff ? R.speaking : '#fff', o: menu === undefined ? 1 : Math.max(menu, speaking ? 1 : 0) },
        ...(menu === undefined ? [] : [{ icon: 'settings1' as IconName, color: '#fff', o: menu }])].map((b, i) => (
        <div
          key={b.icon}
          style={{
            position: 'absolute',
            left: 8 + i * 40,
            bottom: 8,
            width: 32,
            height: 32,
            borderRadius: 16,
            background: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: b.o,
          }}
        >
          <RIcon name={b.icon} size={24} color={b.color} />
        </div>
      ))}
    </div>
  );
};

/** Where TileGrid puts each tile, for cursor targets */
export const gridRects = (n: number, x: number, y: number, w: number, h: number, cols: number, gap = 8) => {
  const rows = Math.ceil(n / cols);
  const tw = (w - gap * (cols - 1)) / cols;
  const th = Math.min((h - gap * (rows - 1)) / rows, tw * 0.75);
  const top = y + (h - (th * rows + gap * (rows - 1))) / 2;
  return Array.from({ length: n }, (_, i) => {
    const r = Math.floor(i / cols);
    const inRow = Math.min(cols, n - r * cols);
    const left = x + (w - (tw * inRow + gap * (inRow - 1))) / 2 + (i % cols) * (tw + gap);
    return { x: left, y: top + r * (th + gap), w: tw, h: th, cx: left + tw / 2, cy: top + r * (th + gap) + th / 2 };
  });
};

export const TileGrid: React.FC<{ tiles: TileState[]; x: number; y: number; w: number; h: number; cols: number; gap?: number }> = ({
  tiles,
  x,
  y,
  w,
  h,
  cols,
  gap = 8,
}) => {
  const rows = Math.ceil(tiles.length / cols);
  const tw = (w - gap * (cols - 1)) / cols;
  const th = Math.min((h - gap * (rows - 1)) / rows, tw * 0.75);
  const top = y + (h - (th * rows + gap * (rows - 1))) / 2;
  return (
    <>
      {tiles.map((t, i) => {
        const r = Math.floor(i / cols);
        const inRow = Math.min(cols, tiles.length - r * cols);
        const left = x + (w - (tw * inRow + gap * (inRow - 1))) / 2 + (i % cols) * (tw + gap);
        return (
          <div key={t.person.id} style={{ position: 'absolute', left, top: top + r * (th + gap) }}>
            <Tile {...t} w={tw} h={th} />
          </div>
        );
      })}
    </>
  );
};

// ---------------------------------------------------------------------------
// Top bar + toolbar (RoomTopBar.tsx, toolbar/styles.tsx, ToolbarNav.tsx)
// ---------------------------------------------------------------------------
const TOOL_GROUPS: IconName[][] = [
  ['camera', 'mic', 'share-screen'],
  ['participants', 'chat', 'qa', 'polls', 'raise-hand', 'recording-base', 'content-folder-1'],
  ['settings', 'phone-disconnect'],
];

export const ToolbarButton: React.FC<{ icon: IconName; active?: boolean; red?: boolean }> = ({ icon, active, red }) => (
  <div
    style={{
      width: 40,
      height: 40,
      borderRadius: 20,
      background: red ? R.red : active ? R.primaryLight : 'transparent',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}
  >
    <RIcon name={icon} size={32} color="#fff" />
  </div>
);

export const RoomChrome: React.FC<{
  activeTool?: IconName;
  recording?: boolean;
  roomName?: string;
  micOff?: boolean;
  camOff?: boolean;
  children?: React.ReactNode;
}> = ({ activeTool, recording, roomName, micOff, camOff, children }) => (
  <div style={{ position: 'relative', width: APP_W, height: APP_H, background: R.room, overflow: 'hidden', fontFamily: RFONT }}>
    {/* Top bar */}
    <Img src={staticFile('logo_mono.svg')} style={{ position: 'absolute', left: TOOLBAR_PLACEMENT === 'bottom' ? 16 : 104, top: 13, height: 24, filter: 'brightness(0) invert(1)' }} />
    {roomName && (
      <div style={{ position: 'absolute', left: 0, right: 0, top: 13, textAlign: 'center', color: '#fff', fontSize: 16, fontWeight: 700 }}>
        {roomName}
      </div>
    )}
    <div style={{ position: 'absolute', right: 8, top: 8, display: 'flex', gap: 8 }}>
      {[
        { icon: 'tiled-mode' as IconName, label: 'Tiled mode' },
        { icon: 'enter-fullscreen' as IconName, label: 'Full screen' },
      ].map((b) => (
        <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 14px', color: '#fff', fontSize: 14, fontWeight: 700 }}>
          <RIcon name={b.icon} size={26} color="#fff" />
          {b.label}
        </div>
      ))}
    </div>
    {recording && (
      <div
        style={{
          position: 'absolute',
          left: 20,
          top: 12,
          width: 56,
          height: 32,
          borderRadius: 8,
          background: R.red,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 2,
        }}
      >
        <RIcon name="rec" size={32} color="#fff" />
      </div>
    )}

    {/* Toolbar: bottom (the default for new teams) or left. Bottom: 88 high, 8px inset, groups left / centre / right */}
    <div
      style={
        TOOLBAR_PLACEMENT === 'bottom'
          ? { position: 'absolute', left: 8, right: 8, bottom: 8, height: TOOLBAR_W - 8, display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: '0 8px', boxSizing: 'border-box' }
          : { position: 'absolute', left: 8, top: TOPBAR_H + 8, width: 80, height: APP_H - TOPBAR_H - 16, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', boxSizing: 'border-box' }
      }
    >
      {TOOL_GROUPS.map((g, gi) => (
        <div key={gi} style={{ display: 'flex', flexDirection: TOOLBAR_PLACEMENT === 'bottom' ? 'row' : 'column', gap: TOOLBAR_PLACEMENT === 'bottom' ? 28 : 22, flex: TOOLBAR_PLACEMENT === 'bottom' ? 1 : undefined, justifyContent: TOOLBAR_PLACEMENT === 'bottom' ? ['flex-start', 'center', 'flex-end'][gi] : undefined, padding: TOOLBAR_PLACEMENT === 'bottom' ? '0 14px' : undefined }}>
          {g.map((icon) => {
            const name: IconName = icon === 'mic' && micOff ? 'mute' : icon === 'camera' && camOff ? 'camera-off' : icon;
            return <ToolbarButton key={icon} icon={name} active={icon === activeTool} red={icon === 'phone-disconnect'} />;
          })}
        </div>
      ))}
    </div>
    {children}
  </div>
);

// Stage area: right of the toolbar, or right of an open side panel
export const stageRect = (panelOpen: boolean) => {
  const bottom = TOOLBAR_PLACEMENT === 'bottom';
  const x = panelOpen ? PANEL.x + PANEL.w + 8 : bottom ? 8 : 96;
  return { x, y: TOPBAR_H + 4, w: APP_W - 8 - x, h: APP_H - TOPBAR_H - (bottom ? TOOLBAR_W + 4 : 12) };
};

// ---------------------------------------------------------------------------
// Light paper surfaces: menu, dialog, toast
// ---------------------------------------------------------------------------
export type MenuItem = { icon?: IconName; label: string; hover?: boolean; red?: boolean };

export const Menu: React.FC<{ x: number; y: number; title?: string; items: MenuItem[]; opacity?: number; width?: number }> = ({
  x,
  y,
  title,
  items,
  opacity = 1,
  width = 290,
}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width,
      background: R.paper,
      borderRadius: 4,
      boxShadow: '0 5px 5px -3px rgba(0,0,0,0.2), 0 8px 10px 1px rgba(0,0,0,0.14), 0 3px 14px 2px rgba(0,0,0,0.12)',
      padding: '8px 0',
      opacity,
      transform: `scale(${0.92 + 0.08 * opacity})`,
      transformOrigin: 'top left',
      fontFamily: RFONT,
      zIndex: 20,
    }}
  >
    {title && <div style={{ padding: '6px 16px', color: 'rgba(0,0,0,0.38)', fontSize: 15, textTransform: 'capitalize' }}>{title}</div>}
    {items.map((it) => (
      <div
        key={it.label}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '8px 16px',
          fontSize: 16,
          color: it.red ? R.red : R.text,
          background: it.hover ? 'rgba(55,113,224,0.1)' : 'transparent',
        }}
      >
        {it.icon && <RIcon name={it.icon} size={24} color={it.red ? R.red : 'rgba(0,0,0,0.7)'} />}
        {it.label}
      </div>
    ))}
  </div>
);

export const Dialog: React.FC<{
  title: string;
  opacity: number;
  width?: number;
  actions: { label: string; primary?: boolean; hover?: boolean }[];
  children?: React.ReactNode;
}> = ({ title, opacity, width = 444, actions, children }) =>
  opacity <= 0 ? null : (
    <>
      <div style={{ position: 'absolute', inset: 0, background: `rgba(0,0,0,${0.5 * opacity})`, zIndex: 30 }} />
      <div
        style={{
          position: 'absolute',
          left: APP_W / 2 - width / 2,
          top: 230,
          width,
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
        <div style={{ padding: '20px 24px 8px', fontSize: 20, fontWeight: 500 }}>{title}</div>
        <div style={{ padding: '8px 24px 12px', fontSize: 16, lineHeight: 1.45, color: R.text2 }}>{children}</div>
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
                color: a.primary ? R.primary : R.grey600,
                background: a.hover ? 'rgba(55,113,224,0.1)' : 'transparent',
              }}
            >
              {a.label}
            </div>
          ))}
        </div>
      </div>
    </>
  );

export const Toast: React.FC<{ text: string; opacity: number; x: number; y: number; icon?: IconName }> = ({ text, opacity, x, y, icon }) =>
  opacity <= 0 ? null : (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        minWidth: 300,
        padding: '14px 18px',
        background: R.paper,
        borderRadius: 8,
        boxShadow: '0 6px 20px rgba(0,0,0,0.3)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontFamily: RFONT,
        fontSize: 15,
        color: R.text,
        opacity,
        transform: `translateY(${(1 - opacity) * -12}px)`,
        zIndex: 25,
      }}
    >
      {icon && <RIcon name={icon} size={26} color={R.primary} />}
      {text}
    </div>
  );
