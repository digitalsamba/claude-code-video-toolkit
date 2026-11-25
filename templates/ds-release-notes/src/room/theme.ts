// Values from react-frontend/src/theme (palette.ts, customThemeOptions.ts, styleConstants.ts).
// Stage, top bar and toolbar are black in every mode; panels, menus and dialogs use the light palette here.
export const R = {
  room: '#000',
  tile: '#272a2e',
  primary: '#3771e0',
  primaryLight: '#5F8DE6',
  badge: '#8fb0ed',
  sectionBg: 'rgba(55,113,224,0.1)',
  paper: '#fff',
  text: 'rgba(0,0,0,0.87)',
  text2: 'rgba(0,0,0,0.6)',
  divider: '#c5c5c5',
  grey200: '#eeeeee',
  grey600: '#797979',
  red: '#f00',
  redDark: '#b20000',
  speaking: '#00ff00',
  waitingAvatar: '#757575',
  menuHover: 'rgba(0,0,0,0.04)',
};

export const RFONT =
  'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","Liberation Sans",sans-serif';

// Logical canvas, same as the account-center demo: a 1440-wide laptop browser
export const APP_W = 1440;
export const APP_H = 810;
export const CHROME_H = 40;

export const TOPBAR_H = 48;
export const TOOLBAR_W = 88;
// New teams get toolbar_position BOTTOM (webapp-api-laravel app/Models/Team.php DEFAULT_TOOLBAR_POSITION); 'left' is
// only the room app's fallback, and it reads as a customised room. Panels sit at the left edge above the toolbar.
export const TOOLBAR_PLACEMENT: 'left' | 'bottom' = 'bottom';
const BOTTOM = TOOLBAR_PLACEMENT === 'bottom';
export const PANEL = BOTTOM
  ? { x: 8, y: TOPBAR_H, w: 360, h: APP_H - TOPBAR_H - TOOLBAR_W }
  : { x: 96, y: TOPBAR_H, w: 360, h: APP_H - TOPBAR_H - 8 };
