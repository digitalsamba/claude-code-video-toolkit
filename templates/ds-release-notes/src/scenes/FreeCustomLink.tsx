import React from 'react';
import { useCurrentFrame } from 'remotion';
import { Stage, cursorPos, fade, near, typed } from '../Stage';
import { Shell } from '../ac/Shell';
import { C } from '../ac/theme';
import { Avatar, Button, FieldLabel, Switch, TextField } from '../ac/primitives';
import { press } from '../Cursor';
import { voice } from '../cues';
import line06a from '../../public/audio/scenes/06a_link.captions.json';
import line06b from '../../public/audio/scenes/06b_link.captions.json';

// DS Free custom room link warning: react-account-center schedule/_components/custom-slug.js (SAMBA-9671).
// Inline red FormHelperText under the slug field, live while editing an existing call.
export const LINK_LEN = 470;

// Every action is anchored to the word it illustrates, so regenerating or re-timing the voice moves the
// clicks, the typing, the cursor and the camera with it. Release.tsx places the audio from LINK_VOICE.lines.
export const LINK_VOICE = voice([
  { file: '06a_link', at: 10, captions: line06a },
  { file: '06b_link', at: 256, captions: line06b },
]);
const { cue } = LINK_VOICE;
const C_ = {
  field: cue('before', { offset: -3 }), // click into the slug field as "before you change" starts
  save: cue('save'), //                   "...when you save."
};
export const LINK_CLICKS = Object.values(C_);
const TYPE_AT = cue('change', { offset: -2 }); // the slug changes on "change"
export const LINK_TYPE_AT = TYPE_AT;
const WARNING = cue('because'); // the warning's reason is read out from here
const SUFFIX = '-q4';
const BASE_SLUG = 'weekly-sync';

const X = 281;
const W = 551;
const SW = X + W - 38; // switch column
const SLUG = { x: X + 262, y: 538, w: W - 262 };
const fieldEnd = { x: SLUG.x + 150, y: SLUG.y + 20 };
const saveBtn = { x: 1416 - 98, y: 149 };

const PATH = [
  { f: 0, x: 760, y: 700 },
  { f: C_.field - 30, x: 760, y: 700 },
  { f: C_.field - 6, ...fieldEnd }, // arrive just before the click
  { f: C_.save - 74, ...fieldEnd },
  { f: C_.save - 24, ...saveBtn },
];
const CAMERA = [
  { f: 0, z: 1, x: 720, y: 405 },
  { f: C_.field - 40, z: 1, x: 720, y: 405 },
  { f: C_.field - 10, z: 1.55, x: 600, y: 470 }, // in on the field
  { f: WARNING - 12, z: 1.55, x: 600, y: 470 },
  { f: WARNING + 14, z: 1.75, x: 560, y: 540 }, // closer, onto the warning text
  { f: C_.save - 84, z: 1.75, x: 560, y: 540 },
  { f: C_.save - 48, z: 1.15, x: 820, y: 330 }, // out to the Save button
];

const PARTICIPANTS = [
  { name: 'Anna Schmidt', email: 'anna.schmidt@acme.eu', color: '#d4a03a' },
  { name: 'Marco Rossi', email: 'marco.rossi@acme.eu', color: '#3a9a8a' },
  { name: 'Lea Novak', email: 'lea.novak@acme.eu', color: '#7b5ea7' },
];

export const FreeCustomLink: React.FC = () => {
  const frame = useCurrentFrame();
  const cur = cursorPos(frame, PATH);
  const focused = frame >= C_.field + 1;
  const slug = BASE_SLUG + typed(frame, SUFFIX, TYPE_AT, 7);
  const changed = slug !== BASE_SLUG;
  const warn = fade(frame, TYPE_AT + 1, TYPE_AT + 10) * (changed ? 1 : 0);

  return (
    <Stage url={`my.digitalsamba.com/schedule/${BASE_SLUG}`} camera={CAMERA} path={PATH} clicks={LINK_CLICKS}>
      <Shell active={0}>
        <div style={{ position: 'absolute', left: X, top: 134, fontSize: 24 }}>Update call</div>
        <Button x={1416 - 196 - 16 - 92} y={130} w={92} label="Cancel" variant="secondary" />
        <Button x={1416 - 196} y={130} w={196} label="Save & Send Invites" hover={near(cur, saveBtn, 98, 19)} pressed={press(frame, C_.save)} />

        <FieldLabel x={X} y={196} label="Title" required />
        <TextField x={X} y={222} w={W} value="Weekly product sync" />
        <FieldLabel x={X} y={280} label="Date and time" required />
        <TextField x={X} y={306} w={330} value="Thu, 5 Nov 2026" />
        <TextField x={X + 346} y={306} w={W - 346} value="10:00 – 10:45" />

        <FieldLabel x={X} y={372} label="Send invitations to participants" info />
        <Switch x={SW} y={372} on />
        <FieldLabel x={X} y={416} label="Customised invitation email" />
        <Switch x={SW} y={416} on={false} />
        <FieldLabel x={X} y={460} label="Use custom room link" info />
        <Switch x={SW} y={460} on />
        <div style={{ position: 'absolute', left: X, top: SLUG.y + 11, fontSize: 14, color: C.textSecondary, whiteSpace: 'nowrap' }}>https://my.digitalsamba.com/c/acme/</div>
        <TextField x={SLUG.x} y={SLUG.y} w={SLUG.w} value={slug} focused={focused} caret={focused} placeholder="e.g. my-call" />
        {warn > 0 && (
          <div style={{ position: 'absolute', left: X, top: SLUG.y + 46, width: W, fontSize: 13, lineHeight: 1.4, color: '#d32f2f', opacity: warn }}>
            Changing this link will stop all links already shared for this call from working. When you save, invited participants and registrants will get the new link by email. Update any links you shared yourself.
          </div>
        )}
        <FieldLabel x={X} y={660} label="Registration" />
        <Switch x={SW} y={660} on />
        <FieldLabel x={X} y={704} label="Event image" />

        {/* Right column: participants */}
        <FieldLabel x={880} y={196} label="Add participants" />
        <TextField x={880} y={222} w={536} placeholder="Search contacts or enter an email" />
        {PARTICIPANTS.map((p, i) => (
          <div key={p.email} style={{ position: 'absolute', left: 880, top: 284 + i * 56, width: 536, height: 48, display: 'flex', alignItems: 'center', gap: 14, fontSize: 15 }}>
            <Avatar inline size={36} initials={p.name.split(' ').map((w) => w[0]).join('')} color={p.color} fontSize={14} />
            <span style={{ width: 170 }}>{p.name}</span>
            <span style={{ color: C.textSecondary }}>{p.email}</span>
          </div>
        ))}
      </Shell>
    </Stage>
  );
};
