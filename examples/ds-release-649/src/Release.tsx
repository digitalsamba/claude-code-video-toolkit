import React from 'react';
import { AbsoluteFill, Audio, Sequence, interpolate, staticFile } from 'remotion';
import { TitleCard, OutroCard, SceneLabel, TITLE_LEN, OUTRO_LEN } from './scenes/Cards';
import { HideTiles, HIDE_LEN, HIDE_CLICKS } from './scenes/HideTiles';
import { MuteByRole, MUTE_LEN, MUTE_CLICKS } from './scenes/MuteByRole';
import { Ban, BAN_LEN, BAN_CLICKS } from './scenes/Ban';
import { SdkBookmark, SDK_LEN, SDK_TOAST_AT } from './scenes/SdkBookmark';
import { Breakout, BREAKOUT_LEN, BREAKOUT_CLICKS } from './scenes/Breakout';
import { Lobby, LOBBY_LEN, LOBBY_CLICKS } from './scenes/Lobby';
import { Free, FREE_LEN, FREE_CLICKS } from './scenes/Free';

// ---------------------------------------------------------------------------
// Release 649 (30fps). Each scene is sized to its VO line plus a beat. VO is Lugh (Qwen3 clone of the
// meeting agent's lugh-ref.wav); 01, 03 and 06 had an extra atempo (0.92 / 0.857 / 0.853) to ~165 wpm.
//   00_title       2.96s   Title card               150f
//   01_hide_tiles 13.07s   Hide tiles by role       420f
//   02_mute_role   8.71s   Mute / cameras by role   320f
//   03_ban         9.08s   Safer ban duration       310f
//   04_sdk         6.01s   SDK bookmarks            260f
//   05_breakout    6.52s   Breakout unmute          240f
//   06_lobby       7.25s   Fixed: lobby admit       260f
//   07_free        6.96s   DS Free roundup          330f   split at its pauses into 07a-d (atempo 0.92),
//                                                         one phrase per vignette: @10, @116, @160, @226
//   08_outro       2.53s   Outro                    120f
// ---------------------------------------------------------------------------
type Scene = {
  id: string;
  len: number;
  vo: string;
  voAt: number;
  moreVo?: [string, number][]; // further phrases of the same line, placed on their own beats
  el: React.ReactNode;
  label?: [string, string];
  clicks?: number[];
  sfx?: { file: string; at: number; volume: number; frames?: number }[];
};

const SCENES: Scene[] = [
  { id: 'title', len: TITLE_LEN, vo: '00_title', voAt: 15, el: <TitleCard /> },
  { id: 'hide', len: HIDE_LEN, vo: '01_hide_tiles', voAt: 10, el: <HideTiles />, label: ['New feature', 'Hide tiles in recordings by role'], clicks: HIDE_CLICKS, sfx: [{ file: 'whoosh', at: 196, volume: 0.2 }] },
  { id: 'mute', len: MUTE_LEN, vo: '02_mute_role', voAt: 10, el: <MuteByRole />, label: ['New feature', 'Mute or turn off cameras by role'], clicks: MUTE_CLICKS },
  { id: 'ban', len: BAN_LEN, vo: '03_ban', voAt: 10, el: <Ban />, label: ['New feature', 'Safer ban duration'], clicks: BAN_CLICKS },
  {
    id: 'sdk',
    len: SDK_LEN,
    vo: '04_sdk',
    voAt: 10,
    el: <SdkBookmark />,
    label: ['New feature', 'Recording bookmarks through the SDK'],
    sfx: [
      { file: 'typing', at: 34, volume: 0.3, frames: 75 },
      { file: 'chime', at: SDK_TOAST_AT, volume: 0.35 },
    ],
  },
  { id: 'breakout', len: BREAKOUT_LEN, vo: '05_breakout', voAt: 10, el: <Breakout />, label: ['New feature', 'Unmute by default in breakout rooms'], clicks: BREAKOUT_CLICKS },
  { id: 'lobby', len: LOBBY_LEN, vo: '06_lobby', voAt: 10, el: <Lobby />, label: ['Fixed', 'Admitting from the lobby'], clicks: LOBBY_CLICKS, sfx: [{ file: 'pop', at: 4, volume: 0.3 }] },
  {
    id: 'free',
    len: FREE_LEN,
    vo: '07a_free',
    voAt: 10,
    moreVo: [
      ['07b_free', 116],
      ['07c_free', 160],
      ['07d_free', 226],
    ],
    el: <Free />, clicks: FREE_CLICKS, sfx: [{ file: 'whoosh', at: 106, volume: 0.15 }, { file: 'whoosh', at: 216, volume: 0.15 }] },
  { id: 'outro', len: OUTRO_LEN, vo: '08_outro', voAt: 10, el: <OutroCard /> },
];

const starts = SCENES.reduce<number[]>((acc, s, i) => [...acc, i === 0 ? 0 : acc[i - 1] + SCENES[i - 1].len], []);
export const RELEASE_LEN = SCENES.reduce((n, s) => n + s.len, 0);

const Sfx: React.FC<{ file: string; at: number; volume: number; frames?: number }> = ({ file, at, volume, frames }) => (
  <Sequence from={at} durationInFrames={frames} layout="none">
    <Audio
      src={staticFile(`audio/sfx/${file}.mp3`)}
      volume={(f) => (frames ? volume * interpolate(f, [frames - 6, frames], [1, 0], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }) : volume)}
    />
  </Sequence>
);

export const Release: React.FC = () => (
  <AbsoluteFill style={{ background: '#eef2f8' }}>
    {SCENES.map((s, i) => (
      <Sequence key={s.id} from={starts[i]} durationInFrames={s.len}>
        {s.el}
        {s.label && <SceneLabel kind={s.label[0]} title={s.label[1]} />}
        <Sequence from={s.voAt} layout="none">
          <Audio src={staticFile(`audio/scenes/${s.vo}.mp3`)} />
        </Sequence>
        {(s.moreVo ?? []).map(([file, at]) => (
          <Sequence key={file} from={at} layout="none">
            <Audio src={staticFile(`audio/scenes/${file}.mp3`)} />
          </Sequence>
        ))}
        {(s.clicks ?? []).map((c) => (
          <Sfx key={`c${c}`} file="click" at={c} volume={0.45} />
        ))}
        {(s.sfx ?? []).map((x) => (
          <Sfx key={`${x.file}${x.at}`} {...x} />
        ))}
        {i > 0 && <Sfx file="whoosh" at={0} volume={0.2} />}
      </Sequence>
    ))}
  </AbsoluteFill>
);
