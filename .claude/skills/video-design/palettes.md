# Act palettes

Palettes that hold up on video, written as fields with their roles. Each is two or three fields
ordered by lightness, so an act change is always a clear cut. Use them as starting points; a brand's
own colour usually replaces the accent, or one field.

Roles, as in `design.md`: **ground**, **ink** (text and plain data), **accent** (the one narrated
thing), **other** (unhighlighted data: one neutral step, never a tint of the ground).

## Brand + bone + black (light, mid, dark)

The default when a brand has one strong colour. The brand colour is the identity field (open and
close); bone carries the record (the long middle, where charts read best); near-black is the
surprise act (a turn, an experiment, the cost). Used for Lugh's work, 2026-11, with Digital Samba
orange.

| Field | Ground | Ink | Accent | Other |
| --- | --- | --- | --- | --- |
| identity | brand (e.g. `#ea580c`) | `#111111` | ink | — |
| record | `#f4f1ea` | `#111111` | brand | `#d9d4c7` |
| lab | `#161615` | `#f4f1ea` | brand | `#3b3a37` |

Check the brand's lightness: a very light brand colour (yellow, mint) sits too close to bone; swap
bone for white or a pale grey a full step away.

## Graphite, one accent (one field)

For a short piece or a single act: no field changes at all, everything carried by one accent.

| Field | Ground | Ink | Accent | Other |
| --- | --- | --- | --- | --- |
| graphite | `#18191b` | `#eff0f1` | brand | `#5f6268` |

From taste's `graphite-safety-orange`.

## Paper, one accent (one field, projector-friendly)

For rooms with light in them: townhalls, a projector at the back.

| Field | Ground | Ink | Accent | Other |
| --- | --- | --- | --- | --- |
| paper | `#ffffff` | `#111111` | brand, one step darker if it's light | `#c8ccd1` |

From taste's `white-signal-red`, with the brand as the accent.

## Loud two-act (bold, campaign pieces)

When the brief is "bright and poppy": two saturated fields of different lightness, plus black type.
Keep it to two; a third saturated field is where it starts to look like a different colour per scene.

| Field | Ground | Ink | Accent | Other |
| --- | --- | --- | --- | --- |
| loud | `#ff8126` | `#111111` | `#111111` | `#ffffff` |
| calm | `#c5f53d` | `#111111` | `#111111` | `#ffffff` |

From taste's `orange-sky-lime` (its sky blue left out: it has the same lightness as the orange).
On a saturated ground the data is two-tone: the narrated bar in ink, the rest solid white. Never ink
at partial opacity over the colour; that is what made the muddy olive and steel-blue bars.

Palettes marked "taste" are from taste-skill's library (https://github.com/Leonxlnx/taste-skill,
Copyright (c) 2026 Leonxlnx, MIT License), adapted into video roles.
