# Genres: borrow the shape, not the work

A genre is a set of beats the audience already expects. Pick one whose beats map onto the real event, then give
each cast member a beat. Borrow form only: no characters, names, lines, designs or theme music from a specific
film or show (renamed or not). Public-domain stories can be used directly.

## Mappings that worked

| Genre | Real event | How the beats mapped |
|---|---|---|
| **80s action-show intro** | a release night | narrator sets the mission; a title card per person with a freeze-frame and a stamp; the plan, the montage of welding and cramming, the win with a cigar. |
| **Creature feature** (original creatures, the "glitchlings") | the same release | the new release arrives in a box with three rules (never restart it with people inside, never leave its secrets lying around, never feed it on a Friday night); each rule gets broken; dawn, all quiet. |
| **Dickens Christmas TV ad** (*A Christmas Carol*, public domain) | the year's updates | Scrooge = the release lead with a runsheet; Past = old updates; Present = features crammed in; Yet to Come = boxes still unticked; redemption = "automate the deploys". |
| **Horror trailer** (invented cast, glitchlings) | a shared topic: the video call that never ends | "it was supposed to be a quick sync"; muted, frozen, unseen screen; the meeting timer passes 46 hours and Leave runs from the cursor; the title slam; then a stinger after the title: "let's take this offline", and a follow-up invite arrives. |
| **Wildlife documentary** (invented cast) | a shared topic: the return to the office | the home habitat (shirt above, pyjamas below); the signal (a "mandatory" email); the migration; the lanyard display; a hot-desk territory fight; the watering hole; the predator ("Got a minute?"); the twist: reunited, the herd joins a video call. |

What made them fit: each genre has a **turn** that matches the event's turn (the win; dawn after the night;
Scrooge's change of heart), and roles that match real roles (the planner, the welder, the one who keeps adding).

## Candidates

| Genre | Beats | Fits |
|---|---|---|
| Heist movie | the crew, the plan on a whiteboard, the vault, the twist, the getaway | a launch, an office move, the quarter-end deal |
| Nature documentary | hushed narrator, the creature in its habitat, the hunt, the migration | a day in the office, the support queue, the return to the office |
| Space mission | countdown, mission control, a problem, improvisation, splashdown | a launch, an event day, the school play's opening night |
| Courtroom drama | the accused, witnesses, the evidence, the verdict | who took the last milk, a postmortem |
| Cooking show | ingredients, prep, the oven, "here's one I made earlier" | how a campaign or a feature gets made |
| Sports commentary | two commentators, the replay, the stats overlay, the trophy | sports day, a sales leaderboard, a hackathon |
| Western | the stranger in town, the showdown at noon | a new hire or new manager, the new system |
| Noir detective | rain, voice-over, the mystery, the twist | the missing stapler, a mystery outage |
| Infomercial | the problem in black and white, "but wait, there's more!" | a new tool, policy or process everyone must adopt |
| Movie trailer | "In a world...", quick cuts, the title, "coming soon" | next year's plan, an away day |
| Horror | the call that never ends, the thing in the inbox | meetings, email, the shared drive |
| Holiday ads | Christmas (Dickens, carollers), Halloween (creature feature), New Year (countdown) | end of year, anything near a holiday |
| Fairy tale (Grimm, public domain) | three tasks, a curse, a happily ever after | a long project, a retirement send-off |

Devices worth reusing:

- **Mock-Latin species lower thirds** in a documentary ("Lanyardus magnificus", "Hotdeskus contestus"): a joke
  on every shot that costs one line of text.
- **Camera-trap cards** (grainy, timestamped "CAM 3 · TUE 09:41") for arrivals.
- **A trailer stinger after the title card:** the scare is over, the music stops dead, and one last small gag
  lands in the silence.
- **A UI that misbehaves:** a runaway Leave button, a timer that accelerates, a spinner that never finishes. Built
  in Remotion, so it costs nothing to animate.

## Writing the beats

- One beat per cast member, each with one action their animated photo can do (`people.md`) and one line or text
  gag from the bible.
- Write the turn first, then work back.
- On-screen text is the punchline: a stamp, a speech bubble, a sign, a list. Keep each to a short phrase.
- An ad has at most one narrator line, usually the tagline over the end card. A show intro can carry a narrator
  throughout (Lugh's expressive clone, sentence-level takes; see CLAUDE.md, Clone Prosody).
- End on the team's logo and a tagline that turns the genre into the team's value ("Every update deserves a
  silent night").

## Look and sound per genre

| Genre | Type (Google Fonts) | Picture | Sound |
|---|---|---|---|
| 80s action | a heavy condensed display, VT323 for labels | VHS overlay: scanlines, tracking jitter, chroma shift, date stamp | synth-rock score, whooshes, a cigar fizz |
| Horror trailer | Creepster for the title, VT323 for timers, a plain UI sans for the app | cold teal grade, grain, vignette, a flicker | drones and a music box, a trailer boom on the title, heartbeat, static, then birdsong for the stinger |
| Wildlife documentary | a classic serif for the title and Latin names, a clean sans for common names | warm grade, fine grain, slow push-ins, b-roll that moves (LTX on a background still) | pizzicato strings and horn, a hushed narrator, ambience per habitat (kettle, train, office, espresso) |
| Creature feature | Creepster for titles, VT323 for terminals | fog, a flickering light, a low blue-green grade | a horror score, munching, a burp |
| Dickens ad | IM Fell English and its SC cut | warm grade, film grain, letterbox bars, falling snow, slow Ken Burns | an original carol with lyrics, bells, a clock strike, chains |

Build the overlay once per genre as a component: grain, letterbox and grade sit over every shot.
