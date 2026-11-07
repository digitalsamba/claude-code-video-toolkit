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

What made them fit: each genre has a **turn** that matches the event's turn (the win; dawn after the night;
Scrooge's change of heart), and roles that match real roles (the planner, the welder, the one who keeps adding).

## Candidates

| Genre | Beats | Fits |
|---|---|---|
| Heist movie | the crew, the plan on a whiteboard, the vault, the twist, the getaway | migrations, launches, a big cutover |
| Nature documentary | hushed narrator, the creature in its habitat, the hunt, the migration | a day in on-call, a sprint, the support queue |
| Space mission | countdown, mission control, a problem, improvisation, splashdown | a launch, a deploy window |
| Courtroom drama | the accused (a bug), witnesses, the evidence, the verdict | a postmortem, a bug hunt |
| Cooking show | ingredients, prep, the oven, "here's one I made earlier" | building a feature, an infra recipe |
| Sports commentary | two commentators, the replay, the stats overlay, the trophy | a hackathon, a quarter's numbers |
| Western | the stranger in town, the showdown at noon | a new hire, a legacy system |
| Noir detective | rain, voice-over, the dame (a dashboard), the twist | a mystery outage |
| Infomercial | the problem in black and white, "but wait, there's more!" | an internal tool launch |
| Movie trailer | "In a world...", quick cuts, the title, "coming soon" | a roadmap, the next quarter |
| Holiday ads | Christmas (Dickens, carollers), Halloween (creature feature), New Year (countdown) | end of year, any release near a holiday |
| Fairy tale (Grimm, public domain) | three tasks, a curse, a happily ever after | a long project |

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
| Creature feature | Creepster for titles, VT323 for terminals | fog, a flickering light, a low blue-green grade | a horror score, munching, a burp |
| Dickens ad | IM Fell English and its SC cut | warm grade, film grain, letterbox bars, falling snow, slow Ken Burns | an original carol with lyrics, bells, a clock strike, chains |

Build the overlay once per genre as a component: grain, letterbox and grade sit over every shot.
