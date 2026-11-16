# Digital Samba: tone of voice for tutorials

For narration and on-screen text in short YouTube tutorials about Digital Samba Free
(my.digitalsamba.com). The machine-readable rules are in `copy.json` beside this file.

Sources, highest authority first. Where they disagree, the higher one wins.

1. `~/work/ds-support/kb/common/tone-guidelines.md` (house style from the marketing style guide, Jorge,
   2026-09-11), `banned-words.md`, `reply-rules.md`, and `kb/ds-free/README.md` and `feature-map.md`.
2. `~/work/ds-blog`. Nothing is in `published/` yet; the two drafts in `articles/` show how the team
   writes long form (`php2shell-response/draft.md` is the closest to house style).
3. https://www.digitalsamba.com/ and /pricing (read 2026-11-16; /free serves the homepage).
4. `~/.claude/agents/doc-enhancer.md`, "Digital Samba Brand Voice" (older, informal).
5. UI strings: `~/work/react-account-center/src/locales/en/common.json`.

## Who's talking

A small, independent European team that has built video calls since 2003 and explains its own product
to a friend. It knows the app inside out, says what to click, and gets out of the way. It's proud
that the product is free, private and hosted in Europe, but it shows that with facts, not slogans. It
sounds like a person: contractions, short sentences, plain words, a light smile now and then.

## Traits

**1. Plain and direct.** Say the thing in the fewest ordinary words.
- Do: "No trackers. No profiling. No selling your data." (digitalsamba.com, 'Commitment to privacy')
- Don't: `Our enterprise-grade solution leverages cutting-edge technology` (doc-enhancer.md, its own
  example of what not to write)

**2. Helpful first.** Lead with what to do, not with apologies or preamble.
- Do: "To schedule a call with custom settings, click the "Schedule" button on your dashboard, then
  select the template you'd like to use." (tone-guidelines.md, DS-Free example)
- Don't: "I hope this email finds you well" or "Great question!" (banned-words.md, support boilerplate)

**3. Honest and exact.** State what is true; don't oversell, don't hedge.
- Do: "Looking back, we missed an obvious shortcut." (ds-blog, php2shell-response/draft.md)
- Don't: `Experience seamless integration with our world-class platform` (doc-enhancer.md)

**4. Private and European, as fact.** Name the concrete thing, not a promise.
- Do: "100% European hosted. Only European sub-processors." (digitalsamba.com, 'Security & privacy')
- Don't: "Industry-leading security features ensure data protection" (doc-enhancer.md)

**5. Warm, with a light touch of cheek.** One wry line is plenty.
- Do: "Your words, not ours." (digitalsamba.com, under the testimonials heading)
- Don't: "Rest assured" or "I completely understand your frustration" (banned-words.md)

**6. Precise about the product.** Name the screen and the button you mean.
- Do: name the surface: "the Embedded dashboard", "your account center" (tone-guidelines.md, 'Unnamed
  product surfaces'); in a tutorial, the label itself: **Call now**.
- Don't: "the dashboard" or "your account" when more than one place fits (same section).

## For tutorials specifically

- **Second person, imperative steps.** "Open **Schedule**. Pick a template. Click **Save**." Not "the user
  clicks" or "we're going to go ahead and click".
- **Name UI labels exactly as the app shows them**, with the app's capitals and symbols. On screen, set
  them in bold; in narration, say them as written. Check each label against
  `react-account-center/src/locales/en/common.json` and the screen being shown. Labels confirmed there:
  **Call now**, **Schedule a call**, **Schedule**, **Scheduled calls**, **Dashboard**, **Contact list**,
  **Add contact**, **Recordings**, **Settings**, **Templates**, **Create template**, **Roles**,
  **Usage**, **History**, **Copy link**, **Join call**, **Save**, **Save & Send Invites**,
  **Sign up**, **Log in**, **End-to-end encryption**.
- **One idea per sentence**, one action per step. Vary the length so it doesn't sound like a list read
  aloud.
- **Show, then say.** Narration names the action and why it matters; the screen shows where. Don't
  describe what the viewer can already see ("a blue button in the top right") unless it is hard to find.
- **Warm vs cheeky.** The site is cheeky ("zero big brother stuff"); support is warm. Tutorials sit on
  the support side: warm, calm and clear, with at most one light, wry line per episode, usually in the
  intro or the close. Never joke inside a step; a joke there costs a viewer the click.
- **Privacy and Europe lightly.** Mention it when the feature is about privacy (**End-to-end
  encryption**, private calls, recordings, guests), or once in an episode's close. Not in every episode,
  and never as a sermon. One concrete fact beats an adjective: "hosted in Europe", not "ultra-secure".
- **No competitor names** in tutorials (Zoom, Teams, Meet, Skype, Whereby) unless Digital Samba asks for
  a comparison episode. Say "other video tools" if you must. The site names them; tutorials don't need to.
- **No call to action that is a call offer** and no "reach out": end on the next thing to try, or the
  help address.
- **Nothing that isn't live today.** No roadmap, no "coming soon" (tone-guidelines.md, 'Unreleased
  products and roadmap').
- **No internal names**: not Enzo, Monza, Janus, SDK event names, or "Meet Now" (the code's name for
  **Call now**).

## Product naming

| Use | Avoid | Note |
|---|---|---|
| Digital Samba Free (first mention), then Digital Samba | DS-Free, DS Free, DS, Samba, "the free version" | DS-Free is support's internal shorthand. The site and pricing page say "Digital Samba Free". |
| Digital Samba (the company) | Digital Samba SL, DigitalSamba | |
| call | meeting, session, room | The app says call: **Call now**, **Schedule a call**, **Scheduled calls**, **Join call**. "Meeting" is also a template name, so keep it for that. |
| template, and the names **Meeting**, **Classroom**, **Webinar** | room type, call type | Built-in templates per `kb/ds-free/README.md`. |
| role (**Attendee**, **Moderator**, **Speaker**, **Student**, **Teacher**) | permission group | |
| your dashboard (**Dashboard**) | account centre, the portal, the admin | Free users land on **Dashboard**; say where they go, by its label. |
| Account Center, only if you must name the whole app | account centre, Account Centre | Product name, keeps US spelling (tone-guidelines.md). |
| my.digitalsamba.com | app.digitalsamba.com | app. is legacy SambaLive (`kb/ds-free/README.md`). |

Don't mention Digital Samba Embedded, SambaLive or white-label platforms in a Free tutorial, except a
one-line pointer if a viewer would otherwise go looking for a feature that only Embedded has (branding,
dark mode, room colours: `feature-map.md`, 'Settings: Branding').

## Spelling and style

From tone-guidelines.md, 'Language rules':

- **British English**: organise, customise, realise, recognise; colour, behaviour, favour; centre (but
  **Account Center**); cancelled, labelled; catalogue, dialogue; licence (noun), license (verb); grey;
  enrolment; analyse, summarise. A UI label keeps the app's own spelling.
- **No dashes as punctuation.** No em dashes, no spaced en dashes or spaced hyphens as a break. Use a
  comma, colon, full stop or brackets. Hyphens inside a word stay: end-to-end, sign-up, 30-day.
- **Plain words.** Contractions are fine. Don't upgrade a word to a fancier one.
- **Numbers**: one to nine in words, 10 and up in digits; digits on both sides of a range ("from 2 to
  15"); digits with units (30 days, 720p). On screen, show a number exactly as the app shows it.
- **Dates**: 3 September 2026. **Times**: 14:00 or 2pm, with the time zone when it matters.
- **Quotes**: single quotes for a quoted term ('Waiting room'), double inside them, straight not curly.
  UI labels are bold, not quoted.
- **Headings and titles**: sentence case ("Schedule your first call"), not Title Case.
- Run every script and on-screen line through `tools/copy_lint.py` before voicing.

## Claims we only make if checked

Each needs a source (the live app, the current site, the code, or a person at Digital Samba) at publish time,
noted in the project. If it can't be checked, leave it out. Settled on 2026-11-16 (Conal):

- **Limits**: Digital Samba Free has none today. "Free" and "no time limit" are fine; quote no numbers, and
  don't promise it stays that way ("forever", "always free", "no catch").
- **Beta**: never. No "beta", "early access" or "join the beta", even where the site still says it.
- **Features**: Free has every feature in the 'Digital Samba features' list on digitalsamba.com/features.
  Branding (logo, colours, CNAME white labelling), the API, SDK, webhooks and phone dial-in are Embedded:
  don't show them as part of Free. Use the page's own feature names.
- **Link lifetime**: a link made with **Generate Link** on the website works for 30 days (the site's own
  code; its copy also says "1 month", so say 30 days). Links to calls scheduled in an account don't expire
  unless you set an end.
- **End-to-end encryption scope**: quote the app. Up to 100 participants; Q&A, polls, shared notes, saving
  whiteboards and recording don't work with it.
- **"No sign-up"**: true for a call started from the website; scheduling, contacts, templates and
  recordings need an account. Say which.
- **Where data lives**: "hosted in Europe" and "GDPR" are safe. City names (Amsterdam, Frankfurt) are
  confirmed for Embedded only.

## Words to avoid

The full list is `~/work/ds-support/kb/common/banned-words.md` (from the marketing team's blog list). In
short:

- **AI vocabulary**: `seamless`, `leverage`, `delve`, `elevate`, `enhance` (as filler), `streamline`,
  `robust`, `harness`, `unleash`, `foster`, `cutting-edge`, `game-changer`, `transformative`, `holistic`,
  `comprehensive`, `landscape`, `realm`, `tapestry`, `deep dive`, `in today's...`, `it's worth noting that`.
- **Support boilerplate**: `Great question!`, `Rest assured`, `Don't hesitate to reach out`,
  `Sorry for any inconvenience`, `Kindly`, `As per`, `I hope this finds you well`.
- **Patterns**: rising three-part lists ("X, Y, and even Z"); a rhetorical question answered with "Well,
  ..."; "Whether you're X or Y"; "Not just X, but Y" more than once; paragraphs all the same shape.
- **Tutorial filler**: `simply`, `just` (when it hides a step), `easy`, `go ahead and`, `let's dive in`,
  `without further ado`. If a step is easy, the viewer will notice.

Replace a banned word with the plainest one (use, improve, check, smooth) or delete it.
