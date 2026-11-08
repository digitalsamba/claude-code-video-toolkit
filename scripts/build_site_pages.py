"""Builds the four use-case pages in site/ from one template, so every page shares the front page's header, footer and
grammar. The front page (site/index.html) is written by hand. Run after editing, then lint:
  python3 scripts/build_site_pages.py && uv run tools/copy_lint.py site/*.html
"""
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
REPO = "https://github.com/digitalsamba/claude-code-video-toolkit"
PLAYLIST = "https://www.youtube.com/playlist?list=PLZt9pcZKDveo"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="https://digitalsamba.github.io/claude-code-video-toolkit/{og}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%233771e0'/%3E%3Cpath d='M12 9v14l11-7z' fill='%23fff'/%3E%3C/svg%3E">
<link rel="stylesheet" href="style.css">
<script src="site.js" defer></script>
</head>
<body>
<header class="frame">
  <nav class="nav" aria-label="Main">
    <a class="mark" href="./">claude-code-video-toolkit</a>
    <ul>
      <li class="hide-sm"><a class="link" href="./#made">Videos</a></li>
      <li class="hide-sm"><a class="link" href="./#how">How it works</a></li>
      <li class="hide-sm"><a class="link" href="./#routes">What to make</a></li>
      <li><a class="button" href="{repo}">GitHub</a></li>
    </ul>
  </nav>
</header>
<main>
"""

FOOT = """</main>

<footer class="foot frame">
  <div class="top">
    <div class="notes"><span>Open source, MIT licensed</span><span>Made by Conal Mullan and <a class="link" href="https://github.com/digitalsamba/claude-code-video-toolkit/graphs/contributors">contributors</a></span><span>For <a class="link" href="https://claude.ai/code">Claude Code</a>, not made by Anthropic</span></div>
    <a class="link" href="#">Back to top</a>
  </div>
  <a class="big" href="{repo}"><span>github.com/digitalsamba/</span><span>claude-code-video-toolkit</span></a>
  <div class="meta">
    <div><a class="link" href="./">Home</a><a class="link" href="{repo}/releases">Releases</a><a class="link" href="{playlist}">Playlist</a></div>
    <div class="c">Every video and picture here was made with the toolkit</div>
    <div class="r">2026</div>
  </div>
</footer>
</body>
</html>
"""


def head(h1, lede):
    return f"""  <div class="article-head frame grid">
    <div class="words">
      <h1>{h1}</h1>
      <p class="lede" data-rise>{lede}</p>
    </div>
    <figure class="clip shutter">
      <video data-loop autoplay muted loop playsinline preload="metadata" poster="media/made-with.jpg" width="768" height="296" aria-label="Clips from videos made with the toolkit"><source src="media/made-with.mp4" type="video/mp4"></video>
      <figcaption>Two seconds from each of five videos it made</figcaption>
    </figure>
  </div>
"""


def split(id_, h2, say, show, field=False):
    say = say.replace("<li>", "<li data-rise>").replace("<p>", "<p data-rise>")
    show = show.replace("<figure>", "<figure data-rise>").replace("<pre>", "<pre data-rise>").replace("<table class=\"facts\">", "<table class=\"facts\" data-rise>")
    cls = "frame field" if field else "frame"
    return f"""
  <section class="{cls}" aria-labelledby="{id_}">
    <div class="grid split" data-stagger>
      <div class="say">
        <h2 id="{id_}" data-rise>{h2}</h2>
        <div>{say}</div>
      </div>
      <div class="show">{show}</div>
    </div>
  </section>
"""


def fig(src, alt, cap, w=None, h=None):
    size = f' width="{w}" height="{h}"' if w else ""
    return f'<figure><img src="{src}" alt="{alt}"{size} loading="lazy"><figcaption>{cap}</figcaption></figure>'


def pair(a, b):
    return f'<div class="pair">{a}{b}</div>'


def page(name, title, desc, og, body):
    html = HEAD.format(title=title, desc=desc, og=og, repo=REPO) + body + FOOT.format(repo=REPO, playlist=PLAYLIST)
    (SITE / name).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- Christmas
page("christmas-team-video.html",
     "A Christmas video of your team, made with Claude Code",
     "Turn your team into A Christmas Carol: their own photos, costumed and animated, the in-jokes from your team chat, an original carol. A 90-second ad for about $2, made with Claude Code.",
     "media/band-mute.jpg",
     head("A Christmas video of your team",
          "Your colleagues as <i>A Christmas Carol</i>: their own photos in costume and moving, the jokes from your team "
          "chat, an original carol. Ninety seconds, about $2.")
     + split("why", "Why it lands",
             "<p>The videos that land hardest are the ones where the audience knows the people. A Christmas ad of your "
             "own team, where the person who always books one more meeting is Scrooge and the ghosts are the colleagues "
             "who were there for the year, gets laughs no stock video can.</p>"
             "<p>The toolkit's <code>parody-short</code> skill is how Claude Code makes one. It was built on three videos "
             "for one team, then on two public shorts with an invented cast, shown here. The team videos stay private "
             "because they star real colleagues.</p>",
             pair(fig("img/youre-on-mute-short.jpg", "A video call grid of four invented colleagues", "You're On Mute", 270, 480),
                  fig("img/the-great-return-short.jpg", "Four invented colleagues in headsets", "The Great Return", 270, 480)))
     + split("ask", "Ask for it",
             "<p>You need Claude Code and the toolkit, a <a class=\"link\" href=\"https://modal.com\">Modal</a> account for "
             "the image and video models (the toolkit deploys them for you), a free "
             "<a class=\"link\" href=\"https://acemusic.ai/api-key\">ACE-Step key</a> for the carol, three to six people "
             "with a photo of each, and <b>each person's OK</b> to appear in it.</p>"
             "<p>Then tell Claude Code what you want, in your own words.</p>",
             "<pre><code>Make a 90-second Christmas TV ad of my team as A Christmas Carol.\n"
             "Cast: Sam (Scrooge, books meetings on Christmas Eve), Priya, Jonas, Ana.\n"
             "Their photos are in ~/Desktop/team. It's for our end-of-year call,\n"
             "so facts don't matter: exaggerate freely.</code></pre>")
     + split("steps", "What Claude Code does",
             "<ol><li><b>Brief.</b> The event, the cast, and one question early: do facts matter? Say no, and nobody "
             "fact-checks a joke.</li>"
             "<li><b>In-jokes.</b> With a chat connector it reads your public channels for catchphrases and running jokes. "
             "DMs are never quoted.</li>"
             "<li><b>The story.</b> <i>A Christmas Carol</i> is public domain: Scrooge, Marley's chains, three ghosts, the "
             "change of heart. One beat per person.</li>"
             "<li><b>Costume and animate.</b> Each photo is costumed with an image edit and animated with LTX. Every clip "
             "goes on a contact sheet first; about one in three is re-rolled.</li>"
             "<li><b>Build.</b> Remotion shots in the Dickens look, with all the text drawn by Remotion.</li>"
             "<li><b>Sound.</b> An original carol written for your team, bells and chains, each sound placed by "
             "measurement.</li>"
             "<li><b>Check.</b> Loudness, frozen clips and sound cues, measured; then the rules below again.</li></ol>",
             pair(fig("media/stage-dana.jpg", "Dana, an invented cast member, at her desk", "A cast member, before costume", 704, 768),
                  fig("media/stage-render.jpg", "A finished frame of four colleagues on a call", "The same cast, in a finished shot", 1080, 608)),
             field=True)
     + split("rules", "The rules that keep it fun",
             "<ul><li><b>Affectionate, always.</b> Everyone is the hero of a gag or in on it. Nothing about performance, "
             "health, personal life or anything said in confidence.</li>"
             "<li><b>Real faces stay internal.</b> Share it with the team, link-only. Ask each person before posting "
             "anywhere public, and leave out anyone who asks.</li>"
             "<li><b>Nothing sensitive on screen.</b> No customer names, hostnames, internal URLs or secrets.</li>"
             "<li><b>Original words and music.</b> Dickens is public domain; a film's characters, lines or theme tune "
             "are not.</li></ul>",
             "<table class=\"facts\"><caption class=\"sr-only\">What it costs on Modal, November 2026</caption><tbody>"
             "<tr><td>Costume edit</td><td>about $0.01 each</td></tr>"
             "<tr><td>Five-second LTX clip</td><td>about $0.08 each</td></tr>"
             "<tr><td>Background image</td><td>about $0.01 each</td></tr>"
             "<tr><td>The carol</td><td>a few cents</td></tr>"
             "<tr class=\"total\"><td>A 90-second ad, five people, with re-rolls</td><td>about $2</td></tr>"
             "</tbody></table>")
     + split("more", "Not Christmas?",
             "<p>The same skill does an 80s action-show intro of a release night, a creature feature for Halloween, a "
             "nature documentary of the office, or a heist movie of a quarter-end deal. Reuse the costumed cast for the "
             "next one: only the costumes, clips and world change.</p>"
             f"<p><a class=\"button\" href=\"{REPO}/tree/main/.claude/skills/parody-short\">Read the parody-short skill</a></p>",
             fig("media/band-mute.jpg", "Four people on a video call, one saying Any other business?",
                 "You're On Mute, a horror trailer about video calls", 1920, 822)))

# ---------------------------------------------------------------- Release notes
page("release-notes-video.html",
     "Release-notes videos from source code, with Claude Code",
     "Product demo and release-notes videos with no screen recording: Claude Code rebuilds the app's screens from its source as React, then drives them with a scripted cursor and camera timed to the narration.",
     "img/ds-release-650.jpg",
     head("Release notes with no screen recording",
          "Claude Code reads your app's source, rebuilds the screens as React and films them with a scripted cursor and "
          "camera. Nothing to record, no test data, nothing that goes stale.")
     + split("why", "Why not record the screen?",
             "<p>A screen recording needs a running app, accounts, test data and a clean state, and it breaks on the next "
             "UI change. A rebuilt screen is static React: it renders the same every time, shows exactly the state the "
             "story needs, and the camera can zoom anywhere without blur.</p>",
             fig("img/ds-release-650.jpg", "A product settings screen rebuilt as React, with a language menu open",
                 "Release 650: six items across two apps, every click on its word"))
     + split("how", "How it works",
             "<ol><li><b>Write the narration first,</b> one sentence per item, and generate it. A cloned voice works well.</li>"
             "<li><b>Rebuild each screen</b> from the app's own components and styles, as a static page at a fixed canvas "
             "size. Claude Code reads the source; you point it at the right files.</li>"
             "<li><b>Drive it with <code>Stage</code> and <code>Cursor</code>:</b> a browser window, camera keyframes for "
             "zoom and pan, and an eased cursor with click ripples. Every click is a frame looked up from the narration "
             "with <code>cue('save')</code>, so a re-timed voice moves the whole scene with it.</li>"
             "<li><b>Check the text</b> with <code>qc.py --layout</code>: text too small for a phone, clipped, or under "
             "WCAG contrast.</li></ol>",
             pair(fig("img/ds-release-649.jpg", "A rebuilt product screen with the cursor on a control", "Release 649"),
                  fig("img/ds-free-schedule-demo.jpg", "A scheduling form with a talking-head narrator tile",
                      "A product demo with a narrator tile")),
             field=True)
     + split("start", "Start",
             "<p>The template ships as a finished build, so it renders as it is. For your release, keep the screen kits "
             "and replace the scenes. The method and its gotchas are in the "
             f"<a class=\"link\" href=\"{REPO}/tree/main/.claude/skills/synthetic-ui-demo\">synthetic-ui-demo skill</a>.</p>",
             "<pre><code>cp -r templates/ds-release-notes projects/my-release\n# then, in Claude Code:\n"
             "Make a release-notes video for these three changes: ...\nThe app's source is in ~/code/my-app.</code></pre>"
             + fig("img/soulx-talking-head.jpg", "A presenter generated as a talking head", "A narrator tile made with SoulX")))

# ---------------------------------------------------------------- Explainer shorts
page("explainer-shorts.html",
     "Vertical explainer shorts with Claude Code",
     "9:16 concept-explainer shorts for YouTube Shorts, Reels and TikTok, made with Claude Code: per-scene narration, cards with legible text, AI b-roll and karaoke captions, for under a dollar.",
     "img/sky-blue-short.jpg",
     head("Vertical explainer shorts",
          "One question, answered in under a minute for Shorts, Reels or TikTok: narration, cards with text you can read, "
          "b-roll that moves, and captions that follow the voice.")
     + split("made", "Two it made",
             "<p><a class=\"link\" href=\"https://demos.digitalsamba.com/video/short-blue-sky.mp4\">Why is the sky blue?</a> "
             "runs 52 seconds and cost about $0.80 to generate. "
             "<a class=\"link\" href=\"https://demos.digitalsamba.com/video/hallucinations_short.mp4\">AI hallucinations</a> "
             "runs 3:41 with a cloned voice.</p>",
             pair(fig("img/sky-blue-short.jpg", "A sunset over the sea with the caption Far more air", "Why is the sky blue?", 270, 480),
                  fig("img/hallucinations-short.jpg", "A vertical explainer frame", "AI hallucinations", 270, 480)))
     + split("how", "How it's built",
             "<p>The whole video comes from one <code>scenes.json</code>: a line of narration and one visual per scene.</p>"
             "<ol><li><b>Voice</b> each scene with Qwen3-TTS, a built-in voice or a clone, with a pace limit so it never "
             "rushes.</li>"
             "<li><b>Pictures by job:</b> Ideogram for anything with text on it, LTX video for b-roll that moves, FLUX for "
             "text-free backgrounds. Image and video models garble letters, so text is never asked of them.</li>"
             "<li><b>Captions</b> word by word from the audio's timing, but spelled from the script, so names come out "
             "right.</li>"
             "<li><b>Build</b> anchored to the audio: each scene starts where its narration does. It renders at every "
             "stage, with placeholder cards before the pictures exist.</li></ol>",
             pair(fig("media/explainer-1.jpg", "A text card defining hallucination as confident fiction", "A card made by Ideogram", 540, 960),
                  fig("media/explainer-2.jpg", "Empty desks in a hall, with a burned caption", "LTX b-roll with captions", 540, 960)),
             field=True)
     + split("start", "Start",
             "<p>Claude Code uses the <code>concept-explainer-short</code> template, pure Python with no Remotion. An agent "
             "with nobody at the terminal can do the same as a background job with a cost cap "
             f"(<a class=\"link\" href=\"{REPO}/blob/main/docs/agent-video-jobs.md\">agent video jobs</a>, experimental).</p>"
             "<p>Already have a 16:9 video? <code>tools/cutdown.py --platform shorts</code> makes the 9:16, 1:1 or 4:5 cut "
             "with ffmpeg.</p>",
             "<pre><code># in Claude Code, from the toolkit:\nMake a 50-second vertical explainer:\nwhy does bread go stale?\n\n"
             "# or as a job any agent can submit:\nuv run tools/video_job.py submit --spec spec.json --max-usd 1</code></pre>"))

# ---------------------------------------------------------------- Lesson
page("extending-ltx-clips.html",
     "Why AI video clips freeze or loop mid-shot, and the fix",
     "An image-to-video clip runs about 5 seconds; a shot runs 7 to 11. Holding the last frame left 63% of an 11-second shot frozen, playing it back and forth made 29% of frames repeats. Continuing the clip from its own last good frame measured 0% of both.",
     "media/stage-render.jpg",
     head("AI clips that don't freeze or loop",
          "An image-to-video clip runs about five seconds. A shot often runs seven to eleven. What fills the gap decides "
          "whether a generated person looks alive.")
     + split("measured", "Three ways to fill it, measured",
             "<p>We hit this making comedy shorts with animated photos: an LTX clip of a person laughing is about five "
             "seconds, and the shot it sits in is longer. Each way of filling it was measured on the same 11-second shot. "
             "The raw LTX clips measured 0% repeats, so the loop was ours, not the model's.</p>",
             "<table class=\"facts\"><caption class=\"sr-only\">Frozen and repeated frames on an 11-second shot</caption>"
             "<tbody><tr><td>Hold the last frame</td><td>63% frozen, reads as a pause</td></tr>"
             "<tr><td>Play the tail back and forth</td><td>29% repeats, reads as a rewind</td></tr>"
             "<tr class=\"total\"><td>Continue the clip from its own frame</td><td>0% frozen, 0% repeats</td></tr>"
             "</tbody></table>"
             + fig("media/stage-sheet.jpg", "A contact sheet: one settled frame from each scene of a render",
                   "Every clip goes on a contact sheet before it is used", 1080, 202))
     + split("fix", "Continue, don't stretch",
             "<p>Take the clip's last <i>good</i> frame, run image-to-video again from it with the same prompt, and join the "
             "two with a cross-fade of about 0.12 seconds. Repeat for longer shots. Each extension is one more clip, about "
             "$0.08 on Modal.</p>"
             "<p>\"Last good\" matters: continuing from an earlier frame also cuts out a bad stretch. A feet-up kick that "
             "grew a third leg never plays if the continuation starts before it.</p>",
             "<pre><code>uv run tools/ltx_extend.py --input people/rob.mp4 --at 4.8 \\\n"
             "    --prompt \"A fixed view that never moves ... the same man keeps laughing\" \\\n"
             "    --output people/rob-long.mp4\n# --times 2 for a third link</code></pre>"
             + fig("media/stage-dana.jpg", "Dana, an invented cast member, at her desk", "A cast portrait, the input to a clip", 704, 768),
             field=True)
     + split("learned", "Learned the hard way",
             "<ul><li><b>Never continue from a close-up.</b> With only a face to go on, the model invents a new scene.</li>"
             "<li><b>Lock the framing in words,</b> a fixed view with no zoom and no cuts, but don't say \"tripod\": it draws "
             "one.</li>"
             "<li><b>Seek before the end.</b> Grabbing a clip's very last frame with ffmpeg returns nothing.</li>"
             "<li><b>Keep what you paid for.</b> A chain that fails at link two keeps link one, so a re-run doesn't pay "
             "twice.</li>"
             "<li><b>Slow a slightly short clip</b> down to about 0.8x before faces look slow-motion; beyond that, extend.</li>"
             "</ul>",
             fig("media/stage-render.jpg", "Four colleagues in headsets, from a shot lengthened with ltx_extend.py",
                 "The Great Return: its long shots were extended, not held", 1080, 608))
     + split("qc", "Catching it automatically",
             "<p>A whole-frame check for movement misses a frozen face, because something else on screen is always moving. "
             "<code>qc.py --render</code> looks per region for motion that stops dead: part of the frame moving for three "
             "seconds or more, then frozen for two or more while the rest goes on. On four renders it found every clip "
             "freeze in the old cuts and none in the fixed ones.</p>"
             f"<p><a class=\"button\" href=\"{REPO}/blob/main/tools/ltx_extend.py\">ltx_extend.py on GitHub</a></p>",
             fig("media/band-mute.jpg", "Four people on a video call", "You're On Mute, checked with qc.py before upload", 1920, 822)))
print("built")
