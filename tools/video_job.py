#!/usr/bin/env python3
"""Brief in, MP4 out: an async video job runner for agents.

One entrypoint turns a scene spec (or a brief, through a pluggable expander) into a
finished, QC-checked MP4 without the caller holding a connection open. `submit` returns
in seconds with a job id and a cost estimate; a detached worker runs the pipeline; every
later call is short and reads state from disk, so it works from a fresh process and
survives the caller going away.

    uv run tools/video_job.py estimate --brief "..." [--format 9:16|16:9] [--duration 45] [--json]
    uv run tools/video_job.py estimate --spec spec.json --json
    uv run tools/video_job.py submit   --spec spec.json [--max-usd 2.00] [--json]
    uv run tools/video_job.py submit   --brief "..." [--format 16:9] [--duration 30] [--json]
    uv run tools/video_job.py status   JOB_ID [--json]
    uv run tools/video_job.py result   JOB_ID [--json]
    uv run tools/video_job.py list     [--json]
    uv run tools/video_job.py cancel   JOB_ID [--json]
    uv run tools/video_job.py spec     --brief "..." [--format 9:16] [--duration 45]   # expander only
    uv run tools/video_job.py presets  [--json]

Pipeline: spec -> voiceover -> visuals -> captions -> (music) -> build -> qc -> publish.
The compositor is templates/concept-explainer-short (copied into the job dir as a
project); 16:9 is the same build with a landscape config.

Spec (what an agent writes; see docs/agent-video-jobs.md):

    {"title": "Why is the sky blue?", "format": "9:16",
     "scenes": [
       {"slug": "hook", "text": "Look up on a clear day...",
        "visual": {"kind": "ltx", "prompt": "slow tilt up a clear blue sky"}},
       {"slug": "why", "text": "Air scatters short blue wavelengths...",
        "visual": {"kind": "ideogram", "prompt": "science poster, headline SHORT WAVES SCATTER"}}]}

Brief -> spec needs an LLM. Agents are the LLM, so they pass --spec. Otherwise set
VIDEO_JOB_EXPANDER to a command that reads {"brief","format","duration_s"} JSON on stdin
and prints a spec JSON on stdout (any provider; e.g. `claude -p`).

Cost: `estimate` is a pure function of the spec and the PRICES table below. `submit`
refuses over --max-usd before any spend; the worker stops before the next paid step once
actual spend would pass the cap. Receipts go to jobs/<id>/receipts.jsonl and to
MEDIA_SPEND_LOG (default logs/media-spend.log) in the same JSON-lines shape longarm's
ledger reads. One job runs at a time (jobs/.lock).

Exit codes: 0 ok, 1 error, 2 refused or not ready (over budget, no expander, job not
done), 3 another job holds the lock.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from win_encoding import ensure_utf8_output  # noqa: E402

try:  # .env keys decide which caption engine is paid and whether R2 is configured
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates" / "concept-explainer-short"
PRICES_VERSION = "2026-10-31"

# ---------------------------------------------------------------------------
# Unit prices (USD). The one table `estimate` reads. Sources: Ideogram list price
# (ideogram.ai/pricing, API tab); LTX-2.3 measured on Modal A100-80GB at $2.50/h
# (docs/ltx2.md benchmarks, 576x1024 or 1024x576 at 121 frames lands at the top of the
# range); Qwen3-TTS and FLUX.2 on Modal A10G at $1.10/h; ElevenLabs TTS at the Creator
# tier per-character rate; ElevenLabs Scribe at $0.40 per audio hour.
# ---------------------------------------------------------------------------
PRICES = {
    "version": PRICES_VERSION,
    "ideogram_image": {"TURBO": 0.03, "DEFAULT": 0.06, "QUALITY": 0.10},
    "ltx_clip": 0.20,
    "flux_image": 0.01,
    "voiceover": {"qwen3": 0.01, "elevenlabs_per_1k_chars": 0.20, "60db": 0.0},
    "captions_per_audio_second": {"scribe": 0.40 / 3600, "whisper": 0.0},
    "music_track": {"acemusic": 0.0, "modal": 0.03, "runpod": 0.03},
}

# Output formats: the compositor's canvas plus the size each generator is asked for.
# Ideogram 4.0 returns 1440x2560 and 1280x720 exactly; LTX sides must be multiples of 64;
# FLUX sides multiples of 16. Captions sit above the platform UI on vertical.
FORMATS = {
    "9:16": {
        "width": 1080, "height": 1920,
        "captions": {"y": 1640, "size": 80, "maxWords": 3, "maxChars": 22},
        "ideogram": "1440x2560", "ltx": (576, 1024), "flux": (720, 1280),
    },
    "16:9": {
        "width": 1920, "height": 1080,
        "captions": {"y": 900, "size": 64, "maxWords": 4, "maxChars": 30},
        "ideogram": "1280x720", "ltx": (1024, 576), "flux": (1280, 720),
    },
}

VISUAL_KINDS = ("ideogram", "ltx", "flux", "text", "none")
# "text" is a constructed visual: a generated text-free background (FLUX) with the scene's
# headline and an accent bar animated in by the build. Deterministic typography, motion
# from the Ken Burns drift and the reveal, about a cent. "flux" is the same background
# without the headline, and is what a scene with no visual at all gets.
BACKGROUND_STYLE = ("deep navy and electric blue, soft volumetric light, subtle abstract geometric forms, "
                    "shallow depth of field, cinematic, high detail, no text, no words, no letters, no logos, "
                    "no people, no faces, no figures")
DEFAULT_MUSIC_PROMPT = ("subtle ambient electronic bed for a short explainer video, soft warm pads, slow "
                        "pulse, minimal, no vocals, no drums, calm and modern")
TITLE_CARD_SECONDS = 2.8
WORDS_PER_SECOND = 2.4          # narration pace the template budgets with
CHARS_PER_WORD = 5.8            # for ElevenLabs per-character pricing
DEFAULT_MAX_USD = 2.00
DEFAULT_LTX_FRAMES = 121        # ~5 s at 24 fps; the build boomerang-loops it
# LTX-2 puts stray text, logos and anime styling in ~30% of clips unless told not to.
DEFAULT_LTX_NEGATIVE = ("text, letters, words, captions, subtitles, logo, watermark, anime, cartoon, "
                        "illustration, worst quality, inconsistent motion, blurry, jittery, distorted")
# Both image models put a stranger in the shot when nobody was asked for: a "slow push-in on a
# server room" came back as a woman praying at the rack, an "abstract cinematic background" as a
# portrait. A person in a video is a casting decision, so the clip prompt has to name one.
NO_PEOPLE_NEGATIVE = "people, person, man, woman, face, portrait, hands, crowd, figure, silhouette"
# LTX cannot draw software. Ask it for a terminal, a dashboard, a timeline or "code lines" and
# it paints a fake interface covered in garbled text (three of three such clips on the first
# live day). Submit refuses these before any spend unless the spec says allowScreens; the
# message tells the agent what to do instead, which prose in its instructions did not achieve.
SCREEN_WORDS = re.compile(r"\b(screens?|interface|ui|dashboards?|terminal|console|code|coding|timeline|"
                          r"export bar|progress bar|loading bar|chat window|chat interface|browser|"
                          r"website|web ?page|app|apps|application|toggle|buttons?|menu|graphs?|charts?|"
                          r"spreadsheet|login|form|popup|pop-up|notifications?|alerts?|cursor|typing|"
                          r"keyboard shortcuts?|monitor|display|window|windows|render(?:ing)? complet\w*|"
                          r"system check|status page|green checkmarks?)\b", re.I)
PEOPLE_WORDS = re.compile(r"\b(people|person|persons|man|men|woman|women|boy|girl|kid|child|children|"
                          r"face|faces|hand|hands|crowd|figure|figures|silhouette|portrait|presenter|"
                          r"narrator|someone|somebody|human|humans|he|she|they|worker|engineer|developer|"
                          r"player|gamer|family|couple|team)\b", re.I)
PHASES = ("queued", "voiceover", "visuals", "captions", "music", "build", "qc", "publish")
TERMINAL = ("done", "failed", "cancelled")
TOOL_TIMEOUT = 1500             # per tool call (LTX cold start ~3 min, VO batch longer)

# Ideogram renders text reliably only from a structured JSON caption; a plain prompt adds
# small filler text ("A REMINDER") to the card. Agents fill this template per ideogram scene.
# The bbox is [top, left, bottom, right] on a 0-1000 grid of the output image, so the same
# numbers work at 1440x2560 and 1280x720. Keep the headline under 6 words, all caps.
IDEOGRAM_CAPTION_TEMPLATE = {
    "high_level_description": "A <dark-navy poster> with the headline <HEADLINE>, showing <one simple "
                              "diagram or icon>, with clear space in the bottom fifth of the frame.",
    "style_description": {
        "aesthetics": "clean, diagrammatic, modern explainer",
        "lighting": "flat even lighting, soft glow on the main object",
        "medium": "flat vector graphic design",
        "art_style": "minimalist infographic poster, generous negative space",
        "color_palette": ["#0A1428", "#4DA8FF", "#FFB347", "#FFFFFF"],
    },
    "compositional_deconstruction": {
        "background": "Deep navy gradient, faint tiny stars.",
        "elements": [
            {"type": "text", "bbox": [80, 80, 200, 920], "text": "<HEADLINE>",
             "desc": "Bold condensed sans-serif headline, all caps, white, centered at the top.",
             "color_palette": ["#FFFFFF"]},
            {"type": "obj", "bbox": [280, 150, 760, 850],
             "desc": "<One object: what it is, its shape, its colour. No text on it.>",
             "color_palette": ["#4DA8FF", "#FFB347"]},
        ],
    },
}


def _diagram_steps(obj: str) -> list[str]:
    """'A -> B -> C' (also '\u2192' or '=>') as 2 to 4 step labels; [] when the object is not a chain."""
    s = obj.replace("\u2192", "->").replace("=>", "->")
    if "->" not in s:
        return []
    steps = [p.strip().strip('"') for p in s.split("->")]
    steps = [p for p in steps if p]
    return steps if 2 <= len(steps) <= 4 else []


def _diagram_caption(headline: str, steps: list[str], fmt: str, description: str | None = None) -> dict:
    """A simple flow diagram on the card: one labelled box per step, an arrow between each
    pair, laid out across the frame in landscape and down it in portrait. Ideogram renders
    the labels legibly because each is its own text element with a bbox; a prose request for
    "a diagram" comes back as a silhouette."""
    cap = json.loads(json.dumps(IDEOGRAM_CAPTION_TEMPLATE))
    n = len(steps)
    chain = " \u2192 ".join(steps)
    cap["high_level_description"] = (description or
        f"A dark-navy explainer poster with the headline {headline} and a simple flow diagram of "
        f"{n} labelled boxes joined by arrows, reading {chain}, with clear space in the bottom fifth.")
    head = cap["compositional_deconstruction"]["elements"][0]
    head["text"] = headline
    elements = [head]
    landscape = fmt in ("16:9", "1:1")
    box_desc = "Rounded rectangle with a thin electric-blue outline and the label '{}' in white sans-serif, centered"
    if landscape:
        gap, left0, span = 60, 100, 800
        width = (span - gap * (n - 1)) // n
        for k, step in enumerate(steps):
            left = left0 + k * (width + gap)
            elements.append({"type": "text", "bbox": [420, left, 620, left + width], "text": step,
                             "desc": box_desc.format(step), "color_palette": ["#4DA8FF", "#FFFFFF"]})
            if k < n - 1:
                elements.append({"type": "obj", "bbox": [495, left + width, 545, left + width + gap],
                                 "desc": "short amber arrow pointing right", "color_palette": ["#FFB347"]})
    else:
        gap, top0, span = 50, 280, 480
        height = (span - gap * (n - 1)) // n
        for k, step in enumerate(steps):
            top = top0 + k * (height + gap)
            elements.append({"type": "text", "bbox": [top, 150, top + height, 850], "text": step,
                             "desc": box_desc.format(step), "color_palette": ["#4DA8FF", "#FFFFFF"]})
            if k < n - 1:
                elements.append({"type": "obj", "bbox": [top + height, 470, top + height + gap, 530],
                                 "desc": "short amber arrow pointing down", "color_palette": ["#FFB347"]})
    cap["compositional_deconstruction"]["elements"] = elements
    return cap


def _caption(headline: str, obj: str, description: str) -> dict:
    """Fill IDEOGRAM_CAPTION_TEMPLATE (used by SPEC_EXAMPLE)."""
    cap = json.loads(json.dumps(IDEOGRAM_CAPTION_TEMPLATE))
    cap["high_level_description"] = description
    cap["compositional_deconstruction"]["elements"][0]["text"] = headline
    cap["compositional_deconstruction"]["elements"][1]["desc"] = obj
    return cap


SPEC_EXAMPLE = {
    "title": "Why is the sky blue?",
    "format": "9:16",
    "voice": {"provider": "qwen3", "speaker": "Ryan", "tone": "warm", "maxWpm": 165},
    "scenes": [
        {"slug": "hook",
         "text": "Look up on a clear day and you see blue in every direction. But sunlight is white.",
         "visual": {"kind": "ltx",
                    "prompt": "Slow tilt up from a green field to a clear blue sky, soft clouds, cinematic"}},
        {"slug": "scattering",
         "text": "Air molecules scatter short blue wavelengths far more than long red ones.",
         "visual": {"kind": "ideogram",
                    "caption": _caption(
                        "SHORT WAVES SCATTER MOST",
                        "A beam of white sunlight entering from the upper left and splitting; short "
                        "blue zigzag waves bouncing off a cluster of small glowing spheres in every "
                        "direction, one long smooth red wave passing straight through.",
                        "A vertical dark-navy science poster with the headline SHORT WAVES SCATTER "
                        "MOST, showing white sunlight splitting and blue waves scattering off air "
                        "molecules while a red wave passes through, with clear space in the bottom fifth.")}},
        {"slug": "cta",
         "text": "So the sky isn't blue. It's blue light, scattered. Follow for one concept a week.",
         "visual": {"kind": "ideogram",
                    "caption": _caption(
                        "ONE CONCEPT A WEEK",
                        "A small amber circle with a white play-button triangle inside it, centered.",
                        "A vertical dark-navy card with the headline ONE CONCEPT A WEEK above a small "
                        "amber play button, nothing else, with clear space in the bottom fifth.")}},
    ],
}


# ---------------------------------------------------------------------------
# Paths and small helpers
# ---------------------------------------------------------------------------

def jobs_dir() -> Path:
    return Path(os.environ.get("VIDEO_JOBS_DIR") or ROOT / "jobs")


def spend_log_path() -> Path:
    """MEDIA_SPEND_LOG, else the agent workspace's ledger when one is set (longarm exports
    LONGARM_WORKSPACE into every tool call), else the toolkit's own logs/."""
    if os.environ.get("MEDIA_SPEND_LOG"):
        return Path(os.environ["MEDIA_SPEND_LOG"])
    if os.environ.get("LONGARM_WORKSPACE"):
        return Path(os.environ["LONGARM_WORKSPACE"]) / "logs" / "media-spend.log"
    return ROOT / "logs" / "media-spend.log"


def reserve_at_submit() -> bool:
    """When true, `submit` books the whole estimate in the shared ledger as one receipt (a
    reservation the daily cap can count at once) and the worker keeps its per-tool receipts
    in the job directory only. On by default under longarm, whose cap sums the ledger before
    every call; off elsewhere so the ledger holds actuals."""
    v = os.environ.get("VIDEO_JOB_RESERVE_AT_SUBMIT")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(os.environ.get("LONGARM_WORKSPACE"))


def emit_cost_event(usd: float, msg: str) -> None:
    """The `cost` progress line longarm's video-spend.py wrapper turns into a receipt. Only
    printed under --progress json, so plain CLI use stays quiet."""
    print(json.dumps({"ts": time.strftime("%H:%M:%S"), "stage": "cost",
                      "msg": f"Est. cost: ${usd:.4f} ({msg})", "pct": None, "elapsed": 0.0}),
          file=sys.stderr, flush=True)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str, fallback: str = "scene") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or fallback


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid: int | None) -> bool:
    """Cross-platform liveness check. Never signals the process."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Spec: normalise, validate, skeleton from a brief, estimate
# ---------------------------------------------------------------------------

class SpecError(ValueError):
    pass


def _spoken_key(s: str) -> str:
    """Words only, lowercased: 'THE PRICE GAP ; cheap, dear' == 'the price gap cheap dear'."""
    return " ".join(re.findall(r"[a-z0-9']+", (s or "").lower()))


def check_narration_is_spoken(scene_id: str, text: str, vis: dict) -> None:
    """The narration is what the voice says; the headline is what the screen shows. An agent
    that writes the headline into the narration field gets its voice reading 'THE PRICE GAP
    intelligence cheap hardware dear' aloud over a card that already says it (seen live on
    2026-11-01). Refuse the two shapes that produce it: narration equal to the headline (with
    or without the subline), and narration that is mostly capitals."""
    words = text.split()
    if len(words) >= 3:
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            raise SpecError(f"scene {scene_id}: the narration is written in capitals, like a headline. The "
                            "narration is the sentence you say; put the headline after ' :: text: ' instead.")
    cap = vis.get("caption")
    head = vis.get("headline") or (cap.get("headline") if isinstance(cap, dict) else None)
    if not head:
        return
    sub = vis.get("subline") or ""
    key = _spoken_key(text)
    if key and key in (_spoken_key(head), _spoken_key(f"{head} {sub}")):
        raise SpecError(f"scene {scene_id}: the narration repeats the headline word for word. The narration "
                        "is what you say (a sentence or two, said plainly); the headline is the 2 to 5 words "
                        "shown on screen. Write the spoken line.")


def auto_background_prompt(text: str, hint: str | None = None) -> str:
    """A text-free FLUX background that evokes the narration. The words themselves never
    go into the image; captions and the headline overlay carry them."""
    if not hint:
        # the first sentence, not the first 18 words: a gist cut mid-clause ("But the
        # constraint was.") reads as nonsense to the image model
        first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        hint = " ".join(first.split()[:18])
    gist = hint.strip().rstrip(".!?,;:")
    return f"Abstract cinematic background evoking: {gist}. {BACKGROUND_STYLE}"


def load_voice_preset(name: str) -> dict:
    """assets/voices/<name>.json (a clone: refAudio + refText + provider), else a brand's
    voice.json by name. Paths inside are relative to the toolkit root."""
    pf = ROOT / "assets" / "voices" / f"{name}.json"
    if pf.exists():
        v = json.loads(pf.read_text(encoding="utf-8"))
        v.pop("note", None)
        return v
    if (ROOT / "brands" / name / "voice.json").exists():
        return {"brand": name}
    raise SpecError(f"unknown voice preset {name!r}: no assets/voices/{name}.json and no brands/{name}/")


def normalize_spec(spec: dict) -> dict:
    """Fill defaults, derive ids and asset paths, and validate. Returns a new dict."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    out = json.loads(json.dumps(spec))  # deep copy
    fmt = str(out.get("format") or "9:16")
    if fmt not in FORMATS:
        raise SpecError(f"format must be one of {', '.join(FORMATS)} (got {fmt!r})")
    out["format"] = fmt
    out["title"] = str(out.get("title") or "Untitled").strip()
    scenes = out.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise SpecError("spec.scenes must be a non-empty list")
    voice = out.get("voice") or {}
    if isinstance(voice, str):
        voice = load_voice_preset(voice)
    if not isinstance(voice, dict):
        raise SpecError("spec.voice must be an object or a preset name")
    voice.setdefault("provider", "qwen3")
    if voice["provider"] not in ("qwen3", "elevenlabs", "60db"):
        raise SpecError("spec.voice.provider must be qwen3, elevenlabs or 60db")
    voice.setdefault("maxWpm", 165)
    out["voice"] = voice
    caps = out.get("captions") or {}
    caps.setdefault("enabled", True)
    caps.setdefault("engine", "auto")
    out["captions"] = caps
    music = out.get("music")
    if music is None or music is True:
        music = {}
    elif music is False:
        music = {"enabled": False}
    elif isinstance(music, str):
        music = {"prompt": music}
    music.setdefault("enabled", True)        # free on the acemusic API; the build ducks it
    music.setdefault("provider", "acemusic")
    music.setdefault("prompt", DEFAULT_MUSIC_PROMPT)
    out["music"] = music
    tc = out.get("titleCard", True)
    out["titleCard"] = ({"text": out["title"], "seconds": TITLE_CARD_SECONDS} if tc is True
                        else ({**tc, "seconds": tc.get("seconds", TITLE_CARD_SECONDS)} if isinstance(tc, dict) else None))
    draft = bool(out.get("draft"))
    ideo = out.get("ideogram") or {}
    ideo.setdefault("speed", "DEFAULT")
    if ideo["speed"] not in PRICES["ideogram_image"]:
        raise SpecError("spec.ideogram.speed must be TURBO, DEFAULT or QUALITY")
    out["ideogram"] = ideo

    seen = set()
    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            raise SpecError(f"scene {i + 1} must be an object")
        s["id"] = f"{i + 1:02d}"
        slug = _slugify(s.get("slug") or (s.get("text") or "")[:24], f"scene{i + 1}")
        while slug in seen:
            slug += "x"
        seen.add(slug)
        s["slug"] = slug
        text = (s.get("text") or "").strip()
        if not text and not s.get("words"):
            raise SpecError(f"scene {s['id']} ({slug}) has no narration text")
        s["text"] = text
        vis = s.get("visual")
        if isinstance(vis, str):
            vis = {"kind": vis}
        if vis is None:
            # no visual named: a generated background (about a cent) unless this is a draft
            vis = {"kind": "none"} if draft else {"kind": "flux", "prompt": auto_background_prompt(text)}
        if not isinstance(vis, dict):
            raise SpecError(f"scene {s['id']}: visual must be an object")
        kind = vis.get("kind") or "none"
        if kind not in VISUAL_KINDS:
            raise SpecError(f"scene {s['id']}: visual.kind must be one of {', '.join(VISUAL_KINDS)}")
        if kind == "text":
            if not vis.get("headline"):
                raise SpecError(f"scene {s['id']}: visual.text needs a headline")
            vis["headline"] = str(vis["headline"]).strip()
            vis["prompt"] = vis.get("prompt") or auto_background_prompt(text, vis.get("background"))
        if kind != "none" and not (vis.get("prompt") or vis.get("caption")):
            raise SpecError(f"scene {s['id']}: visual.{kind} needs a prompt (or a caption for ideogram)")
        check_narration_is_spoken(s["id"], text, vis)
        if kind == "ltx" and len((vis.get("prompt") or "").split()) < 3 and not str(vis.get("prompt", "")).startswith("("):
            raise SpecError(f"scene {s['id']}: the {kind} prompt is '{vis.get('prompt')}', which is not a shot. "
                            "Describe the thing, what moves and the light in a short sentence, e.g. "
                            "'slow push-in on a brass kettle, steam rising, warm kitchen light'.")
        if kind == "ltx" and not out.get("allowScreens"):
            hit = SCREEN_WORDS.search(vis.get("prompt") or "")
            if hit:
                raise SpecError(
                    f"scene {s['id']}: the clip prompt asks for a screen ('{hit.group(0)}'), and the video model "
                    "draws fake software covered in garbled text. Film a physical thing and its light instead "
                    "(the room, the desk, an object, a hand on a key), or make this a 'text:' scene with the idea "
                    "as the headline. To insist, set allowScreens in the spec (--allow-screens).")
        cap = vis.get("caption")
        if kind == "ideogram" and isinstance(cap, dict) and "headline" in cap:
            # short form {headline, object, description}: expand into the full template.
            # An object written as a chain ('A -> B -> C') becomes a labelled flow diagram.
            headline = str(cap["headline"]).strip()
            obj = str(cap.get("object") or "a single simple icon, centered").strip()
            steps = _diagram_steps(obj)
            if steps:
                vis["caption"] = _diagram_caption(headline, steps, out.get("format") or "9:16",
                                                  cap.get("description"))
            else:
                vis["caption"] = _caption(headline, obj,
                                          str(cap.get("description") or
                                              f"A dark-navy poster with the headline {headline}, "
                                              "one simple object below it, clear space in the bottom fifth.").strip())
        vis["kind"] = kind
        s["visual"] = vis
        if kind == "ltx":
            s["asset"] = f"clips/{s['id']}_{slug}.mp4"
        elif kind in ("ideogram", "flux", "text"):
            s["asset"] = f"images/{s['id']}_{slug}.png"
        else:
            s["asset"] = ""
        # an animated headline the build lays over the visual (text kind always; any
        # non-ideogram kind may carry one)
        if kind == "text":
            s["overlay"] = {"headline": vis["headline"], "subline": (vis.get("subline") or "").strip()}
        elif kind != "ideogram" and vis.get("headline"):
            s["overlay"] = {"headline": str(vis["headline"]).strip(), "subline": (vis.get("subline") or "").strip()}
    out["scenes"] = scenes
    return out


def skeleton_spec(brief: str, fmt: str = "9:16", duration_s: int = 45) -> dict:
    """A deterministic, LLM-free spec shape for estimating a brief: N scenes alternating
    motion and card, each with the narration word budget the duration implies."""
    n = max(2, min(8, round(duration_s / 12)))
    words = max(8, round(duration_s * WORDS_PER_SECOND / n))
    scenes = []
    for i in range(n):
        kind = "ltx" if i % 2 == 0 else "text"
        vis = {"kind": kind, "prompt": "(from brief)"}
        if kind == "text":
            vis["headline"] = "(from brief)"
        scenes.append({"slug": f"scene{i + 1}", "words": words, "text": "", "visual": vis})
    return {"title": brief.strip()[:80] or "Untitled", "format": fmt,
            "brief": brief, "duration_s": duration_s, "scenes": scenes}


def _scene_words(scene: dict) -> int:
    if scene.get("text"):
        return len(scene["text"].split())
    return int(scene.get("words") or 0)


def estimate(spec: dict) -> dict:
    """Pure function of the spec and PRICES. Deterministic: 'auto' captions are priced as
    Scribe (the paid engine), the upper bound."""
    spec = normalize_spec(spec)
    scenes = spec["scenes"]
    words = sum(_scene_words(s) for s in scenes)
    narration_s = round(words / WORDS_PER_SECOND, 1)
    rows = []

    provider = spec["voice"]["provider"]
    if provider == "elevenlabs":
        chars = sum(len(s["text"]) if s.get("text") else _scene_words(s) * CHARS_PER_WORD
                    for s in scenes)
        rows.append({"step": "voiceover", "unit": "1k chars (elevenlabs)",
                     "count": round(chars / 1000, 2),
                     "unit_usd": PRICES["voiceover"]["elevenlabs_per_1k_chars"],
                     "usd": chars / 1000 * PRICES["voiceover"]["elevenlabs_per_1k_chars"]})
    else:
        rows.append({"step": "voiceover", "unit": f"scene ({provider})", "count": len(scenes),
                     "unit_usd": PRICES["voiceover"][provider],
                     "usd": len(scenes) * PRICES["voiceover"][provider]})

    kinds = {k: sum(1 for s in scenes if s["visual"]["kind"] == k) for k in VISUAL_KINDS}
    if kinds["ideogram"]:
        unit = PRICES["ideogram_image"][spec["ideogram"]["speed"]]
        rows.append({"step": "visuals", "unit": f"ideogram image ({spec['ideogram']['speed']})",
                     "count": kinds["ideogram"], "unit_usd": unit, "usd": kinds["ideogram"] * unit})
    if kinds["ltx"]:
        rows.append({"step": "visuals", "unit": "ltx clip", "count": kinds["ltx"],
                     "unit_usd": PRICES["ltx_clip"], "usd": kinds["ltx"] * PRICES["ltx_clip"]})
    n_flux = kinds["flux"] + kinds["text"]
    if n_flux:
        rows.append({"step": "visuals", "unit": "flux background", "count": n_flux,
                     "unit_usd": PRICES["flux_image"], "usd": n_flux * PRICES["flux_image"]})

    if spec["captions"]["enabled"]:
        engine = spec["captions"]["engine"]
        eng = "whisper" if engine == "whisper" else "scribe"
        unit = PRICES["captions_per_audio_second"][eng]
        rows.append({"step": "captions", "unit": f"audio second ({eng})", "count": narration_s,
                     "unit_usd": round(unit, 6), "usd": narration_s * unit})

    if spec["music"]["enabled"]:
        prov = spec["music"]["provider"]
        unit = PRICES["music_track"].get(prov, 0.03)
        rows.append({"step": "music", "unit": f"track ({prov})", "count": 1, "unit_usd": unit, "usd": unit})

    for r in rows:
        r["usd"] = round(r["usd"], 4)
    total = round(sum(r["usd"] for r in rows), 4)
    return {
        "estimate_usd": total,
        "breakdown": rows,
        "scenes": len(scenes),
        "narration_words": words,
        "narration_s": narration_s,
        "video_s_approx": round(narration_s + len(scenes) * 1.2 + 0.7, 1),
        "format": spec["format"],
        "prices_version": PRICES_VERSION,
    }


def step_estimates(spec: dict) -> dict:
    """Estimated spend per pipeline step, for the worker's pre-step budget check."""
    est = estimate(spec)
    by_step: dict[str, float] = {}
    for r in est["breakdown"]:
        by_step[r["step"]] = round(by_step.get(r["step"], 0.0) + r["usd"], 4)
    return by_step


# ---------------------------------------------------------------------------
# Brief -> spec (pluggable expander)
# ---------------------------------------------------------------------------

class NoExpander(RuntimeError):
    pass


def expand_brief(brief: str, fmt: str, duration_s: int, timeout: int = 300) -> dict:
    """Run the VIDEO_JOB_EXPANDER command. It reads a JSON request on stdin and prints a
    spec JSON on stdout (code fences tolerated). Any provider: the toolkit doesn't care."""
    cmd = os.environ.get("VIDEO_JOB_EXPANDER", "").strip()
    if not cmd:
        raise NoExpander(
            "No brief expander configured. Pass a finished --spec spec.json (recommended: the "
            "agent is the LLM), or set VIDEO_JOB_EXPANDER to a command that reads "
            '{"brief","format","duration_s"} on stdin and prints a spec JSON. '
            "See docs/agent-video-jobs.md.")
    request = {
        "brief": brief, "format": fmt, "duration_s": duration_s,
        "words_budget": round(duration_s * WORDS_PER_SECOND),
        "spec_example": SPEC_EXAMPLE,
        "ideogram_caption_template": IDEOGRAM_CAPTION_TEMPLATE,
        "instructions": (
            "Write a spec JSON for a short explainer video. Keep total narration near "
            "words_budget. Alternate visual kinds: 'ltx' (motion b-roll; prompt describes motion "
            "and atmosphere, never on-screen text) and 'ideogram' (a designed card). Every ideogram "
            "scene carries a 'caption' object filled from ideogram_caption_template (replace the "
            "<...> placeholders, keep the headline under 6 words, all caps); never a plain prompt, "
            "which adds filler text to the card. Scene 1 must hook in one sentence. Output only JSON."),
    }
    args = shlex.split(cmd, posix=(os.name != "nt"))
    r = subprocess.run(args, input=json.dumps(request), capture_output=True, text=True,
                       timeout=timeout, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"expander exited {r.returncode}: {r.stderr.strip()[-800:]}")
    text = r.stdout.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(f"expander printed no JSON object: {text[:300]!r}")
    spec = json.loads(m.group(0))
    spec.setdefault("format", fmt)
    spec.setdefault("brief", brief)
    return normalize_spec(spec)


def _workspace_paths(rel: str) -> list[Path]:
    """Where a relative spec path may live: the agent workspace first, then cwd, then here."""
    out = []
    if os.environ.get("LONGARM_WORKSPACE"):
        out.append(Path(os.environ["LONGARM_WORKSPACE"]) / rel)
    out += [Path.cwd() / rel, ROOT / rel]
    return out


def read_spec_file(path: str) -> dict:
    if path == "-":
        return parse_spec_text(sys.stdin.read())
    p = Path(path)
    candidates = [p] if p.is_absolute() else _workspace_paths(path)
    for c in candidates:
        if c.exists():
            return parse_spec_text(c.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"spec file not found: {path} (looked in "
                            f"{', '.join(str(c.parent) for c in candidates)})")


def parse_spec_text(text: str) -> dict:
    """JSON, with the slips language models make repaired before giving up: code fences,
    a \\' escape (not valid JSON), trailing commas, and Python-literal dicts (single
    quotes, True/None). A string starting with @ names a file instead."""
    t = text.strip()
    if t.startswith("@"):
        return read_spec_file(t[1:])
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        t = m.group(0)
    attempts = [t, t.replace("\\'", "'"),
                re.sub(r",\s*([}\]])", r"\1", t.replace("\\'", "'"))]
    last: Exception | None = None
    for cand in attempts:
        try:
            return json.loads(cand)
        except ValueError as e:
            last = e
    try:
        import ast
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass
    pos = getattr(last, "pos", None)
    snippet = f" near: ...{t[max(0, pos - 40):pos + 20]}..." if isinstance(pos, int) else ""
    raise SpecError(f"spec is not valid JSON ({last}){snippet}. Tip: write it to a file and pass "
                    "--spec-json @path/to/spec.json")


def spec_from_scene_args(title: str | None, scenes: list[str], fmt: str, voice: str | None = None,
                         music: str | bool | None = None, draft: bool = False, title_card: bool = True,
                         allow_screens: bool = False) -> dict:
    """The flat form, for callers that can't reliably write nested JSON (an open model
    with no editor). One string per scene:

        "narration text"                                  -> placeholder card
        "narration text :: ltx: slow pan over a harbour"   -> motion clip
        "narration text :: card: ONE EXTRA DAY ; an amber calendar icon showing 29"
                                                          -> ideogram card (headline ; object)
        "narration text :: card: THE LOOP ; prompt -> agent -> video"
                                                          -> ideogram card with a labelled flow diagram
        "narration text :: flux: soft navy gradient"       -> text-free still
        "narration text :: text: ONE EXTRA DAY ; why February grows"
                                                          -> generated background + animated headline (; subline)

    The separator is ' :: ' and the kind is the word before the first colon after it."""
    out_scenes = []
    for i, raw in enumerate(scenes):
        text, _, vis = raw.partition(" :: ")
        text = text.strip()
        vis = vis.strip()
        if not text:
            raise SpecError(f"--scene {i + 1}: narration text is empty")
        visual: dict = {"kind": "none"}
        if vis:
            kind, _, rest = vis.partition(":")
            kind = kind.strip().lower()
            rest = rest.strip()
            if kind in ("card", "ideogram"):
                headline, _, obj = rest.partition(";")
                headline = headline.strip().strip('"').upper()
                if not headline:
                    raise SpecError(f"--scene {i + 1}: card needs a headline, e.g. 'card: ONE EXTRA DAY ; an amber calendar'")
                visual = {"kind": "ideogram", "caption": {"headline": headline,
                                                          "object": obj.strip() or "a single simple icon, centered"}}
            elif kind in ("text", "headline", "title"):
                headline, _, sub = rest.partition(";")
                headline = headline.strip().strip('"')
                if not headline:
                    raise SpecError(f"--scene {i + 1}: text needs a headline, e.g. 'text: ONE EXTRA DAY ; a subline'")
                visual = {"kind": "text", "headline": headline, "subline": sub.strip()}
            elif kind in ("ltx", "clip", "video"):
                visual = {"kind": "ltx", "prompt": rest}
            elif kind in ("flux", "still", "background"):
                visual = {"kind": "flux", "prompt": rest}
            elif kind in ("none", "placeholder"):
                visual = {"kind": "none"}
            else:
                raise SpecError(f"--scene {i + 1}: unknown visual '{kind}' (use ltx:, card:, flux: or none)")
            if visual["kind"] in ("ltx", "flux") and not visual.get("prompt"):
                raise SpecError(f"--scene {i + 1}: {kind} needs a prompt after the colon")
        out_scenes.append({"text": text, "visual": visual})
    spec = {"title": title or out_scenes[0]["text"][:60], "format": fmt, "scenes": out_scenes,
            "draft": draft, "titleCard": title_card, "allowScreens": allow_screens}
    if voice:
        spec["voice"] = voice
    if music is not None:
        spec["music"] = music
    return spec


def resolve_spec(brief: str | None, spec_path: str | None, fmt: str, duration_s: int,
                 allow_skeleton: bool) -> tuple[dict, str]:
    """Returns (spec, source) where source is 'spec', 'expander' or 'skeleton'."""
    if spec_path:
        return normalize_spec(read_spec_file(spec_path)), "spec"
    if not brief:
        raise SpecError("pass --brief or --spec")
    try:
        return expand_brief(brief, fmt, duration_s), "expander"
    except NoExpander:
        if allow_skeleton:
            return normalize_spec(skeleton_spec(brief, fmt, duration_s)), "skeleton"
        raise


# ---------------------------------------------------------------------------
# Job state on disk
# ---------------------------------------------------------------------------

def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", job_id or ""):
        raise SpecError(f"not a job id: {job_id!r}")
    return jobs_dir() / job_id


def load_job(job_id: str) -> dict:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        raise FileNotFoundError(f"no such job: {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict) -> None:
    job["updated"] = _now()
    _write_json_atomic(job_dir(job["id"]) / "job.json", job)


def log_line(job: dict, msg: str) -> None:
    with open(job_dir(job["id"]) / "job.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def set_phase(job: dict, phase: str, progress: int | None = None, **fields) -> None:
    job["phase"] = phase
    if progress is not None:
        job["progress"] = progress
    job.update(fields)
    save_job(job)
    log_line(job, f"phase={phase} progress={job.get('progress')}")
    if phase in TERMINAL:
        settle_reservation(job)


def settle_reservation(job: dict) -> float | None:
    """Under the reserve ledger the whole estimate was booked at submit. When the job ends,
    append one correcting line (actual minus reserved, usually negative) so the shared ledger,
    which the daily cap sums naively, holds what was really spent. Without this a cancelled
    job keeps its full estimate against the cap: two cancelled takes on 2026-11-01 reserved
    $2.24, spent $0.14, and locked the agent out for the day. Idempotent via job["settled"];
    returns the delta written, or None when there was nothing to settle."""
    if job.get("ledger") != "reserve" or job.get("phase") not in TERMINAL or "settled" in job:
        return None
    reserved = float(job.get("estimate_usd") or 0.0)
    actual = float(job.get("actual_usd") or 0.0)
    delta = round(actual - reserved, 4)
    if delta:
        entry = {"ts": _now(), "tool": "video_job.py", "cost_usd": delta, "elapsed_s": 0.0, "exit": 0,
                 "args": ["settle", job["id"], job["phase"]], "job_id": job["id"],
                 "reserved_usd": round(reserved, 4), "actual_usd": round(actual, 4)}
        path = spend_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    job["settled"] = delta
    save_job(job)
    log_line(job, f"settled: reserved ${reserved:.2f} actual ${actual:.2f} ledger {delta:+.2f}")
    return delta


def write_receipt(job: dict, tool: str, cost_usd: float | None, elapsed_s: float | None,
                  exit_code: int, args: list[str]) -> None:
    """Same shape as longarm's video-spend.py ledger, plus job_id."""
    entry = {"ts": _now(), "tool": tool, "cost_usd": cost_usd, "elapsed_s": elapsed_s,
             "exit": exit_code, "args": args, "job_id": job["id"]}
    paths = [job_dir(job["id"]) / "receipts.jsonl"]
    if job.get("ledger") != "reserve":
        paths.append(spend_log_path())
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    if cost_usd:
        job["actual_usd"] = round(job.get("actual_usd", 0.0) + cost_usd, 4)
        save_job(job)


# ---------------------------------------------------------------------------
# The lock: one job at a time
# ---------------------------------------------------------------------------

class LockHeld(RuntimeError):
    def __init__(self, holder: dict):
        super().__init__(f"another job holds the lock: {holder.get('job_id')} (pid {holder.get('pid')})")
        self.holder = holder


def lock_path() -> Path:
    return jobs_dir() / ".lock"


def _read_lock() -> dict | None:
    try:
        return json.loads(lock_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lock_is_stale(holder: dict) -> bool:
    """Stale when the job it names is finished, or its worker is known and dead, or no
    worker ever reported in and the lock is old."""
    try:
        job = load_job(holder.get("job_id", ""))
    except (FileNotFoundError, SpecError):
        return True
    if job.get("phase") in TERMINAL:
        return True
    pid = job.get("worker_pid")
    if pid:
        return not pid_alive(pid)
    return time.time() - float(holder.get("ts", 0)) > 120


def acquire_lock(job_id: str, pid: int | None = None) -> None:
    jobs_dir().mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock_path(), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = _read_lock()
            if holder is None or _lock_is_stale(holder):
                try:
                    lock_path().unlink()
                except FileNotFoundError:
                    pass
                continue
            raise LockHeld(holder)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "pid": pid or os.getpid(), "ts": time.time()}, f)
        return
    raise LockHeld(_read_lock() or {})


def update_lock_pid(job_id: str, pid: int) -> None:
    holder = _read_lock()
    if holder and holder.get("job_id") == job_id:
        _write_json_atomic(lock_path(), {**holder, "pid": pid})


def release_lock(job_id: str) -> None:
    holder = _read_lock()
    if holder and holder.get("job_id") == job_id:
        try:
            lock_path().unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Running toolkit tools from the worker
# ---------------------------------------------------------------------------

_COST_RE = re.compile(r"cost:?\s*\$([0-9]+(?:\.[0-9]+)?)", re.I)


def run_tool(job: dict, tool: str, args: list[str], cwd: Path | None = None,
             timeout: int = TOOL_TIMEOUT, fallback_cost: float | None = None) -> dict:
    """Run tools/<tool> under the worker's interpreter, tee its output into the job log,
    pick the cost out of its output (the `cost` progress event or any 'Est. cost: $x'
    line), and write a receipt. Returns {returncode, stdout, stderr, cost, elapsed}."""
    script = ROOT / "tools" / tool if not tool.endswith("build.py") else Path(tool)
    cmd = [sys.executable, str(script)] + args
    log_line(job, "run: " + " ".join(shlex.quote(a) for a in cmd[1:])[:600])
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        rc, out, err = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        rc = 124
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        err += f"\n[timeout after {timeout}s]"
    elapsed = round(time.time() - t0, 1)
    with open(job_dir(job["id"]) / "job.log", "a", encoding="utf-8") as f:
        for stream, text in (("stdout", out), ("stderr", err)):
            tail = text.strip().splitlines()[-40:]
            for line in tail:
                f.write(f"    [{stream}] {line.rstrip()[:400]}\n")
    costs = [float(m.group(1)) for m in _COST_RE.finditer(err + "\n" + out)]
    cost = round(sum(costs), 4) if costs else (fallback_cost if rc == 0 else None)
    if Path(tool).name not in ("build.py", "qc.py"):  # free local steps keep out of the ledger
        write_receipt(job, Path(tool).name, cost, elapsed, rc, args)
    log_line(job, f"done: {Path(tool).name} exit={rc} {elapsed}s cost={cost}")
    return {"returncode": rc, "stdout": out, "stderr": err, "cost": cost, "elapsed": elapsed}


class BudgetExceeded(RuntimeError):
    pass


class StepFailed(RuntimeError):
    pass


def check_budget(job: dict, step: str, step_usd: float) -> None:
    cap = job.get("max_usd")
    if cap is None or step_usd <= 0:
        return
    if job.get("actual_usd", 0.0) + step_usd > cap + 1e-9:
        raise BudgetExceeded(
            f"stopping before {step}: spent ${job.get('actual_usd', 0.0):.2f}, next step ~${step_usd:.2f}, "
            f"cap ${cap:.2f}")


# ---------------------------------------------------------------------------
# Pipeline steps. Each takes the job and its project dir and returns a summary dict.
# ---------------------------------------------------------------------------

def _project(job: dict) -> Path:
    return Path(job["project_dir"])


def create_project(job: dict, spec: dict) -> Path:
    """Copy the template into the job dir and write scenes.json + config.json from the spec."""
    dest = job_dir(job["id"]) / "project"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TEMPLATE_DIR, dest,
                    ignore=shutil.ignore_patterns(".text_cache", "out", "audio", "captions",
                                                  "images", "clips", "__pycache__", "vo_durations.json"))
    for sub in ("audio/scenes", "captions", "images", "clips", "prompts", "out"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    fmt = FORMATS[spec["format"]]
    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    config["format"] = {"width": fmt["width"], "height": fmt["height"], "fps": 30}
    config["captions"] = {**config.get("captions", {}), **fmt["captions"],
                          "enabled": bool(spec["captions"]["enabled"])}
    voice = {**config.get("voice", {}), **spec["voice"]}
    if spec.get("brand"):
        voice["brand"] = spec["brand"]
    config["voice"] = voice
    if spec.get("palette"):
        config["palette"] = {**config["palette"], **spec["palette"]}
    _write_json_atomic(dest / "config.json", config)

    config["music"] = {**config.get("music", {}), "volume": 0.12}
    config["timing"] = {**config.get("timing", {}), "startPad": 0.0}   # frame 0 is the cover
    _write_json_atomic(dest / "config.json", config)

    scenes = []
    for s in spec["scenes"]:
        row = {"id": s["id"], "slug": s["slug"], "visual": s["visual"]["kind"],
               "asset": s["asset"], "text": s["text"]}
        if s.get("overlay"):
            row["overlay"] = s["overlay"]
        scenes.append(row)
    _write_json_atomic(dest / "scenes.json", {"title": spec["title"], "titleCard": spec.get("titleCard"),
                                              "scenes": scenes})
    return dest


def step_voiceover(job: dict, spec: dict) -> dict:
    proj = _project(job)
    voice = spec["voice"]
    scene_dir = proj / "audio" / "scenes"
    for s in spec["scenes"]:
        (scene_dir / f"{s['id']}_{s['slug']}.txt").write_text(s["text"], encoding="utf-8")
    args = ["--provider", voice["provider"], "--scene-dir", str(scene_dir), "--json"]
    if voice.get("maxWpm"):
        args += ["--max-wpm", str(voice["maxWpm"])]
    if voice.get("brand"):
        args += ["--brand", voice["brand"]]
    if voice["provider"] == "qwen3":
        args += ["--cloud", voice.get("cloud", "modal")]
        if voice.get("refAudio"):
            ref = Path(voice["refAudio"])
            if not ref.is_absolute():
                ref = ROOT / ref
            if not voice.get("refText"):
                raise StepFailed("voice.refText is required with voice.refAudio")
            args += ["--ref-audio", str(ref), "--ref-text", voice["refText"]]
        else:
            args += ["--speaker", voice.get("speaker", "Ryan")]
            if voice.get("tone"):
                args += ["--tone", voice["tone"]]
    elif voice["provider"] == "elevenlabs" and voice.get("voiceId"):
        args += ["--voice-id", voice["voiceId"]]
    per_scene = step_estimates(spec).get("voiceover", 0.0)
    r = run_tool(job, "voiceover.py", args, fallback_cost=per_scene)
    if r["returncode"] != 0:
        raise StepFailed(f"voiceover.py exited {r['returncode']}: {r['stderr'].strip()[-600:]}")
    try:
        result = json.loads(r["stdout"])
    except ValueError:
        raise StepFailed(f"voiceover.py printed no JSON: {r['stdout'][-300:]!r}")
    durations, pacing = {}, {}
    for item in result.get("scenes", []):
        name = Path(item.get("output", "")).stem
        if not item.get("success"):
            raise StepFailed(f"scene {name} failed: {item.get('error', '?')}")
        sid = name.split("_", 1)[0]
        durations[sid] = item["duration_seconds"]
        pacing[sid] = {"wpm": item.get("wpm"), "pacing": item.get("pacing")}
    missing = [s["id"] for s in spec["scenes"] if s["id"] not in durations]
    if missing:
        raise StepFailed(f"no audio for scenes {missing}")
    _write_json_atomic(proj / "vo_durations.json", durations)
    return {"durations": durations, "pacing": pacing, "narration_s": round(sum(durations.values()), 2)}


def _visual_args(spec: dict, s: dict, proj: Path) -> tuple[str, list[str], float]:
    """(tool, args, fallback cost) for one scene's visual."""
    fmt = FORMATS[spec["format"]]
    vis = s["visual"]
    out = proj / s["asset"]
    kind = vis["kind"]
    if kind == "ideogram":
        speed = vis.get("speed") or spec["ideogram"]["speed"]
        args = ["--output", str(out), "--resolution", vis.get("resolution") or fmt["ideogram"],
                "--speed", speed, "--json-out"]
        if vis.get("caption"):
            cap_path = proj / "prompts" / f"{s['id']}_{s['slug']}.json"
            cap_path.write_text(json.dumps(vis["caption"], indent=1), encoding="utf-8")
            args = ["--json", str(cap_path)] + args
        else:
            args = ["--prompt", vis["prompt"]] + args
        if spec.get("brand"):
            args += ["--brand", spec["brand"]]
        if spec["ideogram"].get("model"):
            args += ["--model", str(spec["ideogram"]["model"])]
        return "ideogram4.py", args, PRICES["ideogram_image"][speed]
    if kind == "ltx":
        w, h = fmt["ltx"]
        args = ["--prompt", vis["prompt"], "--width", str(w), "--height", str(h),
                "--num-frames", str(vis.get("numFrames") or DEFAULT_LTX_FRAMES),
                "--output", str(out), "--no-open", "--json", "--progress", "json", "--cloud", "modal"]
        negative = vis.get("negativePrompt") or DEFAULT_LTX_NEGATIVE
        if not vis.get("negativePrompt") and not PEOPLE_WORDS.search(vis.get("prompt") or ""):
            negative += ", " + NO_PEOPLE_NEGATIVE
        args += ["--negative-prompt", negative]
        if vis.get("seed") is not None:
            args += ["--seed", str(vis["seed"])]
        if vis.get("quality"):
            args += ["--quality", vis["quality"]]
        return "ltx2.py", args, PRICES["ltx_clip"]
    if kind in ("flux", "text"):
        w, h = fmt["flux"]
        args = ["--prompt", vis["prompt"], "--width", str(w), "--height", str(h),
                "--output", str(out), "--no-open", "--json", "--progress", "json",
                "--cloud", vis.get("cloud", "modal")]
        if vis.get("seed") is not None:
            args += ["--seed", str(vis["seed"])]
        return "flux2.py", args, PRICES["flux_image"]
    raise ValueError(kind)


def step_visuals(job: dict, spec: dict) -> dict:
    proj = _project(job)
    todo = [s for s in spec["scenes"] if s["visual"]["kind"] != "none"]
    made, n = [], len(todo)
    for i, s in enumerate(todo):
        tool, args, unit = _visual_args(spec, s, proj)
        check_budget(job, f"visual {s['id']} ({tool})", unit)
        set_phase(job, "visuals", 30 + int(45 * i / max(n, 1)), step_detail=f"{s['id']}_{s['slug']} via {tool}")
        r = run_tool(job, tool, args, fallback_cost=unit)
        out = proj / s["asset"]
        if r["returncode"] != 0 or not out.exists():
            raise StepFailed(f"{tool} for scene {s['id']} exited {r['returncode']}: "
                             f"{(r['stderr'] or r['stdout']).strip()[-600:]}")
        made.append(s["asset"])
    return {"assets": made}


def _chunk_words(words: list[dict], max_words: int, max_chars: int) -> list[dict]:
    """Same chunking as the template's gen_captions.py (1-3 words per karaoke pill)."""
    chunks, cur = [], []
    for w in words:
        cand = " ".join([c["w"] for c in cur] + [w["w"]])
        if cur and (len(cur) >= max_words or len(cand) > max_chars):
            chunks.append({"text": " ".join(c["w"] for c in cur), "start": cur[0]["s"], "end": cur[-1]["e"]})
            cur = []
        cur.append(w)
    if cur:
        chunks.append({"text": " ".join(c["w"] for c in cur), "start": cur[0]["s"], "end": cur[-1]["e"]})
    for i in range(len(chunks) - 1):
        chunks[i]["end"] = max(chunks[i]["end"], chunks[i + 1]["start"])
    return chunks


def step_captions(job: dict, spec: dict, narration_s: float) -> dict:
    """tools/captions.py (Scribe when ELEVENLABS_API_KEY is set, else local whisper) writes
    <scene>.captions.json; convert each into the template's captions/words_<id>.json."""
    proj = _project(job)
    scene_dir = proj / "audio" / "scenes"
    engine = spec["captions"]["engine"]
    args = [str(scene_dir), "--json"]
    if engine != "auto":
        args += ["--engine", engine]
    paid = engine != "whisper" and bool(os.environ.get("ELEVENLABS_API_KEY"))
    unit = PRICES["captions_per_audio_second"]["scribe"] if paid else 0.0
    r = run_tool(job, "captions.py", args, fallback_cost=round(narration_s * unit, 4))
    if r["returncode"] != 0:
        log_line(job, f"captions skipped: {(r['stderr'] or r['stdout']).strip()[-300:]}")
        return {"skipped": True, "reason": (r["stderr"] or r["stdout"]).strip()[-300:]}
    config = json.loads((proj / "config.json").read_text(encoding="utf-8"))
    caps = config.get("captions", {})
    written = 0
    for s in spec["scenes"]:
        src = scene_dir / f"{s['id']}_{s['slug']}.captions.json"
        if not src.exists():
            continue
        words = [{"w": c["text"].strip(), "s": c["startMs"] / 1000.0, "e": c["endMs"] / 1000.0}
                 for c in json.loads(src.read_text(encoding="utf-8")) if c.get("text", "").strip()]
        chunks = _chunk_words(words, caps.get("maxWords", 3), caps.get("maxChars", 22))
        _write_json_atomic(proj / "captions" / f"words_{s['id']}.json", chunks)
        written += 1
    return {"scenes": written, "engine": "scribe" if paid else "whisper"}


def step_music(job: dict, spec: dict, total_s: float) -> dict:
    proj = _project(job)
    music = spec["music"]
    out = proj / "audio" / "music.mp3"
    args = ["--prompt", music.get("prompt") or DEFAULT_MUSIC_PROMPT,
            "--duration", str(int(min(240, max(30, total_s + 8)))), "--output", str(out),
            "--progress", "json", "--json", "--cloud", music.get("provider") or "acemusic"]
    unit = PRICES["music_track"].get(music.get("provider", "acemusic"), 0.03)
    r = run_tool(job, "music_gen.py", args, fallback_cost=unit)
    if r["returncode"] != 0 or not out.exists():
        log_line(job, "music skipped: " + (r["stderr"] or r["stdout"]).strip()[-300:])
        return {"skipped": True}
    return {"file": "audio/music.mp3"}


def step_build(job: dict) -> Path:
    proj = _project(job)
    # scene VO to -16 LUFS first (qc.py --project --fix); a failure here is not fatal
    run_tool(job, "qc.py", ["--project", str(proj), "--fix", "--json"], timeout=600)
    r = run_tool(job, str(proj / "build.py"), [], cwd=proj, timeout=1800)
    out = proj / "out" / "short.mp4"
    if r["returncode"] != 0 or not out.exists():
        raise StepFailed(f"build.py exited {r['returncode']}: {(r['stderr'] or r['stdout']).strip()[-800:]}")
    return out


def _qc(job: dict, render: Path, fix: bool) -> dict:
    proj = _project(job)
    args = ["--project", str(proj), "--render", str(render), "--json"] + (["--fix"] if fix else [])
    r = run_tool(job, "qc.py", args, timeout=900)
    try:
        return json.loads(r["stdout"])
    except ValueError:
        return {"passed": False, "errors": 1, "warnings": 0,
                "findings": [{"level": "error", "area": "qc", "msg": f"qc.py exited {r['returncode']}: "
                              f"{(r['stderr'] or r['stdout']).strip()[-400:]}"}]}


def step_qc(job: dict, render: Path) -> tuple[Path, dict]:
    """Run qc --fix (normalises loudness to -14 LUFS, writing short.loudnorm.mp4 when it
    has to), then grade the delivered file."""
    first = _qc(job, render, fix=True)
    fixed = first.get("fixed_render")
    final = Path(fixed) if fixed and Path(fixed).exists() else render
    report = _qc(job, final, fix=False) if final != render else first
    summary = {"passed": bool(report.get("passed")), "errors": report.get("errors", 0),
               "warnings": report.get("warnings", 0),
               "findings": [f for f in report.get("findings", []) if f.get("level") in ("error", "warn")][:30],
               "loudness_fixed": bool(fixed)}
    _write_json_atomic(job_dir(job["id"]) / "qc.json", report)
    return final, summary


def _probe_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
        return round(float(r.stdout.strip()), 2)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def step_publish(job: dict, spec: dict, final: Path) -> dict:
    name = f"{_slugify(spec['title'], 'video')}-{spec['format'].replace(':', 'x')}.mp4"
    dest = job_dir(job["id"]) / name
    shutil.copy2(final, dest)
    out = {"mp4": str(dest), "duration_s": _probe_duration(dest), "size_bytes": dest.stat().st_size,
           "url": None, "r2_key": None, "published": None}
    sheet = dest.with_name(dest.stem + "-sheet.png")
    dur = out["duration_s"] or 30
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dest),
                        "-vf", f"fps=1/{max(1.0, dur / 8):.3f},scale=480:-1,tile=4x2", str(sheet)],
                       capture_output=True, timeout=120, check=True)
        out["sheet"] = str(sheet)
    except (OSError, subprocess.SubprocessError) as e:
        log_line(job, f"contact sheet skipped: {e}")
    if job.get("publish_to"):
        target = Path(job["publish_to"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, target)
        out["published"] = str(target)
        log_line(job, f"copied to {target}")
        if out.get("sheet"):
            ts = target.with_name(target.stem + "-sheet.png")
            shutil.copy2(sheet, ts)
            out["published_sheet"] = str(ts)
    try:
        from file_transfer import get_r2_client, upload_to_r2
        client, _ = get_r2_client()
        if client:
            url, key = upload_to_r2(str(dest), "video-jobs")
            out["url"], out["r2_key"] = url, key
            log_line(job, f"uploaded to R2: {key}")
    except Exception as e:  # noqa: BLE001 — upload is best-effort
        log_line(job, f"R2 upload skipped: {e}")
    return out


def run_pipeline(job: dict, spec: dict, steps: dict | None = None) -> dict:
    """Drive the state machine. `steps` lets tests inject fakes for the paid steps."""
    st = {"voiceover": step_voiceover, "visuals": step_visuals, "captions": step_captions,
          "music": step_music, "build": step_build, "qc": step_qc, "publish": step_publish}
    if steps:
        st.update(steps)
    est = step_estimates(spec)
    try:
        if not job.get("project_dir"):
            job["project_dir"] = str(create_project(job, spec))
        check_budget(job, "voiceover", est.get("voiceover", 0.0))
        set_phase(job, "voiceover", 10)
        vo = st["voiceover"](job, spec)
        job["steps"]["voiceover"] = vo
        check_budget(job, "visuals", est.get("visuals", 0.0))
        set_phase(job, "visuals", 30)
        job["steps"]["visuals"] = st["visuals"](job, spec)
        if spec["captions"]["enabled"]:
            check_budget(job, "captions", est.get("captions", 0.0))
            set_phase(job, "captions", 75)
            job["steps"]["captions"] = st["captions"](job, spec, vo.get("narration_s", 0.0))
        if spec["music"]["enabled"]:
            check_budget(job, "music", est.get("music", 0.0))
            set_phase(job, "music", 80)
            job["steps"]["music"] = st["music"](job, spec, vo.get("narration_s", 0.0) + len(spec["scenes"]) * 1.2)
        set_phase(job, "build", 82)
        render = st["build"](job)
        set_phase(job, "qc", 93)
        final, qc = st["qc"](job, render)
        job["steps"]["qc"] = qc
        set_phase(job, "publish", 98)
        job["output"] = {**st["publish"](job, spec, final), "qc": qc}
        set_phase(job, "done", 100, finished=_now())
    except BudgetExceeded as e:
        set_phase(job, "failed", error=f"budget_exceeded: {e}", error_code="budget_exceeded", finished=_now())
    except (StepFailed, SpecError, FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError) as e:
        set_phase(job, "failed", error=f"{type(e).__name__}: {e}", error_code="step_failed", finished=_now())
    return job


# ---------------------------------------------------------------------------
# submit / worker / status / result / list / cancel
# ---------------------------------------------------------------------------

def submit(spec: dict, max_usd: float, spec_source: str = "spec", brief: str | None = None,
           start_worker: bool = True, publish_to: str | None = None) -> dict:
    """Create the job on disk, take the lock, and launch the detached worker. Returns the
    job dict. Raises BudgetExceeded (before anything is written) or LockHeld."""
    spec = normalize_spec(spec)
    empty = [s["id"] for s in spec["scenes"] if not s["text"]]
    if empty:
        raise SpecError(f"scenes {empty} have no narration text; a skeleton can be estimated but not submitted")
    est = estimate(spec)
    if est["estimate_usd"] > max_usd + 1e-9:
        raise BudgetExceeded(f"estimate ${est['estimate_usd']:.2f} is over --max-usd ${max_usd:.2f}; "
                             "raise the cap, drop scenes, or use cheaper visuals (flux, TURBO)")
    job_id = new_job_id()
    acquire_lock(job_id)
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(d / "spec.json", spec)
    job = {
        "id": job_id, "created": _now(), "updated": _now(), "phase": "queued", "progress": 0,
        "title": spec["title"], "format": spec["format"], "scenes": len(spec["scenes"]),
        "brief": brief, "spec_source": spec_source, "spec_path": str(d / "spec.json"),
        "project_dir": None, "estimate_usd": est["estimate_usd"], "actual_usd": 0.0, "max_usd": max_usd,
        "steps": {}, "output": None, "error": None, "worker_pid": None, "log": str(d / "job.log"),
        "publish_to": str(Path(publish_to).resolve()) if publish_to else None,
        "ledger": "reserve" if reserve_at_submit() else "actual",
    }
    save_job(job)
    log_line(job, f"submitted: {spec['title']!r} {spec['format']} {len(spec['scenes'])} scenes "
                  f"est ${est['estimate_usd']:.2f} cap ${max_usd:.2f}")
    if start_worker:
        job["worker_pid"] = spawn_worker(job_id)
        save_job(job)
    return job


def spawn_worker(job_id: str) -> int:
    """Detach the worker so it outlives the caller on every platform."""
    log = open(job_dir(job_id) / "worker.out", "ab")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT,
                    "cwd": str(ROOT), "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", job_id], env=env, **kwargs)
    log.close()
    return p.pid


def worker_main(job_id: str) -> int:
    job = load_job(job_id)
    job["worker_pid"] = os.getpid()
    update_lock_pid(job_id, os.getpid())
    save_job(job)
    spec = json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))
    try:
        run_pipeline(job, spec)
    except Exception as e:  # noqa: BLE001 — anything else is still a failed job, not a lost one
        set_phase(job, "failed", error=f"{type(e).__name__}: {e}", error_code="worker_crash", finished=_now())
    finally:
        release_lock(job_id)
    return 0 if job.get("phase") == "done" else 1


def refresh_status(job: dict) -> dict:
    """Detect a worker that died without writing a terminal phase."""
    if job.get("phase") not in TERMINAL and job.get("worker_pid") and not pid_alive(job["worker_pid"]):
        set_phase(job, "failed", error="worker exited without finishing (see job.log)",
                  error_code="worker_died", finished=_now())
        release_lock(job["id"])
    return job


# Typical seconds per step, measured 2026-10-31 on Modal (LTX warm; a cold start adds ~60 s).
_STEP_SECONDS = {"voiceover": 70, "ltx": 170, "ideogram": 12, "flux": 20, "captions": 8,
                 "music": 60, "build": 60, "qc": 10, "publish": 8}


def eta_seconds(job: dict) -> int | None:
    """Rough time left, from the phase and the assets still to make. None when finished."""
    if job.get("phase") in TERMINAL:
        return None
    try:
        spec = json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return None
    kinds = [s["visual"]["kind"] for s in spec.get("scenes", []) if s["visual"]["kind"] != "none"]
    visuals = sum(_STEP_SECONDS.get(k, 20) for k in kinds)
    order = ["voiceover", "visuals", "captions", "music", "build", "qc", "publish"]
    costs = {"voiceover": _STEP_SECONDS["voiceover"], "visuals": visuals,
             "captions": _STEP_SECONDS["captions"] if spec.get("captions", {}).get("enabled", True) else 0,
             "music": _STEP_SECONDS["music"] if spec.get("music", {}).get("enabled") else 0,
             "build": _STEP_SECONDS["build"], "qc": _STEP_SECONDS["qc"], "publish": _STEP_SECONDS["publish"]}
    phase = job.get("phase", "queued")
    start = order.index(phase) if phase in order else 0
    remaining = sum(costs[p] for p in order[start:])
    if phase == "visuals" and job.get("step_detail"):
        done = [k for k in kinds][:max(0, int((job.get("progress", 30) - 30) / 45 * len(kinds)))]
        remaining -= sum(_STEP_SECONDS.get(k, 20) for k in done)
    return max(15, int(remaining))


def status(job_id: str, log_lines: int = 15) -> dict:
    job = refresh_status(load_job(job_id))
    job["eta_s"] = eta_seconds(job)
    tail: list[str] = []
    try:
        tail = (job_dir(job_id) / "job.log").read_text(encoding="utf-8").splitlines()[-log_lines:]
    except OSError:
        pass
    return {**job, "done": job["phase"] in TERMINAL, "log_tail": tail}


def result(job_id: str) -> dict:
    job = refresh_status(load_job(job_id))
    out = job.get("output") or {}
    if job["phase"] == "done" and out.get("r2_key"):
        try:
            from file_transfer import get_r2_client
            client, cfg = get_r2_client()
            if client:
                out["url"] = client.generate_presigned_url(
                    "get_object", Params={"Bucket": cfg["bucket_name"], "Key": out["r2_key"]}, ExpiresIn=7200)
        except Exception:  # noqa: BLE001
            pass
    return {"id": job["id"], "phase": job["phase"], "progress": job.get("progress"), "title": job.get("title"),
            "format": job.get("format"), "mp4": out.get("mp4"), "published": out.get("published"), "url": out.get("url"),
            "duration_s": out.get("duration_s"), "qc": out.get("qc"), "estimate_usd": job.get("estimate_usd"),
            "actual_usd": job.get("actual_usd"), "error": job.get("error"), "error_code": job.get("error_code"),
            "ready": job["phase"] == "done"}


def list_jobs() -> list[dict]:
    rows = []
    if not jobs_dir().exists():
        return rows
    for p in sorted(jobs_dir().glob("*/job.json"), reverse=True):
        try:
            j = refresh_status(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
        rows.append({k: j.get(k) for k in ("id", "phase", "progress", "title", "format", "estimate_usd",
                                            "actual_usd", "created", "error_code")})
    return rows


def cancel(job_id: str) -> dict:
    job = load_job(job_id)
    if job["phase"] in TERMINAL:
        return job
    pid = job.get("worker_pid")
    if pid and pid_alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
        except OSError as e:
            log_line(job, f"cancel: could not kill {pid}: {e}")
    set_phase(job, "cancelled", error="cancelled by caller", error_code="cancelled", finished=_now())
    release_lock(job_id)
    return job


def presets() -> dict:
    return {
        "prices": PRICES,
        "formats": {k: {"width": v["width"], "height": v["height"]} for k, v in FORMATS.items()},
        "visual_kinds": {
            "ltx": "5 s motion b-roll from a prompt (LTX-2.3 on Modal); describe motion and atmosphere, no text",
            "ideogram": "designed still with legible text (Ideogram 4); fill ideogram_caption_template into "
                        "visual.caption (a plain prompt works but adds filler text to the card)",
            "flux": "text-free generated background with a slow drift (FLUX.2 Klein on Modal, about a cent); "
                    "what a scene with no visual gets",
            "text": "constructed visual: generated background + the scene's headline and an accent bar animated in "
                    "(headline, optional subline, optional background hint); about a cent",
            "none": "gradient placeholder card with the scene slug",
        },
        "voice_defaults": {"provider": "qwen3", "speaker": "Ryan", "maxWpm": 165,
                           "providers": ["qwen3", "elevenlabs", "60db"],
                           "presets": sorted(p.stem for p in (ROOT / "assets" / "voices").glob("*.json"))},
        "defaults": {"max_usd": DEFAULT_MAX_USD, "duration_s": 45, "format": "9:16",
                     "words_per_second": WORDS_PER_SECOND},
        "spec_example": SPEC_EXAMPLE,
        "ideogram_caption_template": IDEOGRAM_CAPTION_TEMPLATE,
        "expander_env": "VIDEO_JOB_EXPANDER",
        "jobs_dir": str(jobs_dir()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(data, as_json: bool, human: str | None = None) -> None:
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        print(human if human is not None else json.dumps(data, indent=2))


def _human_estimate(est: dict, source: str) -> str:
    lines = [f"Estimate: ${est['estimate_usd']:.2f}  ({est['scenes']} scenes, ~{est['video_s_approx']}s, "
             f"{est['format']}, spec from {source})"]
    for r in est["breakdown"]:
        lines.append(f"  {r['step']:10s} {r['count']:>6} x {r['unit']:28s} ${r['usd']:.4f}")
    if source == "skeleton":
        lines.append("  (skeleton: no expander configured, so this is the shape a brief of that length "
                     "usually takes; pass --spec for a real number)")
    return "\n".join(lines)


def _human_status(s: dict) -> str:
    head = (f"{s['id']}  {s['phase']:10s} {s.get('progress', 0):3d}%  {s.get('title')!r} {s.get('format')}  "
            f"est ${s.get('estimate_usd', 0):.2f} spent ${s.get('actual_usd', 0):.2f}")
    if s.get("step_detail") and s["phase"] not in TERMINAL:
        head += f"\n  now: {s['step_detail']}"
    if s.get("eta_s"):
        head += f"\n  about {max(1, round(s['eta_s'] / 60))} min left"
    if s.get("error"):
        head += f"\n  error: {s['error']}"
    if s.get("output", {}) and s["output"].get("mp4"):
        head += f"\n  mp4: {s['output']['mp4']}"
    return head


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description="Async video jobs: brief or spec in, QC-checked MP4 out.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_spec_args(p, with_max=False):
        p.add_argument("--brief", help="One-paragraph description of the video")
        p.add_argument("--spec", help="Finished spec JSON file ('-' for stdin)")
        p.add_argument("--spec-json", help="The spec as a JSON string, or @path/to/spec.json (relative to the "
                                           "agent workspace when LONGARM_WORKSPACE is set)")
        p.add_argument("--scene", action="append", default=[], metavar="TEXT[ :: KIND: PROMPT]",
                       help="Flat form, repeatable, one per scene: narration, then optionally ' :: ltx: motion "
                            "prompt', ' :: card: HEADLINE ; object' (or 'HEADLINE ; A -> B -> C' for a "
                            "flow diagram), ' :: flux: prompt'. No JSON needed")
        p.add_argument("--title", help="Video title (flat form; default: the first scene's text)")
        p.add_argument("--voice", help="Voice preset: assets/voices/<name>.json (a clone) or a brand name")
        p.add_argument("--music", help="Music bed prompt (default: a subtle ambient bed; it's free on acemusic)")
        p.add_argument("--no-music", action="store_true", help="No music bed")
        p.add_argument("--draft", action="store_true", help="Scenes without a visual get a plain placeholder instead of a generated background")
        p.add_argument("--allow-screens", action="store_true",
                       help="Let an ltx: prompt ask for a screen, interface or dashboard (refused by default: the model draws fake software)")
        p.add_argument("--no-title-card", action="store_true", help="Skip the animated title over the first scene")
        p.add_argument("--format", choices=list(FORMATS), default=None, help="Output format (default 9:16)")
        p.add_argument("--duration", type=int, default=45, help="Target seconds, for --brief (default 45)")
        if with_max:
            p.add_argument("--max-usd", type=float, default=float(os.environ.get("VIDEO_JOB_MAX_USD", DEFAULT_MAX_USD)),
                           help=f"Refuse if the estimate is above this (default {DEFAULT_MAX_USD}, env VIDEO_JOB_MAX_USD)")
            p.add_argument("--output", "-o", help="Also copy the finished MP4 here (e.g. an agent's public files/ dir)")
        p.add_argument("--json", action="store_true", help="Machine-readable output")

    add_spec_args(sub.add_parser("estimate", help="Cost estimate; needs no API keys"))
    add_spec_args(sub.add_parser("submit", help="Start a job and return at once"), with_max=True)
    add_spec_args(sub.add_parser("spec", help="Expand a brief into a spec (needs VIDEO_JOB_EXPANDER)"))
    for name in ("status", "result", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
        p.add_argument("--json", action="store_true")
    sub.add_parser("list").add_argument("--json", action="store_true")
    st = sub.add_parser("settle", help="Write the ledger correction for finished reserve-mode jobs that never got one")
    st.add_argument("job_id", nargs="?")
    st.add_argument("--all", action="store_true", help="Every finished job without a settlement")
    st.add_argument("--json", action="store_true")
    sub.add_parser("presets").add_argument("--json", action="store_true")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("job_id")
    for p in sub.choices.values():  # longarm's wrapper injects this on every call
        p.add_argument("--progress", choices=["human", "json"], default="human",
                       help="json: emit a cost event on stderr (the estimate on submit, $0 otherwise)")
    args = ap.parse_args(argv)
    progress_json = getattr(args, "progress", "human") == "json"

    try:
        if args.cmd in ("estimate", "submit", "spec"):
            fmt = args.format or "9:16"
            if args.scene:
                music = False if args.no_music else (args.music or None)
                spec, source = normalize_spec(spec_from_scene_args(
                    args.title, args.scene, fmt, voice=args.voice, music=music, draft=args.draft,
                    title_card=not args.no_title_card, allow_screens=args.allow_screens)), "scenes"
            elif args.spec_json:
                spec, source = normalize_spec(parse_spec_text(args.spec_json)), "spec"
            else:
                spec, source = resolve_spec(args.brief, args.spec, fmt, args.duration,
                                            allow_skeleton=(args.cmd == "estimate"))
            if args.format and args.spec:
                spec["format"] = args.format
                spec = normalize_spec(spec)
            if args.voice and not args.scene:
                spec["voice"] = load_voice_preset(args.voice)
                spec = normalize_spec(spec)
            if args.cmd == "spec":
                _print(spec, True)
                return 0
            est = estimate(spec)
            if args.cmd == "estimate":
                if progress_json:
                    emit_cost_event(0.0, "estimate only, nothing generated")
                _print({**est, "spec_source": source}, args.json, _human_estimate(est, source))
                return 0
            job = submit(spec, args.max_usd, spec_source=source, brief=args.brief, publish_to=args.output)
            if progress_json:
                emit_cost_event(job["estimate_usd"] if job["ledger"] == "reserve" else 0.0,
                                f"{'reserved' if job['ledger'] == 'reserve' else 'estimate'} for job {job['id']}")
            _print({"id": job["id"], "phase": job["phase"], "estimate_usd": job["estimate_usd"],
                    "max_usd": job["max_usd"], "title": job["title"], "format": job["format"],
                    "scenes": job["scenes"], "spec_path": job["spec_path"], "log": job["log"],
                    "poll": f"uv run tools/video_job.py status {job['id']} --json"},
                   args.json, f"submitted {job['id']}  est ${job['estimate_usd']:.2f}  "
                              f"(poll: uv run tools/video_job.py status {job['id']})")
            return 0
        if progress_json and args.cmd in ("status", "result", "list", "cancel", "presets"):
            emit_cost_event(0.0, f"{args.cmd}: no generation")
        if args.cmd == "status":
            s = status(args.job_id)
            _print(s, args.json, _human_status(s))
            return 0
        if args.cmd == "result":
            r = result(args.job_id)
            _print(r, args.json, (f"{r['id']}  {r['phase']}  {r['mp4']}  {r.get('duration_s')}s  "
                                  f"spent ${r.get('actual_usd') or 0:.2f}" + (f"\n  url: {r['url']}" if r.get("url") else "")
                                  + (f"\n  error: {r['error']}" if r.get("error") else "")))
            return 0 if r["ready"] else 2
        if args.cmd == "list":
            rows = list_jobs()
            _print(rows, args.json, "\n".join(
                f"{r['id']}  {r['phase']:10s} {r.get('progress') or 0:3d}%  ${r.get('actual_usd') or 0:.2f}  {r.get('title')!r}"
                for r in rows) or "no jobs")
            return 0
        if args.cmd == "cancel":
            j = cancel(args.job_id)
            _print({"id": j["id"], "phase": j["phase"]}, args.json, f"{j['id']} {j['phase']}")
            return 0
        if args.cmd == "presets":
            _print(presets(), True)
            return 0
        if args.cmd == "settle":
            if not args.job_id and not args.all:
                _print({"error": "settle needs a job id or --all", "error_code": "usage"}, args.json,
                       "settle needs a job id or --all")
                return 2
            ids = [args.job_id] if args.job_id else [j["id"] for j in list_jobs() if j.get("phase") in TERMINAL]
            rows = []
            for jid in ids:
                j = load_job(jid)
                delta = settle_reservation(j)
                rows.append({"id": jid, "phase": j.get("phase"), "reserved_usd": j.get("estimate_usd"),
                             "actual_usd": j.get("actual_usd"), "ledger_delta": delta})
            _print(rows, args.json, "\n".join(
                f"{r['id']}  {r['phase']:10s} reserved ${r['reserved_usd'] or 0:.2f} actual ${r['actual_usd'] or 0:.2f} "
                f"{'ledger ' + format(r['ledger_delta'], '+.2f') if r['ledger_delta'] is not None else 'already settled'}"
                for r in rows) or "nothing to settle")
            return 0
        if args.cmd == "_worker":
            return worker_main(args.job_id)
    except BudgetExceeded as e:
        _print({"error": str(e), "error_code": "budget_exceeded"}, args.json, f"refused: {e}")
        return 2
    except NoExpander as e:
        _print({"error": str(e), "error_code": "no_expander"}, args.json, f"refused: {e}")
        return 2
    except LockHeld as e:
        _print({"error": str(e), "error_code": "lock_held", "holder": e.holder}, args.json, f"busy: {e}")
        return 3
    except (SpecError, FileNotFoundError, ValueError) as e:
        _print({"error": str(e), "error_code": "bad_request"}, args.json, f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
