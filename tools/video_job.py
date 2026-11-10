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
    uv run tools/video_job.py resume   JOB_ID [--max-usd 3.00] [--json]   # carry on a failed job
    uv run tools/video_job.py prune    [--keep 10] [--older-than 7] [--dry-run] [--json]
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
MEDIA_SPEND_LOG (default logs/media-spend.log) as JSON lines (ts, tool, cost_usd,
elapsed_s, exit, args), one per paid call. One job runs at a time (jobs/.lock).

Exit codes: 0 ok, 1 error, 2 refused or not ready (over budget, no expander, job not
done), 3 another job holds the lock. Under --progress json every call, failed ones
included, prints exactly one `cost` event on stderr.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import functools
import json
import os
import platform
import re
import shlex
import shutil
import statistics
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
PRICES_VERSION = "2026-11-01"

# ---------------------------------------------------------------------------
# Unit prices (USD). The one table `estimate` reads. Sources: Ideogram list price
# (ideogram.ai/pricing, API tab); LTX-2.3 measured on Modal A100-80GB at $2.50/h
# (docs/ltx2.md benchmarks, 576x1024 or 1024x576 at 121 frames lands at the top of the
# range); Qwen3-TTS and FLUX.2 on Modal A10G at $1.10/h; ElevenLabs TTS at the Creator
# tier per-character rate; ElevenLabs Scribe at $0.40 per audio hour.
# 2026-11-01: checked against the receipts of the last ten jobs on a production agent host. LTX clips cost
# a median $0.0825 (26 clips, $0.077-0.169 with cold starts), so 0.10, not 0.20; FLUX
# backgrounds $0.0058 (11), so 0.0075, not 0.01. Ideogram DEFAULT ($0.06) and Scribe
# (~$0.004 a job) matched the table.
# ---------------------------------------------------------------------------
PRICES = {
    "version": PRICES_VERSION,
    "ideogram_image": {"TURBO": 0.03, "DEFAULT": 0.06, "QUALITY": 0.10},
    "ltx_clip": 0.10,
    "flux_image": 0.0075,
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

VISUAL_KINDS = ("ideogram", "ltx", "flux", "text", "file", "none")
# "file" is the person's own photo or clip (from a kit's assets or the agent workspace), Ken Burns on a still,
# boomerang on a clip, with an optional headline. Costs nothing and needs no GPU.
BACKGROUNDS = ("auto", "flux", "local")
# Where `text` scenes and scenes with no visual get their background: "flux" generates one (about a cent, needs
# FLUX on Modal or RunPod), "local" draws a flat brand-colour field with grain here (free, no GPU), "auto" is flux
# when this host has it, else local. A scene that names kind "flux" always means FLUX.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CLIP_EXTS = (".mp4", ".mov", ".webm")
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
DEFAULT_MAX_MB = 9.5            # Discord's bot upload limit is 10 MB; jobs encode under this
DEFAULT_LTX_FRAMES = 121        # ~5 s at 24 fps; the build boomerang-loops it
# LTX-2 puts stray text, logos and anime styling in ~30% of clips unless told not to.
DEFAULT_LTX_NEGATIVE = ("text, letters, words, captions, subtitles, logo, watermark, anime, cartoon, "
                        "illustration, worst quality, inconsistent motion, blurry, jittery, distorted")
# Both image models put a stranger in the shot when nobody was asked for: a "slow push-in on a
# server room" came back as a woman praying at the rack, an "abstract cinematic background" as a
# portrait. A person in a video is a casting decision, so the clip prompt has to name one.
NO_PEOPLE_NEGATIVE = "people, person, man, woman, face, portrait, hands, crowd, figure, silhouette"
# A terse clip prompt ("filing cabinet drawers sliding open, papers flowing out") leaves the
# style to the model, which picked a cartoon twice in one job; the same job's prompts that
# carried "cinematic lighting" and "dust motes" came back live-action. So every ltx prompt
# gets a footage suffix unless it already names a style, or the spec sets ltxStyle ("" to
# disable, or a different suffix).
LTX_STYLE_SUFFIX = ("live-action footage, photorealistic, cinematic lighting, shallow depth of field, "
                    "natural colour, 35mm film look")
STYLE_WORDS = re.compile(r"\b(cinematic|photoreal\w*|live.action|footage|35mm|16mm|film look|filmic|documentary|"
                         r"realistic|anamorphic|bokeh|handheld|steadicam|drone shot|time.lapse|timelapse|macro|"
                         r"stop.motion|claymation|watercolou?r|hand.drawn|pixel art|noir)\b", re.I)


def styled_ltx_prompt(prompt: str, spec_style: str | None) -> str:
    """The prompt as sent to LTX: untouched when it names a style or ltxStyle is "", else with
    the footage suffix (or the spec's own)."""
    prompt = (prompt or "").strip().rstrip(",.;")
    suffix = LTX_STYLE_SUFFIX if spec_style is None else str(spec_style).strip()
    if not suffix or STYLE_WORDS.search(prompt):
        return prompt
    return f"{prompt}, {suffix}"
# LTX cannot draw software. Ask it for a terminal, a dashboard, a timeline or "code lines" and
# it paints a fake interface covered in garbled text (three of three such clips on the first
# live day). Submit refuses these before any spend unless the spec says allowScreens; the
# message tells the agent what to do instead, which prose in its instructions did not achieve.
# Conceptual nouns pull the video model away from footage: "scale animation" came back as a
# cartoon cat, "neural net sculpture" as a sculptor's hands. A clip prompt has to name a thing
# a camera could film; the idea it stands for belongs in the narration and the headline.
CONCEPT_WORDS = re.compile(r"\b(animation|animated|animations|sculpture|sculptures|abstract|visuali[sz]ation|"
                           r"visuali[sz]ed|metaphor|metaphorical|symbolic|symbol|symbols|concept|conceptual|"
                           r"infographic|diagram|diagrams|3d|cgi|motion graphics|graphic|graphics|illustration|"
                           r"illustrated|cartoon|anime|constellation|particles|network of|nodes?|data flowing|"
                           r"glowing lines|energy|holographic|hologram)\b", re.I)
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

def max_mb(spec: dict | None = None) -> float | None:
    """The size cap for the delivered MP4: spec.maxMB, else VIDEO_JOB_MAX_MB, else 9.5.
    0 (or a negative number) turns the cap off."""
    v = (spec or {}).get("maxMB")
    if v is None:
        v = os.environ.get("VIDEO_JOB_MAX_MB", DEFAULT_MAX_MB)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = DEFAULT_MAX_MB
    return v if v > 0 else None


def jobs_dir() -> Path:
    return Path(os.environ.get("VIDEO_JOBS_DIR") or ROOT / "jobs")


def agent_workspace() -> Path | None:
    """VIDEO_JOB_WORKSPACE: the calling agent's workspace, when its runtime sets one. Relative spec
    paths resolve there first, the spend ledger defaults to its logs/, and submit reserves the whole
    estimate. LONGARM_WORKSPACE is the older name, read until existing hosts set the new one."""
    v = os.environ.get("VIDEO_JOB_WORKSPACE") or os.environ.get("LONGARM_WORKSPACE")
    return Path(v) if v else None


def spend_log_path() -> Path:
    """MEDIA_SPEND_LOG, else the agent workspace's logs/, else the toolkit's own logs/."""
    if os.environ.get("MEDIA_SPEND_LOG"):
        return Path(os.environ["MEDIA_SPEND_LOG"])
    ws = agent_workspace()
    if ws:
        return ws / "logs" / "media-spend.log"
    return ROOT / "logs" / "media-spend.log"


def reserve_at_submit() -> bool:
    """When true, `submit` books the whole estimate in the shared ledger as one receipt (a
    reservation the daily cap can count at once) and the worker keeps its per-tool receipts
    in the job directory only. On by default when an agent workspace is set (a runtime whose daily
    cap sums the ledger before every call); off elsewhere so the ledger holds actuals."""
    v = os.environ.get("VIDEO_JOB_RESERVE_AT_SUBMIT")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return agent_workspace() is not None


_cost_emitted = False


def emit_cost_event(usd: float, msg: str) -> None:
    """The `cost` progress line a runtime's spend wrapper turns into a receipt. Only printed under
    --progress json, so plain CLI use stays quiet. main() makes sure every call under --progress
    json prints exactly one, errors included: a wrapper that books a flat fee for a receipt with no
    cost would otherwise charge the agent for every failed call."""
    global _cost_emitted
    _cost_emitted = True
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
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows refuses to replace a file another process has open (a status poll
            # reading job.json as the worker writes it); the reader lets go in milliseconds
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def pid_alive(pid: int | None) -> bool:
    """Cross-platform liveness check. Never signals the process."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        # Unverified on a real Windows box (none to hand); written against the Win32 docs.
        # Explicit restype/argtypes: the default int restype truncates a 64-bit HANDLE.
        # A process that exits with code 259 (STILL_ACTIVE) reads as alive; nothing in the
        # pipeline exits with that code.
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            # a live process we may not query (another user's) still exists
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = wintypes.DWORD()
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
    return f"Abstract cinematic background evoking: {background_gist(hint)}. {BACKGROUND_STYLE}"


def background_gist(text: str, max_words: int = 12) -> str:
    """The narration as a theme, not a sentence to letter. FLUX printed 'listen & you'll hear
    it coming' across a background whose prompt quoted a narration with a quotation, an
    ampersand and slashes in it (2026-11-01), despite 'no text' in the style. Quoted phrases
    go, symbols become spaces, and it is lower-cased and capped."""
    t = re.sub(r"[\"\u201c\u201d][^\"\u201c\u201d]*[\"\u201c\u201d]", " ", text or "")
    t = re.sub(r"[^A-Za-z0-9' ]+", " ", t.replace("\u2019", "'"))
    words = [w.strip("'") for w in t.lower().split()]
    return " ".join([w for w in words if w][:max_words])


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
    known = sorted(p.stem for p in (ROOT / "assets" / "voices").glob("*.json"))
    raise SpecError(f"unknown voice preset {name!r}: no assets/voices/{name}.json and no brands/{name}/. "
                    f"Presets here: {', '.join(known) or 'none'}; or leave --voice off for the default narrator")


def kit_dirs() -> list[Path]:
    """Where kits live: the agent workspace's kits/ first (a runtime keeps each person's kit there), then
    the toolkit's own kits/."""
    ws = agent_workspace()
    return ([ws / "kits"] if ws else []) + [ROOT / "kits"]


def list_kits() -> list[dict]:
    seen, out = set(), []
    for d in kit_dirs():
        for kj in sorted(d.glob("*/kit.json")):
            name = kj.parent.name
            if name in seen:
                continue
            seen.add(name)
            try:
                k = json.loads(kj.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            adir = kj.parent / (k.get("assets") or "assets")
            described = k.get("describe") or {}
            recent = kit_recent_files(name)
            files = sorted(p.name for p in adir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS + CLIP_EXTS) if adir.is_dir() else []
            out.append({"name": name, "description": k.get("description"), "path": str(kj.parent),
                        "format": k.get("format"),
                        # what each file shows, so a model that can't see them can choose, and how many of the
                        # last few videos made with this kit used it, so the next one can pick others
                        "assets": [{"file": f, "shows": described.get(f), "usedRecently": recent.get(f, 0)}
                                   for f in files]})
    return out


RECENT_VIDEOS = 3


def kit_recent_files(kit_name: str, videos: int = RECENT_VIDEOS) -> dict[str, int]:
    """file -> how many of the last `videos` finished jobs made with this kit showed it."""
    counts: dict[str, int] = {}
    seen = 0
    for j in list_jobs()[:40]:
        if seen >= videos:
            break
        if j.get("phase") != "done":
            continue
        try:
            old = json.loads((job_dir(j["id"]) / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        k = old.get("kit")
        names = {k.get("name"), Path(k.get("dir") or "").name} if isinstance(k, dict) else {k}
        if kit_name not in names:
            continue
        seen += 1
        for f in {Path(str((sc.get("visual") or {}).get("path"))).name for sc in old.get("scenes", [])
                  if isinstance(sc.get("visual"), dict) and sc["visual"].get("kind") == "file"}:
            counts[f] = counts.get(f, 0) + 1
    return counts


def load_kit(ref: str) -> dict:
    """A kit is set up once in a working session: kits/<name>/kit.json naming a brand, a voice, a
    format, how backgrounds are made, and an assets folder of the person's own photos and clips.
    `ref` is a kit name (searched in kit_dirs()) or a path to the kit folder."""
    cands = [Path(ref)] if ("/" in ref or "\\" in ref) else [d / ref for d in kit_dirs()]
    for c in cands:
        if (c / "kit.json").exists():
            try:
                k = json.loads((c / "kit.json").read_text(encoding="utf-8"))
            except ValueError as e:
                raise SpecError(f"kit {ref!r}: kit.json is not valid JSON ({e})")
            k["name"] = k.get("name") or c.name
            k["dir"] = str(c.resolve())
            k["assets"] = str((c / (k.get("assets") or "assets")).resolve())
            return k
    known = [k["name"] for k in list_kits()]
    raise SpecError(f"unknown kit {ref!r}. Kits here: {', '.join(known) or 'none'}. A kit is a folder with a "
                    "kit.json under kits/ (or the agent workspace's kits/); see kits/README.md")


def brand_colors(spec: dict) -> dict:
    """Colours from the kit's own brand.json, else brands/<brand>/brand.json, else {}."""
    kit = spec.get("kit") or {}
    for p in ([Path(kit["dir"]) / "brand.json"] if kit.get("dir") else []) + (
            [ROOT / "brands" / spec["brand"] / "brand.json"] if spec.get("brand") else []):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("colors") or {}
            except ValueError:
                return {}
    return {}


# What a model writes when it isn't paying attention: dashes and stock AI phrasing. Narration is heard and
# burned as captions, and the title, headlines and sublines are on screen, so these are refused before any
# spend, with the reason.
from copy_lint import COPY_RULES  # noqa: E402  (shared with every public text: tools/copy_lint.py)


def copy_findings(spec: dict) -> list[str]:
    """Copy lint on every word shown or spoken, plus the repeat guard: refuse lines lifted from an earlier
    finished job on this host."""
    out = []
    shown = [("the title", spec.get("title") or "")]
    for s in spec["scenes"]:
        ov = s.get("overlay") or {}
        shown += [(f"scene {s['id']}", s["text"]), (f"scene {s['id']} headline", ov.get("headline") or ""),
                  (f"scene {s['id']} subline", ov.get("subline") or "")]
    for where, text in shown:
        for rx, why in COPY_RULES:
            m = rx.search(text)
            if m:
                out.append(f"{where}: '{m.group(0).strip()}' is {why}")
    seen = {}
    for j in list_jobs()[:40]:
        if j.get("phase") != "done":
            continue
        try:
            old = json.loads((job_dir(j["id"]) / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for sc in old.get("scenes", []):
            k = _spoken_key(sc.get("text") or "")
            if len(k.split()) >= 6:
                seen.setdefault(k, j["id"])
    for s in spec["scenes"]:
        k = _spoken_key(s["text"])
        if k in seen:
            out.append(f"scene {s['id']} repeats a line from job {seen[k]} word for word; every video gets its own "
                       "words, written from what this person asked for (to re-cut that video in another format "
                       "or voice, submit again with \"allowCopy\": true)")
    return out


def _flux_available() -> bool:
    if _skip_host_check():
        return True      # not consulting the host: assume it has everything
    return _has(_NEEDS[("visual", "flux", "modal")]) or _has(_NEEDS[("visual", "flux", "runpod")])


def _hex_rgb(h: str, fallback=(10, 10, 20)) -> tuple:
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def field_clear_of(color: str, ink: str) -> str:
    """The local field's colour, moved one step off the caption pill's ink when they match: the pill is drawn in
    the palette ink, which is also the field's default colour, so on its own field it disappeared."""
    c, k = _hex_rgb(color), _hex_rgb(ink)
    if sum(abs(a - b) for a, b in zip(c, k)) > 36:
        return color
    light = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) > 140
    mixed = [round(v + ((0 if light else 255) - v) * 0.14) for v in c]
    return "#" + "".join(f"{v:02X}" for v in mixed)


def draw_local_background(out: Path, width: int, height: int, color: str, seed: int = 0) -> None:
    """A flat colour field with fine grain: no GPU, no network, no gradient. The build's Ken Burns drift
    moves the grain, and the headline reveal and captions carry the motion."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = np.array(_hex_rgb(color), dtype=np.float32)
    grain = rng.normal(0.0, 5.0, (height, width, 1)).astype(np.float32)
    img = np.clip(base + grain, 0, 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    Image.fromarray(img).save(tmp)
    os.replace(tmp, out)


# Set by worker_main: the spec it reads was normalised and checked by submit, so the file paths it already
# resolved are kept (the worker runs from the toolkit root, not the submitter's directory).
_TRUSTED_SPEC = False


def _input_roots(kit: dict | None) -> list[Path]:
    """The folders a spec may take a person's file from: the kit, the agent workspace, the submitter's
    directory and the toolkit. Nothing outside them reaches a render, whatever path a spec names."""
    roots = [Path(kit["dir"])] if kit else []
    ws = agent_workspace()
    if ws:
        roots.append(ws)
    return [r.resolve() for r in roots + [Path.cwd(), ROOT]]


def _within(p: Path, roots: list[Path]) -> bool:
    return any(p == r or r in p.parents for r in roots)


def _resolve_file(scene_id: str, rel: str, kit: dict | None) -> Path:
    p = Path(rel)
    cands = [p] if p.is_absolute() else (([Path(kit["assets"]) / rel, Path(kit["dir"]) / rel] if kit else [])
                                         + _workspace_paths(rel))
    roots = _input_roots(kit)
    for c in cands:
        if c.is_file():
            if not _within(c.resolve(), roots):
                raise SpecError(f"scene {scene_id}: {rel} is outside the kit, the agent workspace and this "
                                "directory; copy it into the kit's assets")
            if c.suffix.lower() not in IMAGE_EXTS + CLIP_EXTS:
                raise SpecError(f"scene {scene_id}: {rel} is not an image or clip "
                                f"({', '.join(IMAGE_EXTS + CLIP_EXTS)})")
            return c.resolve()
    where = "the kit's assets, " if kit else ""
    raise SpecError(f"scene {scene_id}: no file {rel!r} in {where}the agent workspace or the toolkit")


def normalize_spec(spec: dict) -> dict:
    """Fill defaults, derive ids and asset paths, and validate. Returns a new dict."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    out = json.loads(json.dumps(spec))  # deep copy
    kit = out.get("kit")
    if isinstance(kit, str):
        kit = load_kit(kit)
    elif isinstance(kit, dict) and kit.get("dir") and not _TRUSTED_SPEC:
        kit = load_kit(str(kit["dir"]))      # an already-normalised kit is read again from its own kit.json
    if kit is not None:
        if not isinstance(kit, dict) or not kit.get("dir"):
            raise SpecError("spec.kit must be a kit name")
        out["kit"] = kit
        for key in ("brand", "format", "backgrounds"):
            if kit.get(key) and not out.get(key):
                out[key] = kit[key]
        if kit.get("voice") and not out.get("voice"):
            out["voice"] = kit["voice"]
        if "music" in kit and "music" not in out:
            out["music"] = kit["music"]
    bg = str(out.get("backgrounds") or "auto")
    if bg not in BACKGROUNDS:
        raise SpecError(f"spec.backgrounds must be one of {', '.join(BACKGROUNDS)}")
    out["backgrounds"] = bg
    bg_source = "local" if bg == "local" or (bg == "auto" and not _flux_available()) else "flux"
    colors = brand_colors(out)
    if colors and kit is not None and not out.get("palette"):
        # a kit's video is theirs: its brand sets the accent and the ground
        out["palette"] = {k: v for k, v in {"accent1": colors.get("primary"), "accent2": colors.get("accent"),
                                            "ink": colors.get("bgDark") or colors.get("navyDark")}.items() if v}
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
    if kit is not None and voice.get("refAudio") and not Path(voice["refAudio"]).is_absolute():
        # a kit's cloned voice: its reference clip lives in the kit
        in_kit = Path(kit["dir"]) / voice["refAudio"]
        if in_kit.exists():
            voice["refAudio"] = str(in_kit.resolve())
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
            # no visual named: a background (generated or local, see BACKGROUNDS) unless this is a draft
            vis = {"kind": "none"} if draft else {"kind": "flux", "prompt": auto_background_prompt(text), "auto": True}
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
        if kind == "file":
            if not vis.get("path"):
                raise SpecError(f"scene {s['id']}: visual.file needs a path (a photo or clip in the kit's assets "
                                "or the agent workspace)")
            if not (_TRUSTED_SPEC and vis.get("src") and Path(vis["src"]).is_file()):
                vis["src"] = str(_resolve_file(s["id"], str(vis["path"]), kit))
            # where the subject sits along the side the crop cuts (0-1): the visual's, the scene's, or the
            # kit's focus map (keyed by the path as written or its file name)
            fmap = (kit or {}).get("focus") or {}
            fx = vis.get("focus", s.get("focus", fmap.get(str(vis["path"]), fmap.get(Path(str(vis["path"])).name))))
            s.pop("focus", None)
            if fx is not None:
                try:
                    s["focus"] = min(1.0, max(0.0, float(fx)))
                except (TypeError, ValueError):
                    raise SpecError(f"scene {s['id']}: focus must be a number from 0 to 1")
        else:
            s.pop("focus", None)             # only a photo or clip of the person's own is cropped by focus
            vis.pop("src", None)
        if kind == "text" or (kind == "flux" and vis.get("auto")):
            if vis.get("source") not in (None, "local", "flux"):
                raise SpecError(f"scene {s['id']}: visual.source must be local or flux")
            vis["source"] = vis.get("source") or bg_source
            if vis["source"] == "local":
                vis["color"] = vis.get("color") or (out.get("palette") or {}).get("ink") or \
                    colors.get("bgDark") or colors.get("navyDark") or ""
        else:
            vis.pop("source", None)          # a local field stands in only for a text scene's background
        if kind not in ("none", "file") and not (vis.get("prompt") or vis.get("caption")):
            raise SpecError(f"scene {s['id']}: visual.{kind} needs a prompt (or a caption for ideogram)")
        check_narration_is_spoken(s["id"], text, vis)
        if kind == "ltx" and len((vis.get("prompt") or "").split()) < 3 and not str(vis.get("prompt", "")).startswith("("):
            raise SpecError(f"scene {s['id']}: the {kind} prompt is '{vis.get('prompt')}', which is not a shot. "
                            "Describe the thing, what moves and the light in a short sentence, e.g. "
                            "'slow push-in on a brass kettle, steam rising, warm kitchen light'.")
        if kind == "ltx" and not out.get("allowScreens"):
            hit = CONCEPT_WORDS.search(vis.get("prompt") or "")
            if hit:
                raise SpecError(
                    f"scene {s['id']}: the clip prompt asks for an idea ('{hit.group(0)}'), not a shot, and the video "
                    "model turns ideas into cartoons and clip art. Name a real thing a camera could film and how it "
                    "moves (a brass kettle, a filing cabinet, rain on a window ledge, a lamp switching on); the idea "
                    "lives in the narration and the headline. To insist, set allowScreens (--allow-screens).")
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
        elif kind == "file":
            ext = Path(vis["src"]).suffix.lower()
            s["asset"] = (f"clips/{s['id']}_{slug}{ext}" if ext in CLIP_EXTS else f"images/{s['id']}_{slug}{ext}")
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
    n_local = sum(1 for s in scenes if s["visual"].get("source") == "local")
    n_flux = kinds["flux"] + kinds["text"] - n_local
    if n_flux:
        rows.append({"step": "visuals", "unit": "flux background", "count": n_flux,
                     "unit_usd": PRICES["flux_image"], "usd": n_flux * PRICES["flux_image"]})
    if n_local or kinds["file"]:
        rows.append({"step": "visuals", "unit": "local background or own file (no GPU)",
                     "count": n_local + kinds["file"], "unit_usd": 0.0, "usd": 0.0})

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


EXPANDER_INSTRUCTIONS = (
    "Write a spec JSON for a short explainer video. Keep total narration near words_budget; each scene's "
    "'text' is the sentence or two the narrator says, in plain spoken English (never a headline in capitals). "
    "Give every scene one visual, mixing three kinds: 'text' (a generated background with an animated "
    "headline: {\"kind\": \"text\", \"headline\": \"2 TO 5 WORDS\", \"subline\": \"optional\"}), 'ltx' "
    "(5 s of motion footage: {\"kind\": \"ltx\", \"prompt\": \"...\"}), and 'ideogram' (a designed card with "
    "legible text: {\"kind\": \"ideogram\", \"caption\": {\"headline\": \"...\", \"object\": \"...\"}}; an "
    "object written 'A -> B -> C' becomes a labelled flow diagram). An ltx prompt must name a real physical thing "
    "a camera could film, how it moves and the light (e.g. 'slow push-in on a brass kettle, steam rising, warm "
    "kitchen light'). It must not ask for any screen, window, interface, dashboard, terminal, code, chart, app, "
    "or person, nor an idea (animation, abstract, diagram, network, particles, hologram, energy): those are "
    "refused. Put ideas in a 'text' scene's headline instead. Scene 1 must hook in one sentence. Output only "
    "the JSON object.")


KIT_INSTRUCTIONS = (
    " This video is for the person or product in `kit`: write in their voice and keep to its description. "
    "Prefer their own footage: a scene can be {\"kind\": \"file\", \"path\": \"<a file from kit.assets>\", "
    "\"headline\": \"optional\"}, choosing the file whose `shows` fits the sentence; use each file once, and "
    "prefer files with usedRecently 0 (the last few videos showed the others). "
    "Mix in 'text' scenes for the key line. Don't use 'ltx' or 'ideogram' unless the kit has no fitting file.")


def expand_brief(brief: str, fmt: str, duration_s: int, timeout: int = 300, attempts: int = 2,
                 kit: str | None = None) -> dict:
    """Run the VIDEO_JOB_EXPANDER command. It reads a JSON request on stdin and prints a
    spec JSON on stdout (code fences tolerated). Any provider: the toolkit doesn't care.
    A spec the validator refuses goes back to the expander once with the error attached
    (`previous_error`, `previous_spec`), the same way an agent would fix its own."""
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
        "instructions": EXPANDER_INSTRUCTIONS,
    }
    if kit:
        kdir = load_kit(kit)["dir"]
        k = next((x for x in list_kits() if x["name"] == kit or str(Path(x["path"]).resolve()) == kdir),
                 None) or {"name": kit}
        request["kit"] = {key: k.get(key) for key in ("name", "description", "format", "assets")}
        request["instructions"] = EXPANDER_INSTRUCTIONS + KIT_INSTRUCTIONS
    args = split_command(cmd)
    for attempt in range(max(1, attempts)):
        r = subprocess.run(args, input=json.dumps(request), capture_output=True, text=True,
                           timeout=timeout, cwd=str(ROOT), encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"expander exited {r.returncode}: {r.stderr.strip()[-800:]}")
        text = r.stdout.strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RuntimeError(f"expander printed no JSON object: {text[:300]!r}")
        spec = parse_spec_text(m.group(0))
        spec.setdefault("format", fmt)
        spec.setdefault("brief", brief)
        if kit:
            spec["kit"] = kit
        try:
            n = normalize_spec(spec)
            problems = copy_findings(n)
            if problems:
                raise SpecError("the narration needs another pass: " + "; ".join(problems))
            return n
        except SpecError as e:
            if attempt + 1 >= max(1, attempts):
                raise SpecError(f"the expander's spec was refused: {e}")
            request["previous_spec"] = spec
            request["previous_error"] = (f"Your previous spec was refused: {e} Return the corrected spec, "
                                         "changing only what the error names.")
    raise RuntimeError("unreachable")


def split_command(cmd: str) -> list[str]:
    """A command line as argv. POSIX rules everywhere except Windows, where backslashes in
    paths must survive; shlex's non-POSIX mode keeps the quotes on a quoted token
    ('"C:\\Python\\python.exe"'), so they are stripped here."""
    if os.name != "nt":
        return shlex.split(cmd)
    out = []
    for tok in shlex.split(cmd, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


def _workspace_paths(rel: str) -> list[Path]:
    """Where a relative spec path may live: the agent workspace first, then cwd, then here."""
    out = []
    ws = agent_workspace()
    if ws:
        out.append(ws / rel)
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
            elif kind in ("file", "photo", "image"):
                path, _, head = rest.partition(";")
                headline, _, sub = head.partition(";")
                if not path.strip():
                    raise SpecError(f"--scene {i + 1}: file needs a path, e.g. 'file: photos/candle.jpg ; NEW RANGE'")
                visual = {"kind": "file", "path": path.strip()}
                if headline.strip():
                    visual["headline"] = headline.strip().strip('"')
                    visual["subline"] = sub.strip()
            else:
                raise SpecError(f"--scene {i + 1}: unknown visual '{kind}' (use ltx:, card:, text:, flux:, file: or none)")
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
                 allow_skeleton: bool, kit: str | None = None) -> tuple[dict, str]:
    """Returns (spec, source) where source is 'spec', 'expander' or 'skeleton'. `kit` names a kit the
    spec uses unless it names its own."""
    if spec_path:
        sp = read_spec_file(spec_path)
        if kit and not sp.get("kit"):
            sp["kit"] = kit
        return normalize_spec(sp), "spec"
    if not brief:
        raise SpecError("pass --brief or --spec")
    try:
        return expand_brief(brief, fmt, duration_s, kit=kit), "expander"
    except NoExpander:
        if allow_skeleton:
            sk = skeleton_spec(brief, fmt, duration_s)
            if kit:
                sk["kit"] = kit
            return normalize_spec(sk), "skeleton"
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


class JobUnreadable(SpecError):
    pass


def load_job(job_id: str) -> dict:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        raise FileNotFoundError(f"no such job: {job_id}")
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise JobUnreadable(f"job {job_id}'s job.json is unreadable ({e}); its log is {path.parent / 'job.log'}")
    if not isinstance(job, dict) or job.get("id") != job_id:
        raise JobUnreadable(f"job {job_id}'s job.json is not a job record")
    return job


def save_job(job: dict) -> None:
    job["updated"] = _now()
    _write_json_atomic(job_dir(job["id"]) / "job.json", job)


def log_line(job: dict, msg: str) -> None:
    with open(job_dir(job["id"]) / "job.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


MAX_EVENTS = 200


def add_event(job: dict, kind: str, msg: str, save: bool = True, **fields) -> None:
    """Append a timestamped event to job["events"] (the last MAX_EVENTS are kept), so status
    can say what is happening and what just finished, not only which phase it is in."""
    ev = {"ts": _now(), "kind": kind, "msg": msg}
    ev.update({k: v for k, v in fields.items() if v is not None})
    events = job.setdefault("events", [])
    events.append(ev)
    del events[:-MAX_EVENTS]
    if save:
        save_job(job)


def set_phase(job: dict, phase: str, progress: int | None = None, **fields) -> None:
    if job.get("phase") != phase:
        job["phase_started_ts"] = time.time()
        add_event(job, "phase", phase, save=False, error=fields.get("error"))
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
    # what the shared ledger holds for this job right now: the estimate at submit, the
    # actual after a settle, plus whatever a resume reserved on top
    reserved = float(job["ledger_usd"] if job.get("ledger_usd") is not None else job.get("estimate_usd") or 0.0)
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
    job["ledger_usd"] = round(actual, 4)
    save_job(job)
    log_line(job, f"settled: reserved ${reserved:.2f} actual ${actual:.2f} ledger {delta:+.2f}")
    return delta


def write_receipt(job: dict, tool: str, cost_usd: float | None, elapsed_s: float | None,
                  exit_code: int, args: list[str]) -> None:
    """One JSON line per paid call (ts, tool, cost_usd, elapsed_s, exit, args), plus job_id."""
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


LOCK_GRACE_S = 120


def _lock_is_stale(holder: dict, now: float | None = None) -> bool:
    """Stale when the job it names is finished, or its worker is known and dead, or no
    worker ever reported in and the lock is old. A lock younger than LOCK_GRACE_S is never
    stale while its job is still being set up: submit takes the lock before job.json
    exists, and resume takes it while job.json still says failed."""
    now = now if now is not None else time.time()
    young = now - float(holder.get("ts", 0)) < LOCK_GRACE_S
    try:
        job = load_job(holder.get("job_id", ""))
    except FileNotFoundError:
        return not young
    except SpecError:
        return True
    if job.get("phase") in TERMINAL:
        # phase_started_ts of a terminal phase is when the job ended; a lock taken after
        # that is a resume being set up
        ended = float(job.get("phase_started_ts") or 0.0)
        resuming = young and ended > 0 and float(holder.get("ts", 0)) > ended
        return not resuming
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
    add_event(job, "tool", f"{Path(tool).name} started", tool=Path(tool).name)
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
    add_event(job, "tool", f"{Path(tool).name} {'finished' if rc == 0 else f'failed (exit {rc})'} in {elapsed:.0f}s",
              tool=Path(tool).name, seconds=elapsed, cost_usd=cost, exit=rc)
    return {"returncode": rc, "stdout": out, "stderr": err, "cost": cost, "elapsed": elapsed}


def _error_tail(r: dict, limit: int = 600) -> str:
    """The end of a failed tool's output without the JSON progress lines (heartbeats, cost)
    that --progress json interleaves on stderr, so the error is the error."""
    lines = [ln for ln in (r.get("stderr") or r.get("stdout") or "").splitlines()
             if not ln.lstrip().startswith('{"ts"')]
    return "\n".join(lines).strip()[-limit:]


class BudgetExceeded(RuntimeError):
    pass


def needs_text(m: dict) -> str:
    """What a missing entry asks for: a program to install, or every env var it needs."""
    return m.get("install") or ("set " + " and ".join(m["env"]) if m.get("env") else "a provider this toolkit knows")


class Unavailable(RuntimeError):
    """The spec needs a provider this host has no key or endpoint for. Raised by submit before
    anything is written or spent; `missing` lists what and how to fix it."""
    def __init__(self, missing: list[dict]):
        self.missing = missing
        super().__init__("this host can't make this spec: " + "; ".join(
            f"{m['needed_for']} needs {needs_text(m)}"
            + (f" ({m['hint']})" if m.get("hint") else "")
            for m in missing))


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
    if not spec["music"]["enabled"]:
        # no bed asked for: drop the file key, or qc.py warns "audio/music.mp3 not found" and
        # the agent passes on a problem that isn't one
        config["music"]["file"] = None
    config["timing"] = {**config.get("timing", {}), "startPad": 0.0}   # frame 0 is the cover
    # veryfast: a quarter of medium's CPU on small hosts at the same look for this content;
    # maxMB keeps the file attachable (Discord's bot limit is 10 MB)
    config["encode"] = {"preset": "veryfast", "maxMB": max_mb(spec)}
    _write_json_atomic(dest / "config.json", config)

    scenes = []
    for s in spec["scenes"]:
        row = {"id": s["id"], "slug": s["slug"], "visual": s["visual"]["kind"],
               "asset": s["asset"], "text": s["text"]}
        if s.get("overlay"):
            row["overlay"] = s["overlay"]
        if s.get("focus") is not None:
            row["focus"] = s["focus"]
        scenes.append(row)
    _write_json_atomic(dest / "scenes.json", {"title": spec["title"], "titleCard": spec.get("titleCard"),
                                              "scenes": scenes})
    return dest


# What each step leaves on disk. A step whose outputs exist is skipped, which is what makes
# `resume` safe: a job killed after its visuals reruns only the free steps after them.

def _scene_mp3(proj: Path, s: dict) -> Path:
    return proj / "audio" / "scenes" / f"{s['id']}_{s['slug']}.mp3"


def _nonempty(p: Path) -> bool:
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def voiceover_done(proj: Path, spec: dict) -> dict | None:
    """The saved per-scene durations when every scene's MP3 exists, else None."""
    try:
        durations = json.loads((proj / "vo_durations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if all(s["id"] in durations and _nonempty(_scene_mp3(proj, s)) for s in spec["scenes"]):
        return durations
    return None


def _made(job: dict | None, rel: str) -> bool:
    """A file counts as made only once its tool succeeded (job["made"]), so a partial file
    left by a killed or failing tool is never reused. Jobs from before the list existed
    trust the file."""
    if job is None or "made" not in job:
        return True
    return rel in job["made"]


def _mark_made(job: dict, rel: str) -> None:
    made = job.setdefault("made", [])
    if rel not in made:
        made.append(rel)


def asset_done(proj: Path, s: dict, job: dict | None = None) -> bool:
    return bool(s.get("asset")) and _nonempty(proj / s["asset"]) and _made(job, s["asset"])


def captions_done(proj: Path, spec: dict) -> bool:
    return all(_nonempty(proj / "captions" / f"words_{s['id']}.json") for s in spec["scenes"])


def music_done(proj: Path, job: dict | None = None) -> bool:
    return _nonempty(proj / "audio" / "music.mp3") and _made(job, "audio/music.mp3")


def remaining_estimates(job: dict, spec: dict) -> dict:
    """step_estimates, less what is already on disk: what the job still has to buy. On a
    fresh job this equals step_estimates; on a resumed one the cap is checked against only
    the outputs still missing."""
    est = step_estimates(spec)
    proj = Path(job["project_dir"]) if job.get("project_dir") else None
    if proj is None or not proj.exists():
        return est
    if voiceover_done(proj, spec) is not None:
        est["voiceover"] = 0.0
    missing = [s for s in spec["scenes"] if s["visual"]["kind"] != "none" and not asset_done(proj, s, job)]
    est["visuals"] = round(sum(_visual_args(spec, s, proj)[2] for s in missing), 4)
    if captions_done(proj, spec):
        est["captions"] = 0.0
    if music_done(proj, job):
        est["music"] = 0.0
    return est


def step_voiceover(job: dict, spec: dict) -> dict:
    proj = _project(job)
    voice = spec["voice"]
    scene_dir = proj / "audio" / "scenes"
    have = voiceover_done(proj, spec)
    if have is not None:
        prev = job.get("steps", {}).get("voiceover") or {}
        add_event(job, "voiceover", f"reusing {len(have)} voiced scenes from the earlier run")
        return {**prev, "durations": have, "narration_s": round(sum(have.values()), 2), "reused": True}
    for s in spec["scenes"]:
        (scene_dir / f"{s['id']}_{s['slug']}.txt").write_text(s["text"], encoding="utf-8")
    args = ["--provider", voice["provider"], "--scene-dir", str(scene_dir), "--json"]
    if voice.get("maxWpm"):
        args += ["--max-wpm", str(voice["maxWpm"])]
    if voice.get("minTempo"):
        args += ["--min-tempo", str(voice["minTempo"])]
    if voice.get("brand"):
        args += ["--brand", voice["brand"]]
    if voice["provider"] == "qwen3":
        args += ["--cloud", voice.get("cloud", "modal"), "--progress", "json"]   # relays the Modal cost
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
        raise StepFailed(f"voiceover.py exited {r['returncode']}: {_error_tail(r)}")
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
        wpm = item.get("wpm")
        add_event(job, "voiced", f"scene {name}: {item['duration_seconds']:.1f}s"
                  + (f" at {wpm:.0f} wpm" if isinstance(wpm, (int, float)) else "")
                  + (f" ({item['pacing']})" if item.get("pacing") not in (None, "ok") else ""),
                  save=False, scene=sid, seconds=item["duration_seconds"], wpm=wpm, pacing=item.get("pacing"))
    missing = [s["id"] for s in spec["scenes"] if s["id"] not in durations]
    if missing:
        raise StepFailed(f"no audio for scenes {missing}")
    _write_json_atomic(proj / "vo_durations.json", durations)
    return {"durations": durations, "pacing": pacing, "narration_s": round(sum(durations.values()), 2)}


def _visual_args(spec: dict, s: dict, proj: Path) -> tuple[str, list[str], float]:
    """(tool, args, fallback cost) for one scene's visual. A file or local background is made here,
    by step_visuals, with no tool and no cost."""
    fmt = FORMATS[spec["format"]]
    vis = s["visual"]
    out = proj / s["asset"]
    kind = vis["kind"]
    if kind == "file" or vis.get("source") == "local":
        return "", [], 0.0
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
        args = ["--prompt", styled_ltx_prompt(vis["prompt"], spec.get("ltxStyle")), "--width", str(w), "--height", str(h),
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
    reused = 0
    for i, s in enumerate(todo):
        if asset_done(proj, s, job):
            reused += 1
            made.append(s["asset"])
            continue
        vis = s["visual"]
        if vis["kind"] == "file" or vis.get("source") == "local":
            out = proj / s["asset"]
            out.parent.mkdir(parents=True, exist_ok=True)
            if vis["kind"] == "file":
                tmp = out.with_name(out.stem + ".tmp" + out.suffix)
                shutil.copyfile(vis["src"], tmp)
                os.replace(tmp, out)
                what = f"your file {Path(vis['src']).name}"
            else:
                w, h = FORMATS[spec["format"]]["flux"]
                ink = (spec.get("palette") or {}).get("ink") or "#0A0A14"   # the template's default ink
                field = field_clear_of(vis.get("color") or ink, ink) if spec["captions"]["enabled"] \
                    else (vis.get("color") or ink)
                draw_local_background(out, w, h, field, seed=int(s["id"]))
                what = "a local background"
            _mark_made(job, s["asset"])
            add_event(job, "asset", f"scene {s['id']}: {what}", scene=s["id"], visual=vis["kind"], cost_usd=0.0,
                      file=s["asset"])
            made.append(s["asset"])
            continue
        tool, args, unit = _visual_args(spec, s, proj)
        check_budget(job, f"visual {s['id']} ({tool})", unit)
        set_phase(job, "visuals", 30 + int(45 * i / max(n, 1)), step_detail=f"{s['id']}_{s['slug']} via {tool}")
        out = proj / s["asset"]
        out.unlink(missing_ok=True)          # never build on a partial file from an earlier try
        r = run_tool(job, tool, args, fallback_cost=unit)
        if r["returncode"] != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise StepFailed(f"{tool} for scene {s['id']} exited {r['returncode']}: "
                             f"{_error_tail(r)}")
        _mark_made(job, s["asset"])
        kind = s["visual"]["kind"]
        job.setdefault("timings", {}).setdefault("assets", []).append(
            {"kind": kind, "s": r["elapsed"], "ended": time.time()})
        add_event(job, "asset", f"scene {s['id']} {kind} ready in {r['elapsed']:.0f}s", scene=s["id"],
                  visual=kind, seconds=r["elapsed"], cost_usd=r["cost"], file=s["asset"])
        made.append(s["asset"])
    if reused:
        add_event(job, "visuals", f"reused {reused} of {n} visuals from the earlier run")
    return {"assets": made, "reused": reused or None}


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
    if captions_done(proj, spec):
        prev = job.get("steps", {}).get("captions") or {}
        return {**prev, "reused": True}
    engine = spec["captions"]["engine"]
    args = [str(scene_dir), "--json"]
    if engine != "auto":
        args += ["--engine", engine]
    paid = engine != "whisper" and bool(os.environ.get("ELEVENLABS_API_KEY"))
    unit = PRICES["captions_per_audio_second"]["scribe"] if paid else 0.0
    r = run_tool(job, "captions.py", args, fallback_cost=round(narration_s * unit, 4))
    if r["returncode"] != 0:
        reason = _error_tail(r, 300)
        log_line(job, f"captions skipped: {reason}")
        add_event(job, "captions", "captions skipped: the video renders without them", reason=reason[-200:])
        return {"skipped": True, "reason": reason}
    config = json.loads((proj / "config.json").read_text(encoding="utf-8"))
    caps = config.get("captions", {})
    written = 0
    for s in spec["scenes"]:
        src = scene_dir / f"{s['id']}_{s['slug']}.captions.json"
        if not src.exists():
            continue
        # a word with no letters or digits ("/", "&", "-") reads as a stray mark in a pill
        words = [{"w": c["text"].strip(), "s": c["startMs"] / 1000.0, "e": c["endMs"] / 1000.0}
                 for c in json.loads(src.read_text(encoding="utf-8"))
                 if any(ch.isalnum() for ch in c.get("text", ""))]
        chunks = _chunk_words(words, caps.get("maxWords", 3), caps.get("maxChars", 22))
        _write_json_atomic(proj / "captions" / f"words_{s['id']}.json", chunks)
        written += 1
    eng = "scribe" if paid else "whisper"
    add_event(job, "captions", f"captions for {written} of {len(spec['scenes'])} scenes ({eng})", engine=eng)
    if not written:
        return {"skipped": True, "reason": "captions.py wrote no caption files", "engine": eng}
    return {"scenes": written, "engine": eng}


def step_music(job: dict, spec: dict, total_s: float) -> dict:
    proj = _project(job)
    music = spec["music"]
    out = proj / "audio" / "music.mp3"
    if music_done(proj, job):
        return {"file": "audio/music.mp3", "reused": True}
    out.unlink(missing_ok=True)
    args = ["--prompt", music.get("prompt") or DEFAULT_MUSIC_PROMPT,
            "--duration", str(int(min(240, max(30, total_s + 8)))), "--output", str(out),
            "--progress", "json", "--json", "--cloud", music.get("provider") or "acemusic"]
    unit = PRICES["music_track"].get(music.get("provider", "acemusic"), 0.03)
    r = run_tool(job, "music_gen.py", args, fallback_cost=unit)
    if r["returncode"] != 0 or not out.exists():
        out.unlink(missing_ok=True)      # build.py mixes any music.mp3 it finds; a partial one must go
        reason = _error_tail(r, 300)
        log_line(job, "music skipped: " + reason)
        add_event(job, "music", "music skipped: the video renders without a bed", reason=reason[-200:])
        return {"skipped": True, "reason": reason}
    _mark_made(job, "audio/music.mp3")
    add_event(job, "music", f"music bed ready ({music.get('provider') or 'acemusic'})")
    return {"file": "audio/music.mp3"}


def step_build(job: dict) -> Path:
    proj = _project(job)
    # scene VO to -16 LUFS first (qc.py --project --fix); a failure here is not fatal
    run_tool(job, "qc.py", ["--project", str(proj), "--fix", "--json"], timeout=600)
    r = run_tool(job, str(proj / "build.py"), [], cwd=proj, timeout=1800)
    out = proj / "out" / "short.mp4"
    if r["returncode"] != 0 or not out.exists():
        raise StepFailed(f"build.py exited {r['returncode']}: {_error_tail(r, 800)}")
    return out


def _qc(job: dict, render: Path, fix: bool) -> dict:
    proj = _project(job)
    args = ["--project", str(proj), "--render", str(render), "--json"] + (["--fix"] if fix else [])
    r = run_tool(job, "qc.py", args, timeout=900)
    try:
        return json.loads(r["stdout"])
    except ValueError:
        return {"passed": False, "errors": 1, "warnings": 0,
                "findings": [{"level": "error", "check": "qc", "message": f"qc.py exited {r['returncode']}: "
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
    top = "; ".join((f.get("message") or f.get("msg") or "")[:120] for f in summary["findings"][:3])
    add_event(job, "qc", f"QC {'passed' if summary['passed'] else 'failed'}: {summary['errors']} errors, "
              f"{summary['warnings']} warnings" + (f" ({top})" if top else ""),
              passed=summary["passed"], errors=summary["errors"], warnings=summary["warnings"])
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
    cap = max_mb(spec)
    mb = out["size_bytes"] / 1e6
    out["over_size_cap"] = bool(cap and mb > cap)
    add_event(job, "render", f"{mb:.1f} MB, {out['duration_s']}s" + (f" (over the {cap:g} MB cap)" if out["over_size_cap"] else ""),
              size_bytes=out["size_bytes"], duration_s=out["duration_s"])
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
        add_event(job, "publish", f"copied to {target}", path=str(target))
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


def _record_step(job: dict, step: str, t0: float, result) -> None:
    """Per-step wall time for the host-calibrated ETA. A step that reused an earlier run's
    outputs is not timed: it says nothing about how long the step takes here."""
    if isinstance(result, dict) and result.get("reused") is True:
        return
    job.setdefault("timings", {}).setdefault("steps", {})[step] = round(time.time() - t0, 1)


def run_pipeline(job: dict, spec: dict, steps: dict | None = None) -> dict:
    """Drive the state machine. `steps` lets tests inject fakes for the paid steps. Every
    paid step skips outputs already on disk, so the same call resumes a broken job."""
    st = {"voiceover": step_voiceover, "visuals": step_visuals, "captions": step_captions,
          "music": step_music, "build": step_build, "qc": step_qc, "publish": step_publish}
    if steps:
        st.update(steps)
    job.setdefault("steps", {})
    job.setdefault("host", platform.node())
    try:
        if not job.get("project_dir"):
            job["project_dir"] = str(create_project(job, spec))
        est = remaining_estimates(job, spec)
        check_budget(job, "voiceover", est.get("voiceover", 0.0))
        set_phase(job, "voiceover", 10)
        t0 = time.time()
        vo = st["voiceover"](job, spec)
        job["steps"]["voiceover"] = vo
        _record_step(job, "voiceover", t0, vo)
        check_budget(job, "visuals", est.get("visuals", 0.0))
        set_phase(job, "visuals", 30)
        t0 = time.time()
        job["steps"]["visuals"] = st["visuals"](job, spec)
        _record_step(job, "visuals", t0, job["steps"]["visuals"])
        if spec["captions"]["enabled"]:
            check_budget(job, "captions", est.get("captions", 0.0))
            set_phase(job, "captions", 75)
            t0 = time.time()
            job["steps"]["captions"] = st["captions"](job, spec, vo.get("narration_s", 0.0))
            _record_step(job, "captions", t0, job["steps"]["captions"])
        if spec["music"]["enabled"]:
            check_budget(job, "music", est.get("music", 0.0))
            set_phase(job, "music", 80)
            t0 = time.time()
            job["steps"]["music"] = st["music"](job, spec, vo.get("narration_s", 0.0) + len(spec["scenes"]) * 1.2)
            _record_step(job, "music", t0, job["steps"]["music"])
        set_phase(job, "build", 82)
        t0 = time.time()
        render = st["build"](job)
        _record_step(job, "build", t0, None)
        set_phase(job, "qc", 93)
        t0 = time.time()
        final, qc = st["qc"](job, render)
        job["steps"]["qc"] = qc
        _record_step(job, "qc", t0, None)
        set_phase(job, "publish", 98)
        t0 = time.time()
        job["output"] = {**st["publish"](job, spec, final), "qc": qc}
        _record_step(job, "publish", t0, None)
        video_s = (job["output"] or {}).get("duration_s")
        if video_s:
            job["timings"]["video_s"] = video_s
        set_phase(job, "done", 100, finished=_now())
    except BudgetExceeded as e:
        set_phase(job, "failed", error=f"budget_exceeded: {e}", error_code="budget_exceeded", finished=_now())
    except (StepFailed, SpecError, FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError) as e:
        set_phase(job, "failed", error=f"{type(e).__name__}: {e}", error_code="step_failed", finished=_now())
    return job


# ---------------------------------------------------------------------------
# submit / worker / status / result / list / cancel
# ---------------------------------------------------------------------------

def resolve_output(path: str | None) -> str | None:
    """Where --output copies the finished MP4. It must name an .mp4 file. When
    VIDEO_JOB_OUTPUT_ROOTS is set (os.pathsep-separated directories), it must also sit inside
    one of them, so an agent can't be talked into writing anywhere on the host."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if p.suffix.lower() != ".mp4":
        raise SpecError(f"--output must name an .mp4 file (got {path!r})")
    roots = [Path(r).expanduser().resolve() for r in os.environ.get("VIDEO_JOB_OUTPUT_ROOTS", "").split(os.pathsep)
             if r.strip()]
    if roots and not any(p == r or r in p.parents for r in roots):
        raise SpecError(f"--output {path!r} is outside the allowed directories "
                        f"({', '.join(str(r) for r in roots)}); write it under one of those")
    return str(p)


def submit(spec: dict, max_usd: float, spec_source: str = "spec", brief: str | None = None,
           start_worker: bool = True, publish_to: str | None = None) -> dict:
    """Create the job on disk, take the lock, and launch the detached worker. Returns the
    job dict. Raises BudgetExceeded (before anything is written) or LockHeld."""
    spec = normalize_spec(spec)
    empty = [s["id"] for s in spec["scenes"] if not s["text"]]
    if empty:
        raise SpecError(f"scenes {empty} have no narration text; a skeleton can be estimated but not submitted")
    check_host(spec)
    if not spec.get("allowCopy"):
        problems = copy_findings(spec)
        if problems:
            raise SpecError("the narration needs another pass before anything is spent: " + "; ".join(problems)
                            + ". Rewrite those lines and submit again.")
    est = estimate(spec)
    if est["estimate_usd"] > max_usd + 1e-9:
        raise BudgetExceeded(f"estimate ${est['estimate_usd']:.2f} is over --max-usd ${max_usd:.2f}; "
                             "raise the cap, drop scenes, or use cheaper visuals (flux, TURBO)")
    publish_to = resolve_output(publish_to)
    try:   # before the lock, so a slow prune never holds it with no job.json behind it
        pruned = auto_prune()
    except Exception as e:  # noqa: BLE001 — housekeeping never blocks a submit
        pruned = {"error": str(e)}
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
        "publish_to": publish_to,
        "ledger": "reserve" if reserve_at_submit() else "actual",
        "ledger_usd": est["estimate_usd"] if reserve_at_submit() else 0.0,
        "made": [],
        "host": platform.node(), "phase_started_ts": time.time(), "events": [], "timings": {},
    }
    add_event(job, "submitted", f"{len(spec['scenes'])} scenes, {spec['format']}, est ${est['estimate_usd']:.2f}, "
              f"cap ${max_usd:.2f}", save=False)
    save_job(job)
    log_line(job, f"submitted: {spec['title']!r} {spec['format']} {len(spec['scenes'])} scenes "
                  f"est ${est['estimate_usd']:.2f} cap ${max_usd:.2f}")
    if pruned and (pruned.get("pruned") or pruned.get("error")):
        log_line(job, f"auto-prune: {pruned}")
    if start_worker:
        _start(job)
    return job


def _start(job: dict) -> None:
    """Spawn the worker and record its pid in the job and the lock before returning, so a
    second caller never sees a lock with nobody behind it."""
    pid = spawn_worker(job["id"])
    job["worker_pid"] = pid
    save_job(job)
    update_lock_pid(job["id"], pid)


def spawn_worker(job_id: str) -> int:
    """Detach the worker so it outlives the caller on every platform. POSIX: a new session,
    so the worker leads its own process group and `cancel` can stop its tools with it.
    Windows: DETACHED_PROCESS (no console) in a new process group."""
    log = open(job_dir(job_id) / "worker.out", "ab")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT,
                    "cwd": str(ROOT), "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", job_id], env=env, **kwargs)
    finally:
        log.close()
    return p.pid


def worker_main(job_id: str) -> int:
    global _TRUSTED_SPEC
    _TRUSTED_SPEC = True
    job = load_job(job_id)
    job["worker_pid"] = os.getpid()
    update_lock_pid(job_id, os.getpid())
    save_job(job)
    try:
        try:
            spec = json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise StepFailed(f"the job's spec.json is missing or unreadable ({e}); submit it again")
        run_pipeline(job, spec)
    except Exception as e:  # noqa: BLE001 — anything else is still a failed job, not a lost one
        set_phase(job, "failed", error=f"{type(e).__name__}: {e}", error_code="worker_crash", finished=_now())
    finally:
        release_lock(job_id)
    return 0 if job.get("phase") == "done" else 1


NEVER_STARTED_S = 120


def refresh_status(job: dict) -> dict:
    """Detect a worker that died without writing a terminal phase, or one that never
    started (the submit died between taking the lock and spawning it)."""
    if job.get("phase") in TERMINAL:
        return job
    if job.get("worker_pid") and not pid_alive(job["worker_pid"]):
        set_phase(job, "failed", error="worker exited without finishing (see job.log); resume it to carry on",
                  error_code="worker_died", finished=_now())
        release_lock(job["id"])
    elif (not job.get("worker_pid") and job.get("phase") == "queued"
          and time.time() - float(job.get("phase_started_ts") or (time.time() - _age_s(job.get("created"))))
          > NEVER_STARTED_S):
        set_phase(job, "failed", error="the worker never started; resume it or submit again",
                  error_code="worker_never_started", finished=_now())
        release_lock(job["id"])
    return job


def _age_s(stamp: str | None, now: float | None = None) -> float:
    """Seconds since an ISO-8601 UTC stamp written by _now(); 0 when unreadable."""
    try:
        t = _dt.datetime.strptime(stamp or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return 0.0
    return (now if now is not None else time.time()) - t


def clear_stale_lock() -> None:
    """Drop a lock whose holder is finished, dead, never started, or unreadable, so the
    next agent isn't told lock_held by a job nobody is running."""
    if not lock_path().exists():
        return
    holder = _read_lock()
    if holder is None or _lock_is_stale(holder):
        try:
            lock_path().unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Resume and prune
# ---------------------------------------------------------------------------

RESUMABLE_PHASES = ("failed", "cancelled")


def is_resumable(job: dict) -> bool:
    """A failed or cancelled job whose project is still on disk. Every paid step skips the
    outputs it finds, so a resume pays only for what is missing."""
    if job.get("phase") not in RESUMABLE_PHASES or job.get("pruned"):
        return False
    proj, sp = job.get("project_dir"), job.get("spec_path")
    if job.get("error_code") == "worker_never_started":
        return bool(sp) and Path(sp).exists()
    return bool(proj and sp) and Path(proj).exists() and Path(sp).exists()


def resume(job_id: str, max_usd: float | None = None, start_worker: bool = True) -> dict:
    """Restart a failed or cancelled job from its first missing output. Takes the lock like
    submit. Under the reserve ledger the job had settled to its actual spend when it
    stopped, so the rest of the estimate is booked again (job['resume_reserved_usd']) and
    the job settles once more when it ends."""
    job = refresh_status(load_job(job_id))
    if job.get("phase") not in RESUMABLE_PHASES:
        raise SpecError(f"job {job_id} is {job.get('phase')}; only a failed or cancelled job can be resumed")
    if not is_resumable(job):
        raise SpecError(f"job {job_id} can't be resumed: its project "
                        f"{'was pruned' if job.get('pruned') else 'or spec is gone'}; submit it again")
    try:
        check_host(json.loads(Path(job["spec_path"]).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        pass          # is_resumable checked the spec is there; a host check never blocks on its own bookkeeping
    if max_usd is not None:
        job["max_usd"] = max_usd
    acquire_lock(job_id)
    extra = 0.0
    if job.get("ledger") == "reserve" and "settled" in job:
        extra = round(max(0.0, float(job.get("estimate_usd") or 0) - float(job.get("actual_usd") or 0)), 4)
        job.pop("settled")
        job["ledger_usd"] = round(float(job.get("ledger_usd") or job.get("actual_usd") or 0) + extra, 4)
    job["resume_reserved_usd"] = extra
    job["resumes"] = int(job.get("resumes") or 0) + 1
    prior = job.get("error_code")
    job.update(error=None, error_code=None, finished=None, worker_pid=None, step_detail=None)
    set_phase(job, "queued", 0)
    add_event(job, "resumed", f"resume #{job['resumes']} after {prior}", prior=prior)
    log_line(job, f"resumed (after {prior}); cap ${job['max_usd']:.2f}, spent ${job.get('actual_usd', 0):.2f}")
    if start_worker:
        _start(job)
    return job


PRUNE_KEEP = 10
PRUNE_PUBLISHED_DAYS = 7


def _tree_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def prune(keep: int = PRUNE_KEEP, older_than_days: float | None = None, dry_run: bool = False,
          now: float | None = None) -> dict:
    """Delete the project/ tree (template copy, scene audio, clips, stills; 50+ MB a job) of
    old finished jobs. Keeps job.json, the receipts, the logs, the final MP4 and its sheet,
    so status and result still answer. Never touches a running job, the `keep` newest jobs,
    or a job that published its MP4 in the last 7 days. A pruned job can't be resumed."""
    now = now if now is not None else time.time()
    out = {"pruned": [], "freed_bytes": 0, "skipped": {}, "dry_run": dry_run}
    if not jobs_dir().exists():
        return out
    paths = sorted(jobs_dir().glob("*/job.json"), reverse=True)
    for i, p in enumerate(paths):
        jid = p.parent.name
        proj = p.parent / "project"
        if not proj.exists():
            continue
        if i < keep:
            out["skipped"][jid] = "recent"
            continue
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out["skipped"][jid] = "unreadable job.json"
            continue
        if job.get("phase") not in TERMINAL:
            out["skipped"][jid] = "running"
            continue
        age_days = _age_s(job.get("finished") or job.get("updated") or job.get("created"), now) / 86400
        if older_than_days is not None and age_days < older_than_days:
            out["skipped"][jid] = "too new"
            continue
        if (job.get("output") or {}).get("published") and age_days < PRUNE_PUBLISHED_DAYS:
            out["skipped"][jid] = "published in the last 7 days"
            continue
        size = _tree_bytes(proj)
        if not dry_run:
            shutil.rmtree(proj, ignore_errors=True)
            job["pruned"] = _now()
            _write_json_atomic(p, job)
        out["pruned"].append(jid)
        out["freed_bytes"] += size
    return out


def auto_prune() -> dict | None:
    """The conservative prune `submit` runs: VIDEO_JOB_KEEP (default 10) newest kept, 0 = off."""
    try:
        keep = int(os.environ.get("VIDEO_JOB_KEEP", PRUNE_KEEP))
    except ValueError:
        keep = PRUNE_KEEP
    if keep <= 0:
        return None
    return prune(keep=keep)


# ---------------------------------------------------------------------------
# ETA, calibrated per host
# ---------------------------------------------------------------------------

# Typical seconds per step, measured 2026-10-31 on Modal from a Mac (LTX warm). Used only
# until this host has finished jobs of its own; build is seconds per second of video.
_STEP_SECONDS = {"voiceover": 70, "ltx": 170, "ideogram": 12, "flux": 20, "text": 20, "file": 1, "captions": 8,
                 "music": 60, "build": 60, "qc": 10, "publish": 8}
_BUILD_S_PER_VIDEO_S = 1.2
LTX_COLD_START_S = 90            # Modal's LTX container scales to zero; a cold one adds this
LTX_WARM_WINDOW_S = 600          # ... when the last LTX call ended longer ago than this
ETA_HISTORY = 5
ORDER = ["voiceover", "visuals", "captions", "music", "build", "qc", "publish"]


def _recent_jobs(host: str, exclude: str | None = None, limit: int = 40):
    """job.json dicts from this host, newest first."""
    if not jobs_dir().exists():
        return
    n = 0
    for p in sorted(jobs_dir().glob("*/job.json"), reverse=True):
        if p.parent.name == exclude:
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if j.get("host") != host:
            continue
        yield j
        n += 1
        if n >= limit:
            return


def host_history(host: str, exclude: str | None = None, limit: int = ETA_HISTORY) -> list[dict]:
    """The timings of this host's last `limit` finished jobs."""
    out = []
    for j in _recent_jobs(host, exclude):
        if j.get("phase") == "done" and j.get("timings"):
            out.append(j["timings"])
            if len(out) >= limit:
                break
    return out


def step_model(history: list[dict]) -> dict:
    """Seconds per step and per visual kind (medians of the history, the table without
    one), and build seconds per second of video."""
    def med(xs, default):
        xs = [float(x) for x in xs if isinstance(x, (int, float))]
        return statistics.median(xs) if xs else default
    steps = {k: med([h.get("steps", {}).get(k) for h in history], _STEP_SECONDS[k])
             for k in ("voiceover", "captions", "music", "qc", "publish")}
    kinds = {k: med([a.get("s") for h in history for a in h.get("assets", []) if a.get("kind") == k],
                    _STEP_SECONDS[k]) for k in ("ltx", "ideogram", "flux", "text")}
    rates = [h["steps"]["build"] / h["video_s"] for h in history
             if h.get("video_s") and isinstance(h.get("steps", {}).get("build"), (int, float))]
    return {"steps": steps, "kinds": kinds, "build_rate": med(rates, _BUILD_S_PER_VIDEO_S)}


def eta_seconds(job: dict, now: float | None = None, history: list[dict] | None = None,
                last_ltx: float | None = -1.0) -> int | None:
    """Seconds left: the steps still to run, priced from this host's recent jobs (the
    table when it has none), the current step less the time already spent in it, and a
    cold-start allowance when LTX clips are pending and the last one ended over 10 min ago.
    None when finished. `history` and `last_ltx` are injectable for tests."""
    if job.get("phase") in TERMINAL:
        return None
    try:
        spec = json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = now if now is not None else time.time()
    host = job.get("host") or platform.node()
    recent = None
    if history is None or last_ltx == -1.0:
        recent = list(_recent_jobs(host))          # one pass over job.json files per poll
    if history is None:
        history = [j["timings"] for j in recent if j.get("id") != job.get("id")
                   and j.get("phase") == "done" and j.get("timings")][:ETA_HISTORY]
    m = step_model(history)
    proj = Path(job["project_dir"]) if job.get("project_dir") else None
    pending = [s for s in spec.get("scenes", []) if s["visual"]["kind"] != "none"
               and not (proj and asset_done(proj, s, job))]
    visuals = sum(m["kinds"].get(s["visual"]["kind"], _STEP_SECONDS["flux"]) for s in pending)
    if any(s["visual"]["kind"] == "ltx" for s in pending):
        if last_ltx == -1.0:
            ends = [a.get("ended") for j in (recent or []) for a in (j.get("timings") or {}).get("assets", [])
                    if a.get("kind") == "ltx" and a.get("ended")]
            last_ltx = max(ends) if ends else None
        if last_ltx is None or now - last_ltx > LTX_WARM_WINDOW_S:
            visuals += LTX_COLD_START_S
    try:
        video_s = estimate(spec)["video_s_approx"]
    except (SpecError, KeyError, ValueError):
        video_s = 45.0
    costs = {"voiceover": m["steps"]["voiceover"], "visuals": visuals,
             "captions": m["steps"]["captions"] if spec.get("captions", {}).get("enabled", True) else 0,
             "music": m["steps"]["music"] if spec.get("music", {}).get("enabled") else 0,
             "build": m["build_rate"] * video_s, "qc": m["steps"]["qc"], "publish": m["steps"]["publish"]}
    phase = job.get("phase", "queued")
    if phase not in ORDER:
        return max(15, int(sum(costs.values())))
    i = ORDER.index(phase)
    later = sum(costs[p] for p in ORDER[i + 1:])
    since = job.get("phase_started_ts") or now
    if phase == "visuals":
        ends = [a.get("ended") for a in (job.get("timings") or {}).get("assets", []) if a.get("ended")]
        since = max([since] + [e for e in ends if e and e >= since])
    current = max(0.0, costs[phase] - max(0.0, now - since))
    return max(15, int(later + current))


# ---------------------------------------------------------------------------
# status / result / list / cancel
# ---------------------------------------------------------------------------

def job_warnings(job: dict) -> list[str]:
    """Things the agent should tell its user even though the job succeeded."""
    out = []
    steps = job.get("steps") or {}
    cap = steps.get("captions") or {}
    if cap.get("skipped"):
        why = (cap.get("reason") or "").strip().splitlines()
        out.append("Captions were skipped, so the video has no burned captions"
                   + (f" ({why[-1][:160]})" if why else "")
                   + ". Set ELEVENLABS_API_KEY (Scribe) or run `uv sync --extra whisper` on the host to get them.")
    music = steps.get("music") or {}
    if music.get("skipped"):
        out.append("The music bed was skipped; the video has narration only.")
    pacing = (steps.get("voiceover") or {}).get("pacing") or {}
    fast = sorted(sid for sid, p in pacing.items() if (p or {}).get("pacing") == "fast")
    if fast:
        out.append(f"Scene{'s' if len(fast) > 1 else ''} {', '.join(fast)} came out fast (over ~170 wpm); "
                   "shorten the narration if it sounds rushed.")
    output = job.get("output") or {}
    if output.get("over_size_cap"):
        out.append(f"The MP4 is {output.get('size_bytes', 0) / 1e6:.1f} MB, over the size cap, so it may not "
                   "attach in chat; share the link instead.")
    return out


def status(job_id: str, log_lines: int = 15) -> dict:
    clear_stale_lock()
    job = refresh_status(load_job(job_id))
    job["eta_s"] = eta_seconds(job)
    tail: list[str] = []
    try:
        tail = (job_dir(job_id) / "job.log").read_text(encoding="utf-8").splitlines()[-log_lines:]
    except OSError:
        pass
    return {**job, "done": job["phase"] in TERMINAL, "resumable": is_resumable(job),
            "warnings": job_warnings(job), "log_tail": tail}


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
            "duration_s": out.get("duration_s"), "size_bytes": out.get("size_bytes"), "qc": out.get("qc"),
            "estimate_usd": job.get("estimate_usd"), "actual_usd": job.get("actual_usd"), "error": job.get("error"),
            "error_code": job.get("error_code"), "warnings": job_warnings(job), "resumable": is_resumable(job),
            "ready": job["phase"] == "done"}


def list_jobs() -> list[dict]:
    rows = []
    if not jobs_dir().exists():
        return rows
    clear_stale_lock()
    for p in sorted(jobs_dir().glob("*/job.json"), reverse=True):
        try:
            j = refresh_status(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
        rows.append({k: j.get(k) for k in ("id", "phase", "progress", "title", "format", "estimate_usd",
                                            "actual_usd", "created", "error_code")})
        rows[-1]["resumable"] = is_resumable(j)
    return rows


def _is_our_worker(pid: int, job_id: str) -> bool:
    """Whether pid is still this job's worker and not a recycled pid. POSIX asks ps for
    the command line; elsewhere (or when ps is missing) the pid is trusted."""
    if os.name == "nt":
        return True
    try:
        r = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return True
    if r.returncode != 0:
        return False
    return "_worker" in r.stdout and job_id in r.stdout


def _kill_tree(pid: int) -> None:
    """Stop the worker and the tool it is running (ltx2.py, build.py and their ffmpeg).
    POSIX: the worker leads its own process group (start_new_session), so signal the
    group. Windows: taskkill /T walks the child tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    import signal
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
            return
    except OSError:
        pass
    os.kill(pid, signal.SIGTERM)


def cancel(job_id: str) -> dict:
    job = load_job(job_id)
    if job["phase"] in TERMINAL:
        return job
    pid = job.get("worker_pid")
    if pid and pid_alive(pid) and not _is_our_worker(pid, job_id):
        log_line(job, f"cancel: pid {pid} is no longer this job's worker; not signalling it")
        pid = None
    if pid and pid_alive(pid):
        try:
            _kill_tree(pid)
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
            "file": "the person's own photo or clip: visual.path in the kit's assets or the agent workspace "
                    "(Ken Burns on a still, boomerang on a clip), optional headline/subline; free, no GPU",
            "none": "gradient placeholder card with the scene slug",
        },
        "host": host_capabilities(),
        "backgrounds": {"auto": "flux when this host has FLUX, else local", "flux": "generated, about a cent",
                        "local": "flat brand-colour field with grain, drawn here, free"},
        "kits": list_kits(),
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


# What each provider needs on this host: every env var in its list. A new host runs
# `uv run tools/verify_setup.py` / `modal_setup.py` to fill these; the job never guesses.
_NEEDS = {
    ("voice", "qwen3", "modal"): ["MODAL_QWEN3_TTS_ENDPOINT_URL"],
    ("voice", "qwen3", "runpod"): ["RUNPOD_QWEN3_TTS_ENDPOINT_ID", "RUNPOD_API_KEY"],
    ("voice", "elevenlabs", None): ["ELEVENLABS_API_KEY"],
    ("voice", "60db", None): ["SIXTYDB_API_KEY"],
    ("visual", "ltx", "modal"): ["MODAL_LTX2_ENDPOINT_URL"],
    ("visual", "flux", "modal"): ["MODAL_FLUX2_ENDPOINT_URL"],
    ("visual", "flux", "runpod"): ["RUNPOD_FLUX2_ENDPOINT_ID", "RUNPOD_API_KEY"],
    ("visual", "ideogram", None): ["IDEOGRAM_API_KEY"],
    ("music", "acemusic", None): ["ACEMUSIC_API_KEY"],
    ("music", "modal", None): ["MODAL_MUSIC_GEN_ENDPOINT_URL"],
    ("music", "runpod", None): ["RUNPOD_ACESTEP_ENDPOINT_ID", "RUNPOD_API_KEY"],
}


def _has(envs: list[str]) -> bool:
    return all(os.environ.get(e) for e in envs)


@functools.lru_cache(maxsize=1)
def _local_tools() -> dict:
    """The programs and packages a run needs on this machine, beyond keys. A job needs ffmpeg,
    ffprobe, moviepy and numpy (the build and QC run here); a session route also needs Node 18+
    (Remotion). verify_setup.py checks the same things in more detail."""
    import importlib.util
    node = shutil.which("node")
    node_major = None
    if node:
        try:
            out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
            node_major = int(out.strip().lstrip("v").split(".")[0])
        except Exception:  # noqa: BLE001
            node_major = None
    pkgs = {m: importlib.util.find_spec(m) is not None for m in ("moviepy", "numpy")}
    return {"ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe")),
            "node": node_major, "python_packages": pkgs}


def _job_tools_missing(local: dict) -> list[str]:
    miss = [b for b in ("ffmpeg", "ffprobe") if not local[b]]
    miss += [f"python package {m} (uv sync)" for m, ok in local["python_packages"].items() if not ok]
    return miss


def host_capabilities() -> dict:
    """What this host can make right now, from its environment (.env is loaded at import). An agent
    should offer only these; submit refuses the rest before any spend."""
    def ok(area, name, cloud=None):
        return _has(_NEEDS[(area, name, cloud)])
    flux = {"modal": ok("visual", "flux", "modal"), "runpod": ok("visual", "flux", "runpod")}
    kinds = {"ltx": ok("visual", "ltx", "modal"), "flux": any(flux.values()), "text": True,
             "ideogram": ok("visual", "ideogram"), "file": True, "none": True}
    voices = {"qwen3": {"modal": ok("voice", "qwen3", "modal"), "runpod": ok("voice", "qwen3", "runpod")},
              "elevenlabs": ok("voice", "elevenlabs"), "60db": ok("voice", "60db")}
    music = {p: ok("music", p) for p in ("acemusic", "modal", "runpod")}
    try:
        import importlib.util
        whisper = importlib.util.find_spec("whisper") is not None      # captions.py imports openai-whisper
    except Exception:  # noqa: BLE001
        whisper = False
    local = _local_tools()
    return {
        "routes": {"job": not _job_tools_missing(local),
                   "session": not _job_tools_missing(local) and (local["node"] or 0) >= 18},
        "local": local,
        "visual_kinds": kinds,
        "flux_clouds": flux,
        "voices": voices,
        "music": music,
        "captions": "scribe" if os.environ.get("ELEVENLABS_API_KEY") else ("whisper" if whisper else None),
        "r2_links": bool(os.environ.get("R2_ACCOUNT_ID")),
        "available_visual_kinds": [k for k, v in kinds.items() if v],
        "setup": "uv run tools/verify_setup.py; Modal endpoints: uv run tools/modal_setup.py --all",
    }


def _skip_host_check() -> bool:
    return os.environ.get("VIDEO_JOB_SKIP_HOST_CHECK", "").strip().lower() in ("1", "true", "yes", "on")


def missing_for(spec: dict, include_soft: bool = False) -> list[dict]:
    """What a normalised spec needs that this host doesn't have. Empty means it can run here.
    Captions, R2 and the music bed are soft (the job runs without them and says so), so they're
    listed only with include_soft, marked "soft": true. VIDEO_JOB_SKIP_HOST_CHECK=1 lists nothing."""
    if _skip_host_check():
        return []
    out, seen = [], set()
    for tool in _job_tools_missing(_local_tools()):
        out.append({"needed_for": "building and checking the video on this host", "env": [], "install": tool,
                    "hint": "install it, then run uv run tools/verify_setup.py"})

    def need(area, name, cloud, what, hint="", soft=False):
        key = (area, name, cloud)
        if key not in _NEEDS:
            known = sorted({f"{n}" + (f" on {c}" if c else "") for a, n, c in _NEEDS if a == area})
            raise SpecError(f"{what}: {area} {name!r}" + (f" on {cloud!r}" if cloud else "")
                            + f" isn't one this toolkit knows (known: {', '.join(known)})")
        if key in seen or _has(_NEEDS[key]):
            return
        seen.add(key)
        out.append({"needed_for": what, "env": _NEEDS[key], "hint": hint, **({"soft": True} if soft else {})})

    v = spec["voice"]
    if v["provider"] == "qwen3":
        cloud = v.get("cloud", "modal")
        other = "runpod" if cloud == "modal" else "modal"
        hint = (f'or set "cloud": "{other}" in voice' if _has(_NEEDS[("voice", "qwen3", other)])
                else 'or use "provider": "elevenlabs" with ELEVENLABS_API_KEY')
        need("voice", "qwen3", cloud, f"the qwen3 voice on {cloud}", hint)
    else:
        need("voice", v["provider"], None, f"the {v['provider']} voice")
    for sc in spec["scenes"]:
        kind = sc["visual"]["kind"]
        if kind == "ltx":
            need("visual", "ltx", "modal", "ltx clips", "or make those scenes ideogram cards")
        elif kind in ("flux", "text") and sc["visual"].get("source") != "local":
            cloud = sc["visual"].get("cloud", "modal")
            need("visual", "flux", cloud, f"{kind} scenes (FLUX on {cloud})",
                 'or "backgrounds": "local" for text scenes, an ideogram card, or a file')
        elif kind == "ideogram":
            need("visual", "ideogram", None, "ideogram cards")
    m = spec.get("music") or {}
    if m.get("enabled") and include_soft:
        need("music", m.get("provider", "acemusic"), None, "the music bed",
             'the video renders with narration only; or "music": false', soft=True)
    return out


def check_host(spec: dict) -> None:
    """Raise Unavailable when this host can't make the spec. VIDEO_JOB_SKIP_HOST_CHECK=1 turns it off
    (tests, and a host whose tools find their keys some other way)."""
    missing = missing_for(spec)
    if missing:
        raise Unavailable(missing)


def catalog() -> dict:
    """What the toolkit can make, from _internal/toolkit-registry.json, for an agent to answer "what can you
    make?" and pick a route. Each template's `agent_support` is `job` (one prompt: submit/status/result) or
    `session` (a Claude Code-style working session with the named skill, reviewed by the person who asked)."""
    reg = json.loads((ROOT / "_internal" / "toolkit-registry.json").read_text(encoding="utf-8"))
    skills = reg.get("skills", {})

    def skill_path(name):
        return (skills.get(name) or {}).get("path")

    examples = reg.get("examples", {})
    templates = []
    for key, t in reg.get("templates", {}).items():
        demos = [{"name": k, "description": e.get("description"), "demo": e.get("demo"), "path": e.get("path")}
                 for k, e in examples.items() if e.get("template") == key]
        templates.append({
            "name": key, "description": t.get("description"), "path": t.get("path"),
            "agent_support": t.get("agentSupport", "session"),   # job: one prompt; session: a working session
            "skill": t.get("skill"), "skill_path": skill_path(t.get("skill")) if t.get("skill") else None,
            "examples": demos,
        })
    # Routes that aren't a template: a style built from shared components, with the skill and example to copy.
    builds = []
    for key, e in examples.items():
        if e.get("template") in (None, "custom") and not e.get("showcaseOnly"):
            builds.append({"name": key, "description": e.get("description"), "stack": e.get("stack"),
                           "path": e.get("path"), "demo": e.get("demo")})
    brands = sorted(p.parent.name for p in (ROOT / "brands").glob("*/brand.json"))
    design = {k: {"description": v.get("description"), "skill": v.get("skill"), "default": v.get("default", False),
                  "recommended": v.get("recommended", False)}
              for k, v in reg.get("designSources", {}).items() if not k.startswith("_")}
    return {
        "version": reg.get("version"),
        "templates": templates,
        "example_builds": builds,
        "components": {k: c.get("description") for k, c in reg.get("components", {}).items()},
        "brands": brands,
        "design_sources": design,
        "voice_presets": sorted(p.stem for p in (ROOT / "assets" / "voices").glob("*.json")),
        "kits": list_kits(),
        "host": host_capabilities(),
        "skills": {k: {"path": v.get("path"), "description": v.get("description")} for k, v in skills.items()},
        "how_to_read_a_skill": "Skills are folders of markdown: open <path>/SKILL.md from the toolkit root and "
                               "follow it. They are written for Claude Code but any agent with a shell can use them.",
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
    for ev in (s.get("events") or [])[-4:]:
        head += f"\n  {ev.get('ts', '')[11:19]} {ev.get('msg', '')}"
    if s.get("eta_s"):
        head += f"\n  about {max(1, round(s['eta_s'] / 60))} min left"
    if s.get("error"):
        head += f"\n  error: {s['error']}"
    if s.get("resumable"):
        head += f"\n  resumable: uv run tools/video_job.py resume {s['id']}"
    for w in s.get("warnings") or []:
        head += f"\n  warning: {w}"
    if s.get("output", {}) and s["output"].get("mp4"):
        head += f"\n  mp4: {s['output']['mp4']}"
    return head


def _wants_progress_json(argv: list[str]) -> bool:
    for i, a in enumerate(argv):
        if a == "--progress=json" or (a == "--progress" and i + 1 < len(argv) and argv[i + 1] == "json"):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Every exit path under --progress json prints one cost event, $0 unless something
    was booked: argparse errors, refusals, missing jobs and crashes included."""
    global _cost_emitted
    ensure_utf8_output()
    argv = list(sys.argv[1:] if argv is None else argv)
    _cost_emitted = False
    progress_json = _wants_progress_json(argv)
    rc = 1
    try:
        rc = _main(argv)
    except SystemExit as e:          # argparse: --help (0) or a usage error (2)
        rc = e.code if isinstance(e.code, int) else 1
    finally:
        if progress_json and not _cost_emitted:
            emit_cost_event(0.0, "nothing generated" if rc == 0 else f"failed (exit {rc}), nothing generated")
    return rc


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Async video jobs: brief or spec in, QC-checked MP4 out.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_spec_args(p, with_max=False):
        p.add_argument("--brief", help="One-paragraph description of the video")
        p.add_argument("--spec", help="Finished spec JSON file ('-' for stdin)")
        p.add_argument("--spec-json", help="The spec as a JSON string, or @path/to/spec.json (relative to the "
                                           "agent workspace when VIDEO_JOB_WORKSPACE is set)")
        p.add_argument("--scene", action="append", default=[], metavar="TEXT[ :: KIND: PROMPT]",
                       help="Flat form, repeatable, one per scene: narration, then optionally ' :: ltx: motion "
                            "prompt', ' :: card: HEADLINE ; object' (or 'HEADLINE ; A -> B -> C' for a "
                            "flow diagram), ' :: flux: prompt'. No JSON needed")
        p.add_argument("--title", help="Video title (flat form; default: the first scene's text)")
        p.add_argument("--voice", help="Voice preset: assets/voices/<name>.json (a clone) or a brand name")
        p.add_argument("--kit", help="A kit (kits/README.md): its brand, voice, format and own photos fill the "
                                     "spec, and 'file:' scenes look in its assets")
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
    rs = sub.add_parser("resume", help="Carry on a failed or cancelled job from its first missing output")
    rs.add_argument("job_id")
    rs.add_argument("--max-usd", type=float, default=None, help="Raise (or set) the job's spend cap")
    rs.add_argument("--json", action="store_true")
    pr = sub.add_parser("prune", help="Delete old finished jobs' project trees (keeps job.json, receipts, MP4)")
    pr.add_argument("--keep", type=int, default=PRUNE_KEEP, help=f"Newest jobs never pruned (default {PRUNE_KEEP})")
    pr.add_argument("--older-than", type=float, default=None, metavar="DAYS", help="Only jobs finished this long ago")
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--json", action="store_true")
    sub.add_parser("list").add_argument("--json", action="store_true")
    st = sub.add_parser("settle", help="Write the ledger correction for finished reserve-mode jobs that never got one")
    st.add_argument("job_id", nargs="?")
    st.add_argument("--all", action="store_true", help="Every finished job without a settlement")
    st.add_argument("--json", action="store_true")
    sub.add_parser("presets").add_argument("--json", action="store_true")
    sub.add_parser("catalog", help="What the toolkit can make: templates, example builds, brands, design sources").add_argument("--json", action="store_true")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("job_id")
    for p in sub.choices.values():  # agent runtimes' wrappers inject this on every call
        p.add_argument("--progress", choices=["human", "json"], default="human",
                       help="json: emit a cost event on stderr (the estimate on submit, $0 otherwise)")
    args = ap.parse_args(argv)
    progress_json = getattr(args, "progress", "human") == "json"
    as_json = getattr(args, "json", False)

    try:
        if args.cmd in ("estimate", "submit", "spec"):
            kit = args.kit
            kit_fmt = (load_kit(kit).get("format") if kit else None)
            fmt = args.format or kit_fmt or "9:16"
            if args.scene:
                music = False if args.no_music else (args.music or None)
                sp = spec_from_scene_args(
                    args.title, args.scene, fmt, voice=args.voice, music=music, draft=args.draft,
                    title_card=not args.no_title_card, allow_screens=args.allow_screens)
                if kit:
                    sp["kit"] = kit
                spec, source = normalize_spec(sp), "scenes"
            elif args.spec_json:
                sp = parse_spec_text(args.spec_json)
                if kit and not sp.get("kit"):
                    sp["kit"] = kit
                spec, source = normalize_spec(sp), "spec"
            else:
                spec, source = resolve_spec(args.brief, args.spec, fmt, args.duration,
                                            allow_skeleton=(args.cmd == "estimate"), kit=kit)
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
                missing = missing_for(spec, include_soft=True)
                human = _human_estimate(est, source) + "".join(
                    "\n  " + ("without it" if m.get("soft") else "can't run here")
                    + f": {m['needed_for']} needs {needs_text(m)}" for m in missing)
                _print({**est, "spec_source": source, "host_missing": missing}, args.json, human)
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
        if progress_json and args.cmd in ("status", "result", "list", "cancel", "presets", "catalog", "prune"):
            emit_cost_event(0.0, f"{args.cmd}: no generation")
        if args.cmd == "resume":
            j = resume(args.job_id, max_usd=args.max_usd)
            if progress_json:
                emit_cost_event(j.get("resume_reserved_usd") or 0.0,
                                f"{'reserved the rest of' if j.get('resume_reserved_usd') else 'resumed'} job {j['id']}")
            _print({"id": j["id"], "phase": j["phase"], "resumes": j["resumes"], "actual_usd": j.get("actual_usd"),
                    "estimate_usd": j.get("estimate_usd"), "max_usd": j.get("max_usd"),
                    "poll": f"uv run tools/video_job.py status {j['id']} --json"},
                   args.json, f"resumed {j['id']}  spent so far ${j.get('actual_usd') or 0:.2f}  "
                              f"(poll: uv run tools/video_job.py status {j['id']})")
            return 0
        if args.cmd == "prune":
            r = prune(keep=max(0, args.keep), older_than_days=args.older_than, dry_run=args.dry_run)
            _print(r, args.json, f"{'would prune' if r['dry_run'] else 'pruned'} {len(r['pruned'])} jobs, "
                                 f"{r['freed_bytes'] / 1e6:.0f} MB")
            return 0
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
        if args.cmd == "catalog":
            _print(catalog(), True)
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
    except Unavailable as e:
        _print({"error": str(e), "error_code": "unavailable", "missing": e.missing}, args.json, f"refused: {e}")
        return 2
    except NoExpander as e:
        _print({"error": str(e), "error_code": "no_expander"}, args.json, f"refused: {e}")
        return 2
    except LockHeld as e:
        _print({"error": str(e), "error_code": "lock_held", "holder": e.holder}, as_json, f"busy: {e}")
        return 3
    except JobUnreadable as e:
        _print({"error": str(e), "error_code": "job_unreadable"}, as_json, f"error: {e}")
        return 1
    except FileNotFoundError as e:
        code = "not_found" if str(e).startswith("no such job") else "bad_request"
        _print({"error": str(e), "error_code": code}, as_json, f"error: {e}")
        return 1
    except (SpecError, ValueError) as e:
        _print({"error": str(e), "error_code": "bad_request"}, as_json, f"error: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 — an agent gets a JSON error, never a traceback
        _print({"error": f"{type(e).__name__}: {e}", "error_code": "internal"}, as_json, f"error: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
