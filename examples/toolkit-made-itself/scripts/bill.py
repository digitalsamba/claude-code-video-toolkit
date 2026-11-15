"""What this video cost, measured: src/bill.json for the generate scene's chart.

- Voice: ElevenLabs history (credits per generation) since project.json's `started`, at Starter's
  $5 for 30,000 credits.
- Captions: Scribe at $0.40 an audio hour, over the seconds each captions capture covered.
- Pictures and video: the "Est. cost" lines the tools printed (capture/flux*.txt, capture/ltx*.txt),
  Modal GPU seconds at the tool's rate.
- Music: acemusic's API, free. Found sounds: Openverse, free.
- Sound effects made with sfx.py: ElevenLabs bills them in credits outside the TTS history, so they
  are left out until the account's counter shows them (EXTRA_CREDITS below once known).
Run from anywhere: uv run python examples/toolkit-made-itself/scripts/bill.py (needs the project.json of the build session; kept as the record of how the bill was measured)
"""
import datetime as dt, json, os, pathlib, re, subprocess, urllib.request
here = pathlib.Path(__file__).resolve().parent.parent
root = here.parent.parent
CREDIT_USD = 5 / 30000
SCRIBE_USD_S = 0.40 / 3600
EXTRA_CREDITS = 0

def env(key):
    for line in (root / ".env").read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get(key)

started = dt.datetime.fromisoformat(json.loads((here / "project.json").read_text())["started"].replace("Z", "+00:00")).timestamp()
req = urllib.request.Request("https://api.elevenlabs.io/v1/history?page_size=200", headers={"xi-api-key": env("ELEVENLABS_API_KEY")})
hist = json.load(urllib.request.urlopen(req))["history"]
voice = [h for h in hist if h["date_unix"] >= started]
credits = sum(h["character_count_change_to"] - h["character_count_change_from"] for h in voice) + EXTRA_CREDITS

ansi = re.compile(r"\x1b\[[0-9;]*m")
def capture_costs(prefix):
    total, n = 0.0, 0
    for f in sorted((here / "capture").glob(prefix + "*.txt")):
        for line in ansi.sub("", f.read_text()).splitlines():
            m = re.search(r"Est\. cost: \$([0-9.]+)", line)
            if m:
                total += float(m.group(1)); n += 1
    return total, n

def seconds(f):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)]))
captioned = sum(seconds(f) for f in (here / "public/audio/scenes").glob("*.mp3"))

takes = len(list((here / "public/audio/music").rglob("bg_*.mp3")))
flux, nflux = capture_costs("flux")
ltx, nltx = capture_costs("ltx")
rows = [
    {"label": "Voice", "short": f"{len(voice)} takes", "detail": f"{len(voice)} takes, {credits} ElevenLabs credits", "usd": round(credits * CREDIT_USD, 4)},
    {"label": "Captions", "short": "Scribe", "detail": f"{captioned:.0f} s through Scribe", "usd": round(captioned * SCRIBE_USD_S, 4)},
    {"label": "Music", "short": f"{takes} takes", "detail": f"{takes} takes on acemusic", "usd": 0.0},
    {"label": "Stills", "short": f"{nflux} tries", "detail": f"{nflux} tries on FLUX.2", "usd": round(flux, 4)},
    {"label": "Video", "short": f"{nltx} clip", "detail": f"{nltx} clip on LTX-2.3", "usd": round(ltx, 4)},
]
out = {"rows": rows, "total_usd": round(sum(r["usd"] for r in rows), 4), "measured": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
(here / "src/bill.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
