#!/usr/bin/env python3
"""
Capture reference photos of a person for a character model, from a webcam.

Opens a local page that talks you through a set of poses (front, three-quarter,
both profiles, a smile, glasses off, full body), counts down out loud so you
can look away from the screen, takes a short burst and keeps the sharpest
frame. Photos are saved straight to the output folder; nothing leaves this
machine.

Usage:
    uv run tools/reference_capture.py --out projects/my-character/reference
    uv run tools/reference_capture.py --out refs --port 8770 --no-open

Tips: put the camera at eye level and sit at least a metre back (a close, low
camera distorts the face), face a window for even light, and use a plain wall
behind you. On a Mac an iPhone shows up in the camera list (Continuity Camera)
and is much sharper than the built-in webcam.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
MAX_BYTES = 40 * 1024 * 1024

# name: the file saved. turn: degrees the head turns, positive to the sitter's left
# (the camera then sees the right side of the head). wait: seconds to get into the pose.
POSES = [
    {"name": "front", "title": "Front", "turn": 0, "wait": 4,
     "say": "Face the camera. Relaxed face, mouth closed."},
    {"name": "three-quarter-right-side", "title": "Three quarter, right side", "turn": 45, "wait": 6,
     "say": "Turn your head halfway to your left. Chin level, eyes on something at eye height."},
    {"name": "profile-right-side", "title": "Profile, right side", "turn": 90, "wait": 9,
     "say": "Turn your whole body to your left until your shoulder points at the camera. Chin level, look straight ahead, not at the screen."},
    {"name": "three-quarter-left-side", "title": "Three quarter, left side", "turn": -45, "wait": 9,
     "say": "Now face the camera, then turn your head halfway to your right. Chin level."},
    {"name": "profile-left-side", "title": "Profile, left side", "turn": -90, "wait": 9,
     "say": "Turn your whole body to your right until your other shoulder points at the camera. Chin level, look straight ahead."},
    {"name": "front-smile", "title": "Front, smiling", "turn": 0, "wait": 5,
     "say": "Face the camera again and smile."},
    {"name": "front-no-glasses", "title": "Front, glasses off", "turn": 0, "wait": 6,
     "say": "If you wear glasses, take them off. Face the camera, relaxed face."},
    {"name": "full-body", "title": "Full body", "turn": 0, "wait": 15, "body": True,
     "say": "Stand up and step back until the camera sees you from head to feet, with nothing in front of you. Arms a little away from your sides."},
]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reference capture</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#30343c; --text:#eceef2; --dim:#9aa1ad; --accent:#f2a33a; --ok:#5fc27e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.45 system-ui, sans-serif; }
  main { max-width:1180px; margin:0 auto; padding:20px 16px 40px; display:grid; gap:20px; grid-template-columns:minmax(0,1fr) 320px; }
  @media (max-width: 860px) { main { grid-template-columns:1fr; } }
  h1 { font-size:20px; margin:0 0 4px; }
  p { margin:0; color:var(--dim); }
  .stage { position:relative; background:#000; border-radius:8px; overflow:hidden; aspect-ratio:16/9; max-height:78vh; margin-inline:auto; }
  video { width:100%; height:100%; object-fit:contain; transform:scaleX(-1); display:block; }
  svg.guide { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .count { position:absolute; inset:0; display:grid; place-items:center; font-size:140px; font-weight:700; color:#fff; text-shadow:0 2px 12px #000; pointer-events:none; }
  .flash { position:absolute; inset:0; background:#fff; opacity:0; pointer-events:none; transition:opacity .25s; }
  .now { margin-top:14px; display:flex; gap:16px; align-items:center; }
  .now svg { flex:none; }
  .now h2 { font-size:22px; margin:0; }
  .now p { color:var(--text); }
  .bar { margin-top:14px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  button, select { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:9px 14px; }
  button { cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#1a1300; font-weight:600; }
  button:disabled { opacity:.45; cursor:default; }
  select { max-width:260px; }
  .meta { font-size:13px; color:var(--dim); }
  ol { list-style:none; margin:0; padding:0; display:grid; gap:8px; }
  li { display:grid; grid-template-columns:72px 1fr auto; gap:10px; align-items:center; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; }
  li.current { border-color:var(--accent); }
  li .thumb { width:72px; height:54px; border-radius:4px; background:#0c0d10 center/cover no-repeat; }
  li b { display:block; font-size:14px; font-weight:600; }
  li span { font-size:12px; color:var(--dim); }
  li.done span { color:var(--ok); }
  li button { padding:5px 10px; font-size:13px; }
  .tips { margin-top:18px; font-size:14px; color:var(--dim); }
  .tips li { display:list-item; background:none; border:0; padding:0; margin-left:18px; list-style:disc; }
  .err { color:#ff8f7a; }
</style>
</head>
<body>
<main>
  <section>
    <h1>Reference capture</h1>
    <p>Photos for a character model. They are saved to <code id="out"></code> on this machine and go nowhere else.</p>
    <div class="stage" style="margin-top:14px">
      <video id="video" autoplay playsinline muted></video>
      <svg class="guide" id="guide" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="count" id="count"></div>
      <div class="flash" id="flash"></div>
    </div>
    <div class="now">
      <svg id="compass" width="84" height="84" viewBox="-42 -42 84 84"></svg>
      <div><h2 id="poseTitle"></h2><p id="poseSay"></p></div>
    </div>
    <div class="bar">
      <button class="primary" id="run">Start all poses</button>
      <button id="one">Take this pose (space)</button>
      <button id="stop" disabled>Stop</button>
      <select id="cams" aria-label="Camera"></select>
      <label class="meta"><input type="checkbox" id="voice" checked> Voice</label>
      <span class="meta" id="res"></span>
    </div>
    <p class="meta err" id="err"></p>
    <ul class="tips">
      <li>Camera at eye level, at least a metre away. If you can see the ceiling, the camera is too low.</li>
      <li>Fill the oval with your head and keep your eyes on the line.</li>
      <li>Face a window or a lamp so both sides of the face are lit. Plain wall behind you.</li>
      <li>For the profiles, turn your whole body, not only your neck, and listen for the beeps.</li>
      <li>The diagram shows your head from above. The dot is the camera.</li>
    </ul>
  </section>
  <aside>
    <ol id="list"></ol>
  </aside>
</main>
<script>
const POSES = __POSES__;
const OUT = __OUT__;
const $ = id => document.getElementById(id);
const video = $('video');
let frame = [1600, 900], current = 0, running = false, stopFlag = false, stream = null, audio = null;
const saved = {};   // name -> cache-busting stamp

$('out').textContent = OUT;

function compass(turn) {
  // Head from above, mirrored like the preview: turning to your left moves the nose to screen left.
  const a = (-turn) * Math.PI / 180;
  const nx = Math.sin(a) * 26, ny = 26 * Math.cos(a);
  $('compass').innerHTML =
    `<circle cx="0" cy="36" r="4" fill="#f2a33a"/>` +
    `<g transform="translate(0,-6)"><circle r="20" fill="none" stroke="#eceef2" stroke-width="3"/>` +
    `<line x1="0" y1="0" x2="${nx}" y2="${ny}" stroke="#eceef2" stroke-width="3" stroke-linecap="round"/>` +
    `<circle cx="${nx}" cy="${ny}" r="4" fill="#eceef2"/></g>`;
}

function guide(pose) {
  const s = 'fill="none" stroke="rgba(255,255,255,.55)" stroke-width="4" stroke-dasharray="14 12"';
  const [W, H] = frame, cx = W / 2, cy = H * 0.45, ry = Math.min(H * 0.3, W * 0.42), rx = ry * 0.76;
  $('guide').innerHTML = pose.body
    ? `<rect x="${cx - H * 0.24}" y="${H * 0.05}" width="${H * 0.48}" height="${H * 0.9}" rx="60" ${s}/>`
    : `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" ${s}/>` +
      `<line x1="${cx - rx * 1.2}" y1="${cy - ry * 0.1}" x2="${cx + rx * 1.2}" y2="${cy - ry * 0.1}" stroke="rgba(255,255,255,.35)" stroke-width="3"/>`;
}

function show(i) {
  current = i;
  const p = POSES[i];
  $('poseTitle').textContent = `${i + 1} of ${POSES.length}: ${p.title}`;
  $('poseSay').textContent = p.say;
  compass(p.turn); guide(p); renderList();
}

function renderList() {
  $('list').innerHTML = '';
  POSES.forEach((p, i) => {
    const li = document.createElement('li');
    const done = saved[p.name];
    li.className = (i === current ? 'current ' : '') + (done ? 'done' : '');
    const t = document.createElement('div'); t.className = 'thumb';
    if (done) t.style.backgroundImage = `url(/img/${p.name}.jpg?${done})`;
    const d = document.createElement('div');
    d.innerHTML = `<b></b><span></span>`;
    d.firstChild.textContent = p.title;
    d.lastChild.textContent = done ? 'Saved' : 'Not taken';
    const b = document.createElement('button');
    b.textContent = done ? 'Retake' : 'Select';
    b.disabled = running;
    b.onclick = () => show(i);
    li.append(t, d, b);
    $('list').append(li);
  });
}

function beep(freq, ms) {
  audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; g.gain.value = 0.15;
  o.connect(g); g.connect(audio.destination);
  o.start(); o.stop(audio.currentTime + ms / 1000);
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

function speak(text) {
  return new Promise(resolve => {
    if (!$('voice').checked || !window.speechSynthesis) return resolve();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1; u.onend = resolve; u.onerror = resolve;
    speechSynthesis.cancel(); speechSynthesis.speak(u);
    setTimeout(resolve, 12000);   // some browsers never fire onend
  });
}

function grab() {
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  return c;
}

function sharpness(canvas) {
  // Variance of a Laplacian over the middle of the frame, on a small grey copy.
  const w = 320, h = Math.round(320 * canvas.height / canvas.width);
  const s = document.createElement('canvas'); s.width = w; s.height = h;
  const ctx = s.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  let sum = 0, sq = 0, n = 0;
  for (let y = Math.round(h * .15); y < h * .85; y++) for (let x = Math.round(w * .25); x < w * .75; x++) {
    const v = 4 * g[y * w + x] - g[y * w + x - 1] - g[y * w + x + 1] - g[(y - 1) * w + x] - g[(y + 1) * w + x];
    sum += v; sq += v * v; n++;
  }
  return sq / n - (sum / n) ** 2;
}

async function capture(pose) {
  let best = null, bestScore = -1;
  for (let i = 0; i < 5; i++) {
    const c = grab(), sc = sharpness(c);
    if (sc > bestScore) { best = c; bestScore = sc; }
    await sleep(160);
  }
  $('flash').style.opacity = .8; setTimeout(() => $('flash').style.opacity = 0, 120);
  beep(1320, 180);
  const blob = await new Promise(r => best.toBlob(r, 'image/jpeg', 0.95));
  const res = await fetch(`/save?name=${pose.name}`, { method: 'POST', body: blob });
  if (!res.ok) throw new Error(`Save failed: ${await res.text()}`);
  saved[pose.name] = Date.now();
}

async function takePose(i) {
  show(i);
  const p = POSES[i];
  await speak(p.say);
  for (let t = p.wait; t > 3 && !stopFlag; t--) await sleep(1000);
  for (let t = 3; t >= 1 && !stopFlag; t--) { $('count').textContent = t; beep(660, 120); await sleep(1000); }
  $('count').textContent = '';
  if (stopFlag) return;
  await capture(p);
  renderList();
}

async function run(all) {
  if (running) return;
  if (!video.videoWidth) { $('err').textContent = 'No camera picture yet.'; return; }
  running = true; stopFlag = false; $('err').textContent = '';
  $('run').disabled = $('one').disabled = true; $('stop').disabled = false;
  try {
    if (all) {
      for (let i = 0; i < POSES.length && !stopFlag; i++) await takePose(i);
      if (!stopFlag) await speak('All done. Check the photos and retake any you do not like.');
    } else {
      await takePose(current);
      if (!stopFlag && current < POSES.length - 1) show(current + 1);
    }
  } catch (e) { $('err').textContent = e.message; }
  running = false; $('count').textContent = '';
  if (window.speechSynthesis) speechSynthesis.cancel();
  $('run').disabled = $('one').disabled = false; $('stop').disabled = true;
  renderList();
}

async function startCamera(deviceId) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  const want = { width: { ideal: 3840 }, height: { ideal: 2160 } };
  if (deviceId) want.deviceId = { exact: deviceId };
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: want, audio: false });
  } catch (e) {
    $('err').textContent = `Camera not available: ${e.message}. Allow camera access for this page and reload.`;
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  const s = stream.getVideoTracks()[0].getSettings();
  const vw = video.videoWidth || s.width, vh = video.videoHeight || s.height;
  $('res').textContent = `${vw} x ${vh}`;
  frame = vw >= vh ? [1600, 900] : [900, 1600];
  document.querySelector('.stage').style.aspectRatio = `${vw} / ${vh}`;
  $('guide').setAttribute('viewBox', `0 0 ${frame[0]} ${frame[1]}`);
  guide(POSES[current]);
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  $('cams').innerHTML = '';
  cams.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
    o.selected = d.deviceId === s.deviceId;
    $('cams').append(o);
  });
}

$('cams').onchange = e => startCamera(e.target.value);
$('run').onclick = () => run(true);
$('one').onclick = () => run(false);
$('stop').onclick = () => { stopFlag = true; };
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !e.target.matches('select, input')) { e.preventDefault(); run(false); }
});

(async () => {
  try {
    const have = await (await fetch('/list')).json();
    have.forEach(n => { saved[n] = Date.now(); });
  } catch (e) {}
  show(0);
  startCamera();
})();
</script>
</body>
</html>
"""


def make_handler(out_dir: Path):
    page = (PAGE.replace("__POSES__", json.dumps(POSES))
                .replace("__OUT__", json.dumps(str(out_dir)))).encode("utf-8")
    names = {p["name"] for p in POSES}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: saves are printed below
            pass

        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/list":
                have = sorted(n for n in names if (out_dir / f"{n}.jpg").is_file())
                return self._send(200, json.dumps(have).encode(), "application/json")
            if path.startswith("/img/") and path.endswith(".jpg"):
                name = path[len("/img/"):-len(".jpg")]
                f = out_dir / f"{name}.jpg"
                if name in names and f.is_file():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            self._send(404, b"not found")

        def do_POST(self):
            url = urlparse(self.path)
            name = (parse_qs(url.query).get("name") or [""])[0]
            if url.path != "/save" or not NAME_RE.match(name) or name not in names:
                return self._send(400, b"unknown pose")
            # Only this page may save: a request from another site carries its own Origin.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
                return self._send(403, b"wrong origin")
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_BYTES:
                return self._send(400, b"bad size")
            data = self.rfile.read(size)
            if not data.startswith(b"\xff\xd8"):
                return self._send(400, b"not a JPEG")
            target = out_dir / f"{name}.jpg"
            tmp = target.with_suffix(".jpg.part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            print(f"saved {target} ({size // 1024} KB)", flush=True)
            self._send(200, b"ok")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture reference photos of a person from a webcam, pose by pose.")
    ap.add_argument("--out", default="reference-capture", help="Folder for the photos (created if missing)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="Don't open the browser")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(out_dir))
    except OSError as e:
        print(f"Can't listen on port {args.port}: {e}. Try --port.", file=sys.stderr)
        return 1
    # localhost, not 127.0.0.1: browsers allow the camera on http://localhost
    url = f"http://localhost:{args.port}/"
    print(f"Reference capture at {url}\nSaving to {out_dir}\nCtrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
