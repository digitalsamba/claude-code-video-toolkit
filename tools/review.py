#!/usr/bin/env python3
"""Review a project's scenes in a local browser; comments and approvals stay local."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import urlsplit

from project_manifest import ProjectValidationError, load_project, manifest_path, resolve_project_path
from win_encoding import ensure_utf8_output


UI_ROOT = Path(__file__).resolve().parent / "review_ui"
MEDIA_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm", ".mov",
               ".mp3", ".wav", ".m4a", ".aac", ".ogg"}
VIDEO_TYPES = {".mp4", ".webm", ".mov"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
MAX_REQUEST_BYTES = 32 * 1024


class ReviewError(ValueError):
    pass


class StaleReviewError(ReviewError):
    pass


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    """One project, serialized mutations, and approvals tied to current file bytes."""

    def __init__(self, project: str | Path):
        self.manifest = manifest_path(project).resolve()
        self.root = self.manifest.parent
        load_project(self.manifest)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def _state_path(self) -> Path:
        return resolve_project_path(self.root, ".video-toolkit/review.json")

    def _load_state(self) -> dict:
        path = self._state_path()
        if not path.exists():
            return {"version": 1, "scenes": {}}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("scenes"), dict):
                raise ValueError("expected version 1 and a scenes object")
            for record in state["scenes"].values():
                if not isinstance(record, dict) or not isinstance(record.get("comments", []), list):
                    raise ValueError("invalid scene record")
                for comment in record.get("comments", []):
                    if not isinstance(comment, dict) or not all(isinstance(comment.get(k), str) for k in ("id", "text", "createdAt")):
                        raise ValueError("invalid comment")
                approval = record.get("approval")
                if approval is not None and (not isinstance(approval, dict) or not all(
                        isinstance(approval.get(k), str) for k in ("fingerprint", "createdAt"))):
                    raise ValueError("invalid approval")
            return state
        except (ValueError, UnicodeError) as error:
            raise ReviewError(f"Cannot read .video-toolkit/review.json: {error}") from error

    def _save_state(self, state: dict) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".review-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def _source_fingerprint(self, project: dict) -> str:
        context = {key: value for key, value in project.items() if key not in ("scenes", "sessions", "updated", "phase")}
        digest = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        candidates = [self.root / name for name in ("config.json", "scenes.json", "remotion.config.ts", "VOICEOVER-SCRIPT.md")]
        source = self.root / "src"
        if source.exists():
            candidates.extend(path for path in source.rglob("*") if path.suffix in (".ts", ".tsx", ".js", ".jsx", ".css", ".json"))
        for path in sorted(candidates):
            if path.is_file():
                contained = resolve_project_path(self.root, path.relative_to(self.root).as_posix())
                digest.update(path.relative_to(self.root).as_posix().encode())
                digest.update(_digest_file(contained).encode())
        return digest.hexdigest()

    def _qc_findings(self) -> list[dict]:
        candidates = ['.video-toolkit/reports/qc-after.json', '.video-toolkit/reports/qc-before.json', 'out/qc.json']
        paths = [resolve_project_path(self.root, name) for name in candidates]
        path = next((item for item in paths if item.is_file()), None)
        if path is None:
            return []
        if path.stat().st_size > 2 * 1024 * 1024:
            return [{"level": "warning", "message": "QC report is too large to display."}]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            findings = data.get("findings", []) if isinstance(data, dict) else []
            return [{"level": str(f.get("level", "warning")), "message": str(f.get("message", "")),
                     'timeSeconds': f.get('startSeconds', f.get('timeSeconds'))}
                    for f in findings[:100] if isinstance(f, dict)]
        except (ValueError, UnicodeError):
            return [{"level": "warning", "message": "out/qc.json could not be read."}]

    def _snapshot(self) -> tuple[dict, dict[str, Path]]:
        project = load_project(self.manifest)
        state = self._load_state()
        context = self._source_fingerprint(project)
        media: dict[str, Path] = {}
        digests: dict[Path, str] = {}

        def reference(value: object, *, public: bool) -> tuple[dict | None, str, str | None]:
            if not isinstance(value, str) or not value:
                return None, "", None
            try:
                # Only explicitly referenced, supported media can leave this server.
                if any(part.startswith(".") or part == "_internal" for part in value.replace("\\", "/").split("/")):
                    raise ReviewError("hidden or internal files are not review media")
                base = self.root / "public" if public else self.root
                path = resolve_project_path(base, value)
                if path.suffix.lower() not in MEDIA_TYPES:
                    raise ReviewError("unsupported media type")
                if not path.is_file():
                    return None, f"missing:{value}", f"Missing asset: {value}"
                key = hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:24]
                media[key] = path
                if path not in digests:
                    digests[path] = _digest_file(path)
                kind = "video" if path.suffix.lower() in VIDEO_TYPES else "image" if path.suffix.lower() in IMAGE_TYPES else "audio"
                return {"url": f"/media/{key}", "kind": kind, "name": value}, digests[path], None
            except (OSError, ReviewError, ProjectValidationError) as error:
                return None, f"unavailable:{value}", f"Unavailable asset: {value} ({error})"

        rendered, render_hash, render_issue = reference(project.get("render", {}).get("file") or project.get("output"), public=False)
        scenes = []
        cursor = 0.0
        for scene in project["scenes"]:
            visual = scene.get("visual", {})
            assets = [visual.get("asset"), *visual.get("assets", [])]
            thumbnail = scene.get("thumbnailFile") or visual.get("thumbnail")
            audio_ref = scene.get("audio", {}).get("file") or scene.get("audioFile")
            refs = [reference(value, public=True) for value in assets if value]
            thumb, thumb_hash, thumb_issue = reference(thumbnail, public=True)
            audio, audio_hash, audio_issue = reference(audio_ref, public=True)
            script_hash = ""
            if scene.get("scriptFile"):
                script = resolve_project_path(self.root, scene["scriptFile"])
                script_hash = _digest_file(script) if script.is_file() else "missing-script"
            fingerprint = hashlib.sha256(json.dumps({"scene": scene, "context": context,
                "assets": [item[1] for item in refs], "thumbnail": thumb_hash, "audio": audio_hash,
                "script": script_hash, "render": render_hash}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            record = state["scenes"].get(scene["id"], {})
            approval = record.get("approval")
            status = "approved" if approval and approval["fingerprint"] == fingerprint else "stale" if approval else "pending"
            duration = scene.get("durationSeconds")
            start = scene.get("startSeconds", cursor)
            preview = rendered if rendered and rendered["kind"] == "video" else next((item[0] for item in refs if item[0] and item[0]["kind"] != "audio"), None)
            issues = [item[2] for item in refs if item[2]] + [item for item in (thumb_issue, audio_issue, render_issue) if item]
            scenes.append({"id": scene["id"], "type": scene["type"], "title": scene.get("title") or scene["id"],
                "durationSeconds": duration, "startSeconds": start, "narration": scene.get("narration") or "",
                "assetStatus": scene.get("status", "planned"), "preview": preview, "thumbnail": thumb,
                "previewStart": start if preview is rendered else 0, "audio": audio,
                "fingerprint": fingerprint, "reviewStatus": status, "comments": record.get("comments", []),
                "issues": issues})
            cursor = start + (duration or 0)
        return {"name": project["name"], "template": project["template"], "phase": project.get("phase", "planning"),
                "scenes": scenes, "durationSeconds": cursor, "warnings": self._qc_findings(), "token": self.token}, media

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()[0]

    def media_path(self, key: str) -> Path | None:
        with self.lock:
            return self._snapshot()[1].get(key)

    def mutate(self, body: object) -> dict:
        if not isinstance(body, dict) or not all(isinstance(body.get(key), str) for key in ("sceneId", "fingerprint", "action")):
            raise ReviewError("Expected sceneId, fingerprint and action strings")
        with self.lock:
            snapshot, _ = self._snapshot()
            scene = next((item for item in snapshot["scenes"] if item["id"] == body["sceneId"]), None)
            if not scene:
                raise ReviewError("Unknown scene")
            if body["fingerprint"] != scene["fingerprint"]:
                raise StaleReviewError("The scene changed. Refresh before saving your review.")
            state = self._load_state()
            record = state["scenes"].setdefault(scene["id"], {"comments": [], "approval": None})
            if body["action"] == "approve":
                record["approval"] = {"fingerprint": scene["fingerprint"], "createdAt": _stamp()}
            elif body["action"] == "reopen":
                record["approval"] = None
            elif body["action"] == "comment":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                    raise ReviewError("Comment must contain 1–4000 characters")
                position = body.get('timeSeconds')
                if position is not None and (isinstance(position, bool) or not isinstance(position, (int, float))
                        or not 0 <= position < 86400):
                    raise ReviewError('timeSeconds must be between zero and 86400')
                record.setdefault("comments", []).append({"id": secrets.token_hex(8), "text": text.strip(),
                    "createdAt": _stamp(), 'timeSeconds': position})
            else:
                raise ReviewError("Unknown action; use approve, reopen or comment")
            self._save_state(state)
            return self._snapshot()[0]


def make_server(project: str | Path, port: int = 0) -> ThreadingHTTPServer:
    store = ReviewStore(project)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _trusted_host(self) -> bool:
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def _headers(self, status: int, content_type: str, length: int, extra: dict | None = None) -> None:
            self.send_response(status)
            for key, value in {"Content-Type": content_type, "Content-Length": str(length), "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
                **(extra or {})}.items():
                self.send_header(key, value)
            self.end_headers()

        def _json(self, status: int, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def do_GET(self):
            if not self._trusted_host():
                self._json(403, {"error": "Use the localhost review URL"})
                return
            route = urlsplit(self.path).path
            try:
                if route == "/api/project":
                    self._json(200, store.snapshot())
                elif route in ("/", "/ui/review.css", "/ui/review.js"):
                    name = "index.html" if route == "/" else route.rsplit("/", 1)[1]
                    payload = (UI_ROOT / name).read_bytes()
                    kind = {"index.html": "text/html", "review.css": "text/css", "review.js": "text/javascript"}[name]
                    self._headers(200, f"{kind}; charset=utf-8", len(payload))
                    self.wfile.write(payload)
                elif re.fullmatch(r"/media/[a-f0-9]{24}", route):
                    path = store.media_path(route.rsplit("/", 1)[1])
                    if path is None:
                        self._json(404, {"error": "Media not found"})
                        return
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    range_header = self.headers.get("Range")
                    if range_header:
                        match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                        if not match or int(match[1]) >= size:
                            self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                            return
                        start, end = int(match[1]), min(int(match[2]) if match[2] else end, end)
                        if end < start:
                            self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                            return
                    headers = {"Accept-Ranges": "bytes"}
                    if range_header:
                        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                    self._headers(206 if range_header else 200, mimetypes.guess_type(path.name)[0] or "application/octet-stream", max(0, end - start + 1), headers)
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            chunk = stream.read(min(256 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                else:
                    self._json(404, {"error": "Not found"})
            except (OSError, ValueError, ProjectValidationError) as error:
                self._json(400, {"error": str(error)})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self._json(400, {'error': 'Invalid Content-Length'})
                return
            if not 0 < length <= MAX_REQUEST_BYTES:
                self._json(400, {'error': 'Expected a JSON body under 32 KiB'})
                return
            # Consume a bounded body before rejecting authorization; Windows
            # otherwise resets the socket with unread data and hides the 403.
            raw_body = self.rfile.read(length)
            expected_origins = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
            if (not self._trusted_host() or self.headers.get("Origin", expected_origins[0]) not in expected_origins
                    or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), store.token)):
                self._json(403, {"error": "Review token or origin is invalid"})
                return
            if self.path != "/api/review":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get_content_type() != "application/json":
                    self._json(400, {"error": "Expected a JSON body under 32 KiB"})
                    return
                body = json.loads(raw_body)
                self._json(200, store.mutate(body))
            except StaleReviewError as error:
                self._json(409, {"error": str(error)})
            except (OSError, ValueError, ProjectValidationError) as error:
                self._json(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.review_store = store
    return server


def main() -> int:
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Project directory or project.json")
    parser.add_argument("--port", type=int, default=0, help="Local port; default picks a free port")
    args = parser.parse_args()
    try:
        server = make_server(args.project, args.port)
    except (OSError, ValueError, ProjectValidationError) as error:
        parser.exit(1, f"Cannot start review: {error}\n")
    print(f"Review ready: http://127.0.0.1:{server.server_port}", flush=True)
    print("Press Ctrl+C to stop. Comments and approvals are saved locally.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
