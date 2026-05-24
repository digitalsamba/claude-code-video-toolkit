"""Shared file transfer helpers for cloud GPU tools.

Provides R2 upload/download with fallback to free services (litterbox, 0x0.st).
Used by all cloud GPU tools to avoid duplicating file transfer logic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def get_r2_client():
    """Get boto3 S3 client configured for Cloudflare R2.

    Returns (client, config_dict) or (None, None) if R2 is not configured.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if not r2_config:
        return None, None

    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        return client, r2_config
    except ImportError:
        print("  boto3 not installed, skipping R2", file=sys.stderr)
        return None, None


def upload_to_r2(file_path: str, prefix: str) -> tuple[str | None, str | None]:
    """Upload to Cloudflare R2 and return presigned download URL.

    Returns (presigned_url, object_key) or (None, None) on failure.
    """
    client, config = get_r2_client()
    if not client:
        return None, None

    import uuid
    file_name = Path(file_path).name
    object_key = f"{prefix}/{uuid.uuid4().hex[:8]}_{file_name}"

    try:
        client.upload_file(file_path, config["bucket_name"], object_key)

        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return url, object_key
    except Exception as e:
        print(f"  R2 upload error: {e}", file=sys.stderr)
        return None, None


def download_from_r2(object_key: str, output_path: str) -> bool:
    """Download object from R2 to local path."""
    client, config = get_r2_client()
    if not client:
        return False

    try:
        client.download_file(config["bucket_name"], object_key, output_path)
        return True
    except Exception as e:
        print(f"  R2 download error: {e}", file=sys.stderr)
        return False


def delete_from_r2(object_key: str) -> bool:
    """Delete object from R2 after job completion."""
    client, config = get_r2_client()
    if not client or not object_key:
        return False

    try:
        client.delete_object(Bucket=config["bucket_name"], Key=object_key)
        return True
    except Exception:
        return False


def _upload_to_litterbox(file_path: str, file_name: str) -> str | None:
    """Upload to litterbox.catbox.moe (200MB limit, 24h retention)."""
    import subprocess
    result = subprocess.run(
        [
            "curl", "-s",
            "-F", "reqtype=fileupload",
            "-F", "time=24h",
            "-F", f"fileToUpload=@{file_path}",
            "https://litterbox.catbox.moe/resources/internals/api.php",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
    return None


def _upload_to_0x0(file_path: str, file_name: str) -> str | None:
    """Upload to 0x0.st (512MB limit, 30 day retention)."""
    import subprocess
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{file_path}", "https://0x0.st"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
    return None


def upload_to_storage(file_path: str, prefix: str) -> tuple[str | None, str | None]:
    """Upload a file to temporary storage for cloud GPU job input.

    Tries R2 first, falls back to litterbox (24h/200MB), then 0x0.st (30d/512MB).

    Returns (url, r2_key) where r2_key is set only for R2 uploads (for cleanup).
    """
    file_size = Path(file_path).stat().st_size
    file_name = Path(file_path).name

    print(f"Uploading {file_name} ({file_size // 1024}KB)...", file=sys.stderr)

    url, r2_key = upload_to_r2(file_path, prefix)
    if url:
        print(f"  Upload complete (R2)", file=sys.stderr)
        return url, r2_key

    # Fall back to free services
    for service_name, upload_func in [("litterbox", _upload_to_litterbox), ("0x0.st", _upload_to_0x0)]:
        try:
            url = upload_func(file_path, file_name)
            if url:
                print(f"  Upload complete ({service_name})", file=sys.stderr)
                return url, None
        except Exception as e:
            print(f"  {service_name} failed: {e}", file=sys.stderr)
            continue

    print("All upload services failed", file=sys.stderr)
    return None, None


def download_from_url(url: str, output_path: str, verbose: bool = True) -> bool:
    """Download file from URL to local path with streaming."""
    import requests

    try:
        if verbose:
            print(f"Downloading result...", file=sys.stderr)

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            size_kb = Path(output_path).stat().st_size // 1024
            print(f"  Downloaded: {output_path} ({size_kb}KB)", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        return False


def get_r2_payload_config() -> dict | None:
    """Get R2 config dict for embedding in cloud GPU job payloads.

    Returns the dict to include as payload["input"]["r2"], or None if R2
    is not configured. This is the format expected by RunPod/Modal handlers.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if not r2_config:
        return None

    return {
        "endpoint_url": r2_config["endpoint_url"],
        "access_key_id": r2_config["access_key_id"],
        "secret_access_key": r2_config["secret_access_key"],
        "bucket_name": r2_config["bucket_name"],
    }
