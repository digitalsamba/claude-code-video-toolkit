#!/usr/bin/env python3
"""One-shot: update the RunPod Qwen3-TTS template to a new image tag.

Usage:
    python3 scratch/update_qwen3_template.py <new-image-ref>

Example:
    python3 scratch/update_qwen3_template.py \
        ghcr.io/conalmullan/video-toolkit-qwen3-tts:0.2.0-voicedesign
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from dotenv import load_dotenv
from qwen3_tts import (
    QWEN3_TTS_TEMPLATE_NAME,
    list_runpod_templates,
    runpod_graphql_query,
)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set")

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    new_image = sys.argv[1]

    # Find the existing Qwen3-TTS template by name (safer than matching imageName
    # since we're the ones changing it).
    templates = list_runpod_templates(api_key)
    match = next(
        (t for t in templates if t.get("name") == QWEN3_TTS_TEMPLATE_NAME),
        None,
    )
    if not match:
        sys.exit(
            f"No template named '{QWEN3_TTS_TEMPLATE_NAME}' found.\n"
            f"Existing templates: {[t.get('name') for t in templates]}"
        )

    tpl_id = match["id"]
    prev_image = match.get("imageName", "?")
    print(f"Template: {QWEN3_TTS_TEMPLATE_NAME} (id={tpl_id})")
    print(f"  Current imageName: {prev_image}")
    print(f"  New imageName:     {new_image}")

    # saveTemplate is the upsert mutation — passing id updates in place.
    mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
            isServerless
        }
    }
    """
    variables = {
        "input": {
            "id": tpl_id,
            "name": QWEN3_TTS_TEMPLATE_NAME,
            "imageName": new_image,
            "isServerless": True,
            "containerDiskInGb": 30,
            "volumeInGb": 0,
            "dockerArgs": "",
            "env": [],
        }
    }

    data = runpod_graphql_query(api_key, mutation, variables)
    result = data.get("saveTemplate")
    if not result or result.get("imageName") != new_image:
        sys.exit(f"Update failed: {json.dumps(data, indent=2)}")

    print(f"  ✓ Template updated.")
    print(
        f"\nNext cold-start on endpoint {os.getenv('RUNPOD_QWEN3_TTS_ENDPOINT_ID')} "
        f"will pull the new image."
    )


if __name__ == "__main__":
    main()
