#!/usr/bin/env python3
"""
Video Generation Pipeline - Interactive Orchestration

Human-in-the-loop workflow for generating video segments with AI.
Each step requires user validation before proceeding.

Workflow:
  1. Load/create project with scene definitions
  2. For each scene:
     a. Edit frame (Qwen-Edit) → USER REVIEWS → approve/retry/adjust
     b. Generate video (Wan I2V) → USER REVIEWS → approve/retry/adjust
     c. Extract last frame for next scene
  3. Stitch approved segments

Usage:
  python tools/videogen.py new --name "my-video" --reference image.png
  python tools/videogen.py edit --project my-video --scene 1
  python tools/videogen.py animate --project my-video --scene 1
  python tools/videogen.py status --project my-video
  python tools/videogen.py stitch --project my-video
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Load environment
from dotenv import load_dotenv
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
QWEN_EDIT_ENDPOINT = os.getenv("RUNPOD_QWEN_EDIT_ENDPOINT_ID")
WAN_I2V_ENDPOINT = os.getenv("RUNPOD_WAN_I2V_ENDPOINT_ID")

PROJECTS_DIR = Path("projects/videogen")


def log(msg: str, level: str = "info"):
    """Print formatted log message."""
    prefix = {"info": "→", "success": "✓", "error": "✗", "warn": "⚠", "prompt": "?"}
    print(f"{prefix.get(level, '→')} {msg}")


def encode_image(path: str) -> str:
    """Encode image file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_and_save(base64_data: str, output_path: str):
    """Decode base64 and save to file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))
    log(f"Saved: {output_path}", "success")


def open_file(path: str):
    """Open file with system default application."""
    if sys.platform == "darwin":
        subprocess.run(["open", path])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", path])
    else:
        log(f"Please open manually: {path}", "warn")


def call_runpod(endpoint_id: str, payload: dict, timeout: int = 300) -> dict:
    """Call RunPod serverless endpoint and wait for result."""
    if not RUNPOD_API_KEY:
        log("RUNPOD_API_KEY not set in .env", "error")
        sys.exit(1)

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    log(f"Calling RunPod endpoint {endpoint_id}...")
    start = time.time()

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        elapsed = time.time() - start
        log(f"Response received in {elapsed:.1f}s", "success")

        if result.get("status") == "FAILED":
            log(f"Job failed: {result.get('error', 'Unknown error')}", "error")
            return {"error": result.get("error", "Unknown error")}

        return result.get("output", result)

    except requests.exceptions.Timeout:
        log(f"Request timed out after {timeout}s", "error")
        return {"error": "timeout"}
    except Exception as e:
        log(f"Request failed: {e}", "error")
        return {"error": str(e)}


def prompt_user(message: str, options: list[str] = None) -> str:
    """Prompt user for input with optional choices."""
    if options:
        print(f"\n{message}")
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        while True:
            try:
                choice = input("Choice: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    return options[int(choice) - 1]
                elif choice.lower() in [o.lower() for o in options]:
                    return choice
            except (ValueError, IndexError):
                pass
            print("Invalid choice, try again.")
    else:
        return input(f"{message}: ").strip()


def confirm(message: str, default: bool = True) -> bool:
    """Ask user for yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{message} {suffix}: ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


# =============================================================================
# Project Management
# =============================================================================

def load_project(name: str) -> dict:
    """Load project state from JSON."""
    project_dir = PROJECTS_DIR / name
    project_file = project_dir / "project.json"

    if not project_file.exists():
        log(f"Project not found: {name}", "error")
        sys.exit(1)

    with open(project_file) as f:
        return json.load(f)


def save_project(project: dict):
    """Save project state to JSON."""
    project_dir = PROJECTS_DIR / project["name"]
    project_file = project_dir / "project.json"

    project["updated_at"] = datetime.now().isoformat()

    with open(project_file, "w") as f:
        json.dump(project, f, indent=2)


def cmd_new(args):
    """Create a new video generation project."""
    name = args.name or prompt_user("Project name")
    reference = args.reference

    if not reference:
        reference = prompt_user("Path to reference image")

    if not Path(reference).exists():
        log(f"Reference image not found: {reference}", "error")
        sys.exit(1)

    # Create project directory
    project_dir = PROJECTS_DIR / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "frames").mkdir(exist_ok=True)
    (project_dir / "videos").mkdir(exist_ok=True)

    # Copy reference image
    import shutil
    ref_dest = project_dir / "reference.png"
    shutil.copy(reference, ref_dest)
    log(f"Copied reference to {ref_dest}", "success")

    # Initialize project structure
    project = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "reference_image": str(ref_dest),
        "scenes": [],
        "current_scene": 0,
        "status": "planning"
    }

    # Interactive scene planning
    log("\n📋 Scene Planning", "info")
    log("Define scenes for your video. Each scene has:")
    log("  - edit_prompt: How to modify the frame (for Qwen-Edit)")
    log("  - motion_prompt: What motion to generate (for Wan I2V)")
    print()

    while True:
        scene_num = len(project["scenes"]) + 1
        log(f"Scene {scene_num}:", "prompt")

        edit_prompt = prompt_user("  Edit prompt (what to change in frame)")
        if not edit_prompt:
            break

        motion_prompt = prompt_user("  Motion prompt (what movement to generate)")
        if not motion_prompt:
            motion_prompt = "Subtle natural movement"

        scene = {
            "id": scene_num,
            "edit_prompt": edit_prompt,
            "motion_prompt": motion_prompt,
            "status": "pending",
            "edit_attempts": [],
            "video_attempts": [],
            "approved_frame": None,
            "approved_video": None
        }
        project["scenes"].append(scene)

        if not confirm("Add another scene?"):
            break

    save_project(project)
    log(f"\nProject created: {project_dir}", "success")
    log(f"Scenes defined: {len(project['scenes'])}")
    log(f"\nNext: python tools/videogen.py edit --project {name} --scene 1")


def cmd_status(args):
    """Show project status."""
    project = load_project(args.project)

    print(f"\n📊 Project: {project['name']}")
    print(f"   Status: {project['status']}")
    print(f"   Reference: {project['reference_image']}")
    print(f"   Scenes: {len(project['scenes'])}")
    print()

    for scene in project["scenes"]:
        status_icon = {
            "pending": "⬜",
            "editing": "🔄",
            "edited": "📝",
            "animating": "🔄",
            "animated": "🎬",
            "approved": "✅",
            "failed": "❌"
        }.get(scene["status"], "❓")

        print(f"   Scene {scene['id']}: {status_icon} {scene['status']}")
        print(f"      Edit: {scene['edit_prompt'][:50]}...")
        print(f"      Motion: {scene['motion_prompt'][:50]}...")
        if scene.get("approved_frame"):
            print(f"      Frame: {scene['approved_frame']}")
        if scene.get("approved_video"):
            print(f"      Video: {scene['approved_video']}")
        print()


# =============================================================================
# Edit Step (Qwen-Edit)
# =============================================================================

def cmd_edit(args):
    """Run edit step for a scene with user validation."""
    if not QWEN_EDIT_ENDPOINT:
        log("RUNPOD_QWEN_EDIT_ENDPOINT_ID not set in .env", "error")
        sys.exit(1)

    project = load_project(args.project)
    scene_id = args.scene
    scene = next((s for s in project["scenes"] if s["id"] == scene_id), None)

    if not scene:
        log(f"Scene {scene_id} not found", "error")
        sys.exit(1)

    project_dir = PROJECTS_DIR / project["name"]

    # Determine input frame
    if scene_id == 1:
        input_frame = project["reference_image"]
    else:
        prev_scene = project["scenes"][scene_id - 2]
        if not prev_scene.get("approved_frame"):
            log(f"Scene {scene_id - 1} frame not approved yet", "error")
            sys.exit(1)
        input_frame = prev_scene["approved_frame"]

    log(f"\n🖼️  Edit Scene {scene_id}", "info")
    log(f"Input: {input_frame}")
    log(f"Prompt: {scene['edit_prompt']}")

    # Allow prompt adjustment
    if args.prompt:
        scene["edit_prompt"] = args.prompt
        log(f"Using custom prompt: {args.prompt}")

    if confirm("\nProceed with edit?"):
        scene["status"] = "editing"
        save_project(project)

        # Call Qwen-Edit
        attempt = len(scene["edit_attempts"]) + 1
        payload = {
            "input": {
                "image_base64": encode_image(input_frame),
                "prompt": scene["edit_prompt"],
                "seed": args.seed if args.seed else None
            }
        }

        result = call_runpod(QWEN_EDIT_ENDPOINT, payload)

        if "error" in result:
            log(f"Edit failed: {result['error']}", "error")
            scene["status"] = "failed"
            save_project(project)
            return

        # Save result
        output_path = project_dir / "frames" / f"scene{scene_id}_edit_v{attempt}.png"
        decode_and_save(result["edited_image_base64"], str(output_path))

        scene["edit_attempts"].append({
            "attempt": attempt,
            "path": str(output_path),
            "seed": result.get("seed"),
            "inference_time_ms": result.get("inference_time_ms"),
            "timestamp": datetime.now().isoformat()
        })
        save_project(project)

        # Show result for review
        log("\n📋 Review the edited frame:", "prompt")
        open_file(str(output_path))

        print("\nOptions:")
        choice = prompt_user("What would you like to do?", [
            "approve - Use this frame",
            "retry - Generate again (same prompt)",
            "adjust - Modify prompt and retry",
            "skip - Move on without approving"
        ])

        if choice.startswith("approve"):
            scene["approved_frame"] = str(output_path)
            scene["status"] = "edited"
            log("Frame approved!", "success")
        elif choice.startswith("retry"):
            log("Retrying... run the edit command again")
        elif choice.startswith("adjust"):
            new_prompt = prompt_user("New edit prompt")
            scene["edit_prompt"] = new_prompt
            log("Prompt updated. Run edit command again.")
        else:
            log("Skipped without approval")

        save_project(project)

        if scene["status"] == "edited":
            log(f"\nNext: python tools/videogen.py animate --project {project['name']} --scene {scene_id}")


# =============================================================================
# Animate Step (Wan I2V)
# =============================================================================

def cmd_animate(args):
    """Run animation step for a scene with user validation."""
    if not WAN_I2V_ENDPOINT:
        log("RUNPOD_WAN_I2V_ENDPOINT_ID not set in .env", "error")
        sys.exit(1)

    project = load_project(args.project)
    scene_id = args.scene
    scene = next((s for s in project["scenes"] if s["id"] == scene_id), None)

    if not scene:
        log(f"Scene {scene_id} not found", "error")
        sys.exit(1)

    if not scene.get("approved_frame"):
        log(f"Scene {scene_id} has no approved frame. Run edit first.", "error")
        sys.exit(1)

    project_dir = PROJECTS_DIR / project["name"]

    log(f"\n🎬 Animate Scene {scene_id}", "info")
    log(f"Frame: {scene['approved_frame']}")
    log(f"Motion: {scene['motion_prompt']}")

    # Allow prompt adjustment
    if args.prompt:
        scene["motion_prompt"] = args.prompt
        log(f"Using custom prompt: {args.prompt}")

    # Allow frame count adjustment
    num_frames = args.frames or 81  # Default ~5 seconds

    if confirm(f"\nGenerate {num_frames} frames (~{num_frames/16:.1f}s)?"):
        scene["status"] = "animating"
        save_project(project)

        # Call Wan I2V
        attempt = len(scene["video_attempts"]) + 1
        payload = {
            "input": {
                "image_base64": encode_image(scene["approved_frame"]),
                "prompt": scene["motion_prompt"],
                "num_frames": num_frames,
                "seed": args.seed if args.seed else None
            }
        }

        result = call_runpod(WAN_I2V_ENDPOINT, payload, timeout=600)

        if "error" in result:
            log(f"Animation failed: {result['error']}", "error")
            scene["status"] = "edited"  # Revert to edited state
            save_project(project)
            return

        # Save video
        video_path = project_dir / "videos" / f"scene{scene_id}_v{attempt}.mp4"
        decode_and_save(result["video_base64"], str(video_path))

        # Save last frame for chaining
        last_frame_path = project_dir / "frames" / f"scene{scene_id}_lastframe_v{attempt}.png"
        if result.get("last_frame_base64"):
            decode_and_save(result["last_frame_base64"], str(last_frame_path))

        scene["video_attempts"].append({
            "attempt": attempt,
            "video_path": str(video_path),
            "last_frame_path": str(last_frame_path),
            "seed": result.get("seed"),
            "inference_time_ms": result.get("inference_time_ms"),
            "num_frames": num_frames,
            "timestamp": datetime.now().isoformat()
        })
        save_project(project)

        # Show result for review
        log("\n📋 Review the generated video:", "prompt")
        open_file(str(video_path))

        print("\nOptions:")
        choice = prompt_user("What would you like to do?", [
            "approve - Use this video",
            "retry - Generate again (same prompt)",
            "adjust - Modify motion prompt and retry",
            "skip - Move on without approving"
        ])

        if choice.startswith("approve"):
            scene["approved_video"] = str(video_path)
            scene["approved_last_frame"] = str(last_frame_path)
            scene["status"] = "approved"
            log("Video approved!", "success")
        elif choice.startswith("retry"):
            log("Retrying... run the animate command again")
        elif choice.startswith("adjust"):
            new_prompt = prompt_user("New motion prompt")
            scene["motion_prompt"] = new_prompt
            log("Prompt updated. Run animate command again.")
        else:
            log("Skipped without approval")

        save_project(project)

        # Check if there's a next scene
        if scene["status"] == "approved" and scene_id < len(project["scenes"]):
            next_scene = project["scenes"][scene_id]
            # Use last frame as input for next scene's edit
            next_scene["input_frame"] = str(last_frame_path)
            save_project(project)
            log(f"\nNext: python tools/videogen.py edit --project {project['name']} --scene {scene_id + 1}")
        elif scene["status"] == "approved":
            log(f"\nAll scenes complete! Run: python tools/videogen.py stitch --project {project['name']}")


# =============================================================================
# Stitch Step
# =============================================================================

def cmd_stitch(args):
    """Stitch approved videos into final output."""
    project = load_project(args.project)
    project_dir = PROJECTS_DIR / project["name"]

    # Check all scenes are approved
    approved_videos = []
    for scene in project["scenes"]:
        if scene["status"] != "approved":
            log(f"Scene {scene['id']} not approved (status: {scene['status']})", "error")
            sys.exit(1)
        approved_videos.append(scene["approved_video"])

    log(f"\n🎬 Stitching {len(approved_videos)} videos", "info")

    # Create concat file for ffmpeg
    concat_file = project_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for video in approved_videos:
            f.write(f"file '{video}'\n")

    # Output path
    output_path = project_dir / f"{project['name']}_final.mp4"

    # Run ffmpeg concat
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path)
    ]

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"FFmpeg failed: {result.stderr}", "error")
        sys.exit(1)

    log(f"\n✅ Final video: {output_path}", "success")
    open_file(str(output_path))

    project["status"] = "complete"
    project["final_video"] = str(output_path)
    save_project(project)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video Generation Pipeline - Interactive Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s new --name demo --reference portrait.png
  %(prog)s status --project demo
  %(prog)s edit --project demo --scene 1
  %(prog)s animate --project demo --scene 1
  %(prog)s stitch --project demo
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # new
    p_new = subparsers.add_parser("new", help="Create new project")
    p_new.add_argument("--name", help="Project name")
    p_new.add_argument("--reference", help="Path to reference image")

    # status
    p_status = subparsers.add_parser("status", help="Show project status")
    p_status.add_argument("--project", "-p", required=True, help="Project name")

    # edit
    p_edit = subparsers.add_parser("edit", help="Run edit step for a scene")
    p_edit.add_argument("--project", "-p", required=True, help="Project name")
    p_edit.add_argument("--scene", "-s", type=int, required=True, help="Scene number")
    p_edit.add_argument("--prompt", help="Override edit prompt")
    p_edit.add_argument("--seed", type=int, help="Random seed")

    # animate
    p_animate = subparsers.add_parser("animate", help="Run animation step for a scene")
    p_animate.add_argument("--project", "-p", required=True, help="Project name")
    p_animate.add_argument("--scene", "-s", type=int, required=True, help="Scene number")
    p_animate.add_argument("--prompt", help="Override motion prompt")
    p_animate.add_argument("--frames", type=int, help="Number of frames (default: 81)")
    p_animate.add_argument("--seed", type=int, help="Random seed")

    # stitch
    p_stitch = subparsers.add_parser("stitch", help="Stitch approved videos")
    p_stitch.add_argument("--project", "-p", required=True, help="Project name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command
    commands = {
        "new": cmd_new,
        "status": cmd_status,
        "edit": cmd_edit,
        "animate": cmd_animate,
        "stitch": cmd_stitch,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
