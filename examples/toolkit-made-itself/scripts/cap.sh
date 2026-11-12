#!/usr/bin/env bash
# Run a toolkit command from the toolkit root and keep its real output for the on-screen terminal.
#   scripts/cap.sh NAME uv run tools/voiceover.py ...
# (CAP_DIR=dir runs it from there instead, e.g. the project for npx remotion)
# writes capture/NAME.txt: first line "$ <command>", then stdout and stderr as printed.
set -o pipefail
name=$1; shift
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
out="$here/capture/$name.txt"
cd "${CAP_DIR:-$root}" || exit 1
start=$(date +%s)
{ python3 -c 'import shlex,sys; print("$ " + shlex.join(sys.argv[1:]))' "$@"; "$@" 2>&1; } | tee "$out"
code=${PIPESTATUS[0]}
printf '# elapsed %ss, exit %s\n' "$(( $(date +%s) - start ))" "$code" >> "$out"
exit "$code"
