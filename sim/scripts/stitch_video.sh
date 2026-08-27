#!/bin/bash
# Stitch a directory of frame_XXXXX.png images into an mp4 with ffmpeg.
# Usage: ./stitch_video.sh <frames_dir> <output.mp4> [fps]
set -euo pipefail
FRAMES_DIR="${1:?frames dir required}"
OUTPUT="${2:?output mp4 path required}"
FPS="${3:-15}"

ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p "$OUTPUT"
echo "[RESULT] Saved video to $OUTPUT"
