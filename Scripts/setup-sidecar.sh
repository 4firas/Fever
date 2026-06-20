#!/usr/bin/env bash
# Provisions the Python 3.12 sidecar venv (via uv) and downloads the pose model.
# Dev path used by `swift run`; bundle.sh stages an embedded copy for the .app.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/Sidecar/.venv"
MODEL_DIR="$ROOT/Models"
MODEL="$MODEL_DIR/pose_landmarker_full.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

echo "==> Creating venv (Python 3.12) at $VENV"
uv venv --python 3.12 "$VENV"
echo "==> Installing sidecar deps"
VIRTUAL_ENV="$VENV" uv pip install -r "$ROOT/Sidecar/requirements.txt"

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL" ]; then
  echo "==> Downloading pose_landmarker_full.task"
  curl -fL "$MODEL_URL" -o "$MODEL"
fi
echo "==> Done. Interpreter: $VENV/bin/python3  Model: $MODEL"
"$VENV/bin/python3" --version
