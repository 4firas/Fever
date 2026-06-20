#!/usr/bin/env bash
# Stages a RELOCATABLE Python 3.12 (uv-managed python-build-standalone) + the
# sidecar deps into <dest>, for embedding in Fever.app/Contents/Resources.
# Usage: stage-embedded-python.sh <dest_sidecar_dir>
set -euo pipefail
DEST="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv python install 3.12 >/dev/null 2>&1 || true
PYSTD="$(uv python find 3.12)"                       # .../bin/python3.12
PYHOME="$(cd "$(dirname "$PYSTD")/.." && pwd)"       # standalone interpreter root

echo "==> staging relocatable python from $PYHOME"
rm -rf "$DEST/python"
mkdir -p "$DEST/python"
cp -R "$PYHOME/." "$DEST/python/"

# This is now our private, self-contained interpreter — drop uv's PEP-668
# "externally managed" marker so pip can install the sidecar deps into it.
find "$DEST/python" -name "EXTERNALLY-MANAGED" -delete

PY="$DEST/python/bin/python3"
[ -x "$PY" ] || PY="$DEST/python/bin/python3.12"
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade pip >/dev/null
echo "==> installing sidecar deps into the embedded interpreter"
"$PY" -m pip install -r "$ROOT/Sidecar/requirements.txt"

cp "$ROOT/Sidecar/pose_server.py" "$DEST/pose_server.py"

# Sanity: the embedded interpreter must import the stack.
"$PY" -c "import mediapipe, numpy; print('   embedded mediapipe', mediapipe.__version__, 'numpy', numpy.__version__)"
echo "==> staged sidecar -> $DEST"
