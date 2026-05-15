#!/usr/bin/env bash
# Copy pdf2ppt-specific LaMa files into a local advimman/lama clone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMA_ROOT="${LAMA_REPO_ROOT:-$REPO_ROOT/lama}"
PATCH_ROOT="$REPO_ROOT/scripts/lama_patches"

if [[ ! -d "$LAMA_ROOT/.git" ]]; then
  echo "LaMa repo not found at $LAMA_ROOT." >&2
  echo "Clone it first, for example:" >&2
  echo "  git clone https://github.com/advimman/lama.git \"$LAMA_ROOT\"" >&2
  exit 1
fi

if [[ ! -d "$PATCH_ROOT/bin" ]]; then
  echo "Patch directory missing: $PATCH_ROOT/bin" >&2
  exit 1
fi

install -m 755 "$PATCH_ROOT/bin/pdf2ppt_predict_server.py" "$LAMA_ROOT/bin/pdf2ppt_predict_server.py"
install -m 755 "$PATCH_ROOT/bin/predict.py" "$LAMA_ROOT/bin/predict.py"

echo "Applied pdf2ppt LaMa patches to $LAMA_ROOT"
