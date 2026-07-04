#!/usr/bin/env bash
# Download TTS model weights into ./models/ (git-ignored). Weights are NOT
# shipped in this repo — this script fetches them from their upstream homes.
#
# Usage:
#   scripts/download_models.sh [piper|kokoro|supertonic|all]
#
# Requires: curl, and for Supertonic/Kokoro either `huggingface-cli` or git-lfs.
# Review each model's license before use (see NOTICE). The Supertonic weights
# are BigScience OpenRAIL-M (use-based restrictions).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${TTS_MODELS_DIR:-$REPO/models}"
WHAT="${1:-all}"
mkdir -p "$MODELS"

dl() { # url dest
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then echo "  ✓ exists: ${dest#$MODELS/}"; return; fi
  echo "  ↓ $url"
  curl -fL --retry 3 -o "$dest" "$url"
}

piper() {
  echo "== Piper voices =="
  mkdir -p "$MODELS/piper"
  local base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
  # lessac-medium (the fast "average" voice) + ryan-high (quality) used by the bench
  dl "$base/lessac/medium/en_US-lessac-medium.onnx"       "$MODELS/piper/en_US-lessac-medium.onnx"
  dl "$base/lessac/medium/en_US-lessac-medium.onnx.json"  "$MODELS/piper/en_US-lessac-medium.onnx.json"
  dl "$base/ryan/high/en_US-ryan-high.onnx"               "$MODELS/piper/en_US-ryan-high.onnx"
  dl "$base/ryan/high/en_US-ryan-high.onnx.json"          "$MODELS/piper/en_US-ryan-high.onnx.json"
}

kokoro() {
  echo "== Kokoro v1.0 (fp32 + int8 + voices) =="
  local base="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
  dl "$base/kokoro-v1.0.onnx"       "$MODELS/kokoro-v1.0.onnx"
  dl "$base/kokoro-v1.0.int8.onnx"  "$MODELS/kokoro-v1.0.int8.onnx"
  dl "$base/voices-v1.0.bin"        "$MODELS/voices-v1.0.bin"
}

supertonic() {
  echo "== Supertonic-3 (OpenRAIL-M — review license!) =="
  echo "   Requires huggingface-cli. Layout: models/supertonic/{onnx,voice_styles}"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "   ! huggingface-cli not found. Install: pip install huggingface_hub" >&2
    echo "   Then: huggingface-cli download Supertone/supertonic-3 --local-dir $MODELS/supertonic" >&2
    return 1
  fi
  huggingface-cli download Supertone/supertonic-3 --local-dir "$MODELS/supertonic"
}

pocket() {
  echo "== Pocket-TTS ONNX (export from Kyutai weights) =="
  echo "   Pocket has no ready ONNX release. Export with the vendored script:"
  echo "     pip install torch onnx onnxruntime safetensors"
  echo "     python benches/pocket-cpp/export_onnx.py --output-dir $MODELS/pocket"
  echo "   (needs the Kyutai pocket-tts weights; see benches/pocket-cpp/README.md)"
}

case "$WHAT" in
  piper) piper ;;
  kokoro) kokoro ;;
  supertonic) supertonic ;;
  pocket) pocket ;;
  all) piper; kokoro; echo "(supertonic + pocket skipped in 'all' — run explicitly)";;
  *) echo "usage: $0 [piper|kokoro|supertonic|pocket|all]"; exit 1 ;;
esac
echo "Done. Models in: $MODELS"
