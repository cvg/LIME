#!/usr/bin/env bash
# Download the released LIME weights from Hugging Face to weights/lime_v1, where
# scripts/predict.py and demo/viewer.py look by default.
#
#   bash scripts/download_weights.sh
#
# scripts/setup_env.sh runs it for you. It needs the `hf` command from huggingface_hub,
# which that environment has (pip install -U huggingface_hub otherwise). The weights are
# 8.9 GB. Run it again to resume an interrupted download.
set -euo pipefail

HF_REPO="BoboSun/LIME"     # the Hugging Face model repo

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/weights/lime_v1"

# A folder that didn't come from Hugging Face has no download metadata to resume from.
if [[ -f "$DEST/config.json" && ! -d "$DEST/.cache/huggingface" ]]; then
    echo "$DEST is already there. Delete it to download again."
    exit 0
fi
if command -v hf >/dev/null; then
    HF=hf
elif command -v huggingface-cli >/dev/null; then
    HF=huggingface-cli
else
    echo "This needs huggingface_hub: pip install -U huggingface_hub" >&2
    exit 1
fi

# Files that are already complete are skipped, so running this again resumes.
"$HF" download "$HF_REPO" --local-dir "$DEST"
echo "Done: $DEST"
