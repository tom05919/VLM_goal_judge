#!/usr/bin/env bash
# Download MiniCPM-V-4.6 (BF16 weights, loaded as 8-bit via bitsandbytes) using minicpm-v46 env ONLY.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${MINICPM_ENV_NAME:-minicpm-v46}"
ENV_PREFIX="${MINICPM_ENV:-/root/miniforge3/envs/${ENV_NAME}}"
DEST="$SCRIPT_DIR/models/MiniCPM-V-4.6"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Run ./install_minicpm_env.sh first (env ${ENV_NAME} not found)" >&2
  exit 1
fi

echo "Downloading openbmb/MiniCPM-V-4.6 to ${DEST}..."
# HF XET can hang at 0 bytes on some networks; disable it for the large weight file.
rm -f "$DEST/.cache/huggingface/download/model.safetensors.lock" \
      "$DEST/.cache/huggingface/download/"*.incomplete 2>/dev/null || true

env -i PATH="$ENV_PREFIX/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 \
  "$ENV_PREFIX/bin/python" -u -c "
from huggingface_hub import hf_hub_download, snapshot_download
dest = '${DEST}'
# Small metadata files first (fast).
snapshot_download('openbmb/MiniCPM-V-4.6', local_dir=dest, ignore_patterns=['model.safetensors'])
# Large weights separately with resume support.
hf_hub_download('openbmb/MiniCPM-V-4.6', 'model.safetensors', local_dir=dest)
"

echo "Verifying download..."
test -f "$DEST/model.safetensors"
python3 -c "import os; s=os.path.getsize('${DEST}/model.safetensors'); assert s>2_000_000_000, s; print(f'model.safetensors: {s/1e9:.2f} GB')"
test -f "$DEST/config.json"
echo "Download complete: ${DEST}"
