#!/usr/bin/env bash
# Download MiniCPM-V-4.6-AWQ into models/ using minicpm-v46 env ONLY.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${MINICPM_ENV_NAME:-minicpm-v46}"
ENV_PREFIX="${MINICPM_ENV:-/root/miniforge3/envs/${ENV_NAME}}"
DEST="$SCRIPT_DIR/models/MiniCPM-V-4.6-AWQ"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Run ./install_minicpm_env.sh first (env ${ENV_NAME} not found)" >&2
  exit 1
fi

echo "Downloading openbmb/MiniCPM-V-4.6-AWQ to ${DEST}..."
env -i PATH="$ENV_PREFIX/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  "$ENV_PREFIX/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/MiniCPM-V-4.6-AWQ', local_dir='${DEST}')
"

echo "Verifying download..."
test -f "$DEST/model.safetensors"
test -f "$DEST/config.json"
test -f "$DEST/processor_config.json"
echo "Download complete: ${DEST}"
