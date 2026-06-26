#!/usr/bin/env bash
# ONE-TIME bootstrap for the minicpm-v46 conda env ONLY.
# Does NOT modify qwen-vlm or any other environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${MINICPM_ENV_NAME:-minicpm-v46}"
CONDA="${CONDA:-/root/miniforge3/bin/conda}"
MAMBA="${MAMBA:-/root/miniforge3/bin/mamba}"
ENV_PREFIX="${MINICPM_ENV:-/root/miniforge3/envs/${ENV_NAME}}"

if [ ! -x "$CONDA" ]; then
  echo "conda not found at $CONDA" >&2
  exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
  echo "Creating conda env ${ENV_NAME} at ${ENV_PREFIX}..."
  CONDA_NO_PLUGINS=true "$MAMBA" create -n "$ENV_NAME" python=3.10 -y
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "minicpm env not found at $ENV_PREFIX" >&2
  exit 1
fi

echo "Installing pip packages into ${ENV_NAME} ONLY (isolated from PYTHONPATH)..."
env -i PATH="$ENV_PREFIX/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  "$ENV_PREFIX/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-minicpm-v46.txt"

echo "Ensuring numpy 1.26.4 in ${ENV_NAME}..."
CONDA_NO_PLUGINS=true "$MAMBA" install -n "$ENV_NAME" -y --force-reinstall numpy=1.26.4

echo "Verifying imports in ${ENV_NAME}..."
env -i PATH="$ENV_PREFIX/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  "$ENV_PREFIX/bin/python" -c "
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
print(f'minicpm-v46 env OK (transformers {transformers.__version__})')
"

echo "Done. Activate with: conda activate ${ENV_NAME}"
