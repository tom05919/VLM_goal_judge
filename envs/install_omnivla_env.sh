#!/usr/bin/env bash
# Recreate/update `omnivla` (full model local ROS and ZeroMQ serve).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_conda_lib.sh"

REMOVE=0
UPDATE=0
for arg in "$@"; do
  case "$arg" in
    --remove) REMOVE=1 ;;
    --update) UPDATE=1 ;;
    --with-ros) ;; # Backward-compatible no-op: ROS is now always installed.
    -h|--help)
      echo "Usage: $0 [--remove] [--update]"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

_go2_clear_inherited_env
_go2_init_conda
WS="$(_go2_workspace_root)"
YML="${SCRIPT_DIR}/environment-omnivla.yml"
REQ="${SCRIPT_DIR}/requirements-omnivla.txt"
OMNIVLA="${WS}/omni-VLA/OmniVLA"
TORCH_INDEX="${GO2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
_go2_require_dir "$OMNIVLA" "OmniVLA checkout"

if [[ "$REMOVE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env remove -n omnivla -y || true
fi

if ! _go2_env_exists omnivla; then
  "${GO2_MAMBA[@]}" env create -f "$YML"
elif [[ "$UPDATE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env update -n omnivla -f "$YML"
else
  echo "[omnivla] Env already exists. Pass --remove to recreate or --update to refresh."
  exit 0
fi

set +u
conda activate omnivla
set -u
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

"${GO2_MAMBA[@]}" install --force-reinstall -y -c conda-forge \
  numpy=1.26.4 scipy=1.11.4
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url "$TORCH_INDEX"
python -m pip install -r "$REQ"
python -m pip install -e "$OMNIVLA"
# Reassert pins that transitive TensorFlow metadata otherwise upgrades.
python -m pip install -r "$REQ"

cd "$OMNIVLA"
python -c "import numpy, torch, zmq, rclpy, tensorflow_datasets; assert torch.__version__.startswith('2.2.'), torch.__version__; print('omnivla imports OK')"
python inference/run_omnivla.py --help >/dev/null
python -m pip check

echo "[omnivla] Done. This env supports both --mode local and --mode serve."
