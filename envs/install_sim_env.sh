#!/usr/bin/env bash
# Recreate/update `sim` (RoboStack Go2 driver + OmniVLA-edge).
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
YML="${SCRIPT_DIR}/environment-sim.yml"
REQ="${SCRIPT_DIR}/requirements-sim.txt"
OMNIVLA="${WS}/omni-VLA/OmniVLA"
TORCH_INDEX="${GO2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
_go2_require_dir "${OMNIVLA}" "OmniVLA checkout"

if [[ "$REMOVE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env remove -n sim -y || true
fi

if ! _go2_env_exists sim; then
  "${GO2_MAMBA[@]}" env create -f "$YML"
elif [[ "$UPDATE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env update -n sim -f "$YML"
else
  echo "[sim] Env already exists. Pass --remove to recreate or --update to refresh."
  exit 0
fi

set +u
conda activate sim
set -u
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url "$TORCH_INDEX"
python -m pip install -r "$REQ"

# Edge runs from the source tree. Installing OmniVLA metadata here would force
# its full-model torch==2.2 dependency and break the edge env.
cd "$OMNIVLA"
python -c "import numpy, torch, rclpy, clip, yaml, matplotlib, efficientnet_pytorch; assert torch.__version__.startswith('2.7.'), torch.__version__; print('sim imports OK')"
python inference/run_omnivla_edge.py --help >/dev/null

echo "[sim] Done. Download omnivla-edge weights and build the Go2 SDK (envs/VERSIONS.md)."
