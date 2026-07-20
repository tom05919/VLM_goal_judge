#!/usr/bin/env bash
# Recreate/update `perception` (stop judge + scan + ZMQ client).
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
STACK="$(_go2_stack_root)"
WS="$(_go2_workspace_root)"
YML="${SCRIPT_DIR}/environment-perception.yml"
REQ="${SCRIPT_DIR}/requirements-perception.txt"
UNIDEPTH="${STACK}/depth_implementation/UniDepth"
GROUNDED_SAM="${STACK}/segmentation_implementation/Grounded-SAM-2"
OMNIVLA="${WS}/omni-VLA/OmniVLA"
TORCH_INDEX="${GO2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
_go2_require_dir "$UNIDEPTH" "UniDepth checkout"
_go2_require_dir "$GROUNDED_SAM" "Grounded-SAM-2 checkout"
_go2_require_dir "$OMNIVLA" "OmniVLA checkout"

if [[ "$REMOVE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env remove -n perception -y || true
fi

if ! _go2_env_exists perception; then
  "${GO2_MAMBA[@]}" env create -f "$YML"
elif [[ "$UPDATE" -eq 1 ]]; then
  "${GO2_MAMBA[@]}" env update -n perception -f "$YML"
else
  echo "[perception] Env already exists. Pass --remove to recreate or --update to refresh."
  exit 0
fi

set +u
conda activate perception
set -u
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url "$TORCH_INDEX"
# RoboStack declares NumPy <2, while UniDepth requires >=2. A clean Python 3.11
# env was integration-tested with this controlled override, including rclpy
# init/node/shutdown and sensor/geometry message imports. The pinned pip
# requirements intentionally apply the final NumPy/SciPy versions.
python -m pip install -r "$REQ"
python -m pip install -e "$UNIDEPTH" --no-build-isolation --no-deps
SAM2_BUILD_CUDA=0 python -m pip install \
  -e "$GROUNDED_SAM" --no-build-isolation --no-deps

export PYTHONPATH="${STACK}/depth_implementation:${STACK}/segmentation_implementation:${OMNIVLA}/inference"
python -c "import numpy, scipy, torch, zmq, rclpy; from sensor_msgs.msg import Image; from geometry_msgs.msg import Twist; print('perception core imports OK')"
python -c "from run_unidepth_depth import DepthEstimator; from run_grounded_sam2 import GroundedSAM2Segmenter; print('UniDepth + SAM2 imports OK')"
python "${STACK}/stop_judge.py" --help >/dev/null
python "${STACK}/server_client.py" --help >/dev/null
python -m pip check

echo "[perception] Done. Download checkpoints/caches listed in envs/VERSIONS.md."
