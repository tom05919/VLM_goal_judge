#!/usr/bin/env bash
# ONE-TIME bootstrap for the qwen-vlm conda env. Packages persist after this;
# do not re-run on every conda activate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN="${QWEN_ENV:-/root/miniforge3/envs/qwen-vlm}"
MAMBA="${MAMBA:-/root/miniforge3/bin/mamba}"

if [ ! -x "$QWEN/bin/python" ]; then
  echo "qwen-vlm env not found at $QWEN" >&2
  exit 1
fi

echo "Installing pip packages into $QWEN (isolated from PYTHONPATH)..."
env -i PATH="$QWEN/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  "$QWEN/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Installing ROS 2 packages into qwen-vlm via mamba..."
CONDA_NO_PLUGINS=true "$MAMBA" install -n qwen-vlm -y \
  -c conda-forge -c robostack-humble \
  ros-humble-rclpy ros-humble-geometry-msgs ros-humble-sensor-msgs

echo "Ensuring numpy 1.26.4 (avoids pip/conda metadata conflicts)..."
CONDA_NO_PLUGINS=true "$MAMBA" install -n qwen-vlm -y --force-reinstall numpy=1.26.4

echo "Verifying imports..."
env -i PATH="$QWEN/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  "$QWEN/bin/python" -c "import qwen_vl_utils, rclpy, transformers; print('qwen-vlm env OK')"

echo "Done. Activate with: conda activate qwen-vlm"
