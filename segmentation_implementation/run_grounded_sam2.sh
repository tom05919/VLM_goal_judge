#!/usr/bin/env bash
# Grounded-SAM-2 image segmentation via official HF demo script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSAM_ROOT="$SCRIPT_DIR/Grounded-SAM-2"
ENV_PREFIX="${PERCEPTION_ENV:-/root/miniforge3/envs/perception}"

IMG="${1:-$SCRIPT_DIR/../run_images/run3_no-stop/1_ex_omnivla_edge.jpg}"
TEXT_PROMPT="${2:-door.}"
OUTPUT_DIR="${3:-$GSAM_ROOT/outputs/grounded_sam2_hf_demo}"

CKPT="$GSAM_ROOT/checkpoints/sam2.1_hiera_small.pt"
if [ ! -f "$CKPT" ]; then
  echo "SAM2 checkpoint missing: $CKPT" >&2
  echo "Run: curl -L -o $CKPT https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" >&2
  exit 1
fi

cd "$GSAM_ROOT"
env -i PATH="$ENV_PREFIX/bin:/usr/bin:/bin" HOME="${HOME:-/root}" \
  HF_HUB_DISABLE_XET=1 \
  "$ENV_PREFIX/bin/python" grounded_sam2_hf_model_demo.py \
  --grounding-model IDEA-Research/grounding-dino-base \
  --sam2-checkpoint "$CKPT" \
  --sam2-model-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --img-path "$IMG" \
  --text-prompt "$TEXT_PROMPT" \
  --output-dir "$OUTPUT_DIR"

echo "Outputs written to $OUTPUT_DIR"
