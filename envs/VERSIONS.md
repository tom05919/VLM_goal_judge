# Go2 stack source and weight revisions

These manifests are tracked with `goal_stop_judge`. Check out the stack release
commit containing this file; do not copy `envs/` outside the repository.

## Workspace layout

Run these commands from the directory that will contain the three project trees:

```bash
git clone https://github.com/tom05919/VLM_goal_judge.git goal_stop_judge

mkdir -p omni-VLA
git clone https://github.com/tom05919/Omni-VLA_Go2.git omni-VLA/OmniVLA
# Baseline before the current repair work:
git -C omni-VLA/OmniVLA checkout 990bc6ffb3d16295bb79b0e77c80879b6064a2d0

mkdir -p unofficial_sdk_unitree_go_2/src
git clone https://github.com/abizovnuralem/go2_ros2_sdk.git \
  unofficial_sdk_unitree_go_2/src/src
git -C unofficial_sdk_unitree_go_2/src/src \
  checkout 4e186b5f89bfec1f32c85676cbe22d4958e4f0fa

git clone https://github.com/lpiccinelli-eth/UniDepth.git \
  goal_stop_judge/depth_implementation/UniDepth
git -C goal_stop_judge/depth_implementation/UniDepth \
  checkout 8d8cfe4c7ee15297099983607febf0d4f32eb3d6

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git \
  goal_stop_judge/segmentation_implementation/Grounded-SAM-2
git -C goal_stop_judge/segmentation_implementation/Grounded-SAM-2 \
  checkout b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
```

The OmniVLA SHA above is a baseline. After the local repair changes are
committed, replace it with that commit before publishing the setup guide.

## Model weights

```bash
git clone https://huggingface.co/NHirose/omnivla-edge \
  omni-VLA/OmniVLA/omnivla-edge
git -C omni-VLA/OmniVLA/omnivla-edge \
  checkout b1361b7e24f101edea795a98b00a826b61a97394

git clone https://huggingface.co/NHirose/omnivla-original \
  omni-VLA/OmniVLA/omnivla-original
git -C omni-VLA/OmniVLA/omnivla-original \
  checkout e36a84d4923c041149d441f93f3bdb7092bb5f07

git clone https://huggingface.co/NHirose/omnivla-finetuned-cast \
  omni-VLA/OmniVLA/omnivla-finetuned-cast
git -C omni-VLA/OmniVLA/omnivla-finetuned-cast \
  checkout 7d3744a72cd89218be4d223783f0742819b6c6db
```

- Original full model: `InferenceConfig.vla_path="./omnivla-original"`,
  `resume_step=120000`.
- CAST model: `vla_path="./omnivla-finetuned-cast"`,
  `resume_step=210000`.
- SAM2 checkpoint:
  `goal_stop_judge/segmentation_implementation/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt`
  from
  `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt`.
- Grounding DINO cache: `IDEA-Research/grounding-dino-base`.
- UniDepth cache: `lpiccinelli/unidepth-v2-vits14`.

## Environments

- `sim`: Python 3.11.15, RoboStack, edge torch 2.7/cu126.
- `perception`: Python 3.11.15, RoboStack, torch 2.6/cu124, NumPy 2.2.
- `omnivla`: Python 3.11.15, RoboStack, full torch 2.2/cu121, NumPy 1.26.

Override the CUDA wheel source with `GO2_TORCH_INDEX_URL`. Override conda base
with `GO2_CONDA_BASE`. Never source `/opt/ros` with these RoboStack envs.
