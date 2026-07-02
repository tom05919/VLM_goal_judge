#!/usr/bin/env python3
"""Run UniDepthV2 metric depth on a single RGB image. Output is depth in meters."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "lpiccinelli/unidepth-v2-vits14"
DEFAULT_IMAGE = SCRIPT_DIR.parent / "run_images/run3_no-stop/1_ex_omnivla_edge.jpg"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


class DepthEstimator:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading UniDepthV2 ({model_name}) on {self.device}...", flush=True)
        self.model = UniDepthV2.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
        if type(self.model).__name__ != "UniDepthV2":
            raise RuntimeError(f"Expected UniDepthV2, got {type(self.model).__name__}")
        self.model.resolution_level = 1
        print("UniDepthV2 ready", flush=True)

    def estimate(self, image: Image.Image) -> torch.Tensor:
        """Return metric depth in meters, shape (H, W)."""
        rgb = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        with torch.inference_mode():
            depth = self.model.infer(rgb)["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(0).squeeze(0)
        elif depth.ndim == 3:
            depth = depth.squeeze(0)
        return depth


def main() -> None:
    parser = argparse.ArgumentParser(description="UniDepthV2 monocular metric depth")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    depth = DepthEstimator(args.model).estimate(Image.open(args.image))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    depth_path = args.output_dir / f"{stem}_depth_meters.pt"
    torch.save(depth.cpu(), depth_path)

    depth_np = depth.float().cpu().numpy()
    print(f"depth shape: {tuple(depth.shape)} dtype: {depth.dtype}")
    print(f"depth range (m): min={depth_np.min():.3f} max={depth_np.max():.3f} mean={depth_np.mean():.3f}")
    print(f"Saved raw depth tensor: {depth_path}")


if __name__ == "__main__":
    main()
