#!/usr/bin/env python3
"""Orchestrate depth + segmentation perception for the stop judge."""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OMNIVLA_INFERENCE = ROOT.parent / "omni-VLA" / "inference"
sys.path.insert(0, str(ROOT / "depth_implementation"))
sys.path.insert(0, str(ROOT / "segmentation_implementation"))
sys.path.insert(0, str(OMNIVLA_INFERENCE))

from center_target import calculate_offset
from run_grounded_sam2 import GroundedSAM2Segmenter, SegmentationResult
from run_unidepth_depth import DepthEstimator
from stop_signal import (
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_STOP_DISTANCE_M,
    DEFAULT_STOP_SIGNAL_PATH,
    clear_stop,
    trigger_stop,
    write_target_distance,
)

DEFAULT_IMAGE = ROOT / "current_img.jpg"


@dataclass
class PerceptionOutput:
    depth_meters: torch.Tensor  # (H, W)
    segmentation: SegmentationResult
    image: Image.Image


@dataclass
class StopDecision:
    should_stop: bool
    distance_m: float | None
    detected: bool


def process_perception(
    perception: PerceptionOutput, stop_distance_m: float = DEFAULT_STOP_DISTANCE_M
) -> StopDecision:
    if not perception.segmentation.scores:
        return StopDecision(should_stop=False, distance_m=None, detected=False)

    depth_map = perception.depth_meters.squeeze().detach().cpu().numpy()
    best_i = int(np.argmax(perception.segmentation.scores))
    target_mask = perception.segmentation.masks[best_i]
    if target_mask.shape != depth_map.shape:
        target_mask = cv2.resize(
            target_mask.astype(np.uint8),
            (depth_map.shape[1], depth_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    target_depth_values = depth_map[target_mask]
    target_depth_values = target_depth_values[
        np.isfinite(target_depth_values) & (target_depth_values > 0)
    ]
    if target_depth_values.size == 0:
        return StopDecision(should_stop=False, distance_m=None, detected=True)

    distance_m = float(np.median(target_depth_values))
    should_stop = distance_m < stop_distance_m
    print(f"median distance to target: {distance_m:.3f} m (stop if < {stop_distance_m} m)")
    return StopDecision(
        should_stop=should_stop,
        distance_m=distance_m,
        detected=True,
    )


def _overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45
) -> np.ndarray:
    out = rgb.astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return out.astype(np.uint8)


def save_perception_visualizations(
    perception: PerceptionOutput,
    stem: str = "frame",
    output_dir: Path | None = None,
) -> list[Path]:
    output_dir = output_dir or ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_map = perception.depth_meters.squeeze().detach().cpu().numpy()
    masks = perception.segmentation.masks
    saved: list[Path] = []

    valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid.size:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    depth_norm = np.clip((depth_map - vmin) / (vmax - vmin + 1e-6), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
    depth_path = output_dir / f"{stem}_depth.jpg"
    cv2.imwrite(str(depth_path), depth_color_bgr)
    saved.append(depth_path)

    rgb = np.array(perception.image.convert("RGB"))
    mask_colors = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 64)]

    for i, mask in enumerate(masks):
        mask_path = output_dir / f"{stem}_mask_{i}.jpg"
        color = mask_colors[i % len(mask_colors)]
        Image.fromarray(_overlay_mask(rgb, mask, color)).save(mask_path, quality=95)
        saved.append(mask_path)

    if len(masks):
        union = masks.any(axis=0)
    else:
        union = np.zeros(depth_map.shape, dtype=bool)
    if union.shape != depth_rgb.shape[:2]:
        union = cv2.resize(
            union.astype(np.uint8),
            (depth_rgb.shape[1], depth_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    union_path = output_dir / f"{stem}_masks_union.jpg"
    union_vis = np.zeros_like(depth_rgb)
    union_vis[union] = depth_rgb[union]
    Image.fromarray(union_vis).save(union_path, quality=95)
    saved.append(union_path)

    return saved


def next_run_dir(base: Path | None = None) -> Path:
    """Create run1, run2, ... under run_images2 (or the given base)."""
    base = base or (ROOT / "run_images2")
    base.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^run(\d+)$")
    nums = [
        int(m.group(1))
        for p in base.iterdir()
        if p.is_dir() and (m := pattern.match(p.name))
    ]
    run_dir = base / f"run{max(nums, default=0) + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class StopJudge:
    def __init__(self, text_prompt: str, load_depth: bool = True):
        self.text_prompt = text_prompt
        self.depth_estimator = None

        print("loading Grounded SAM2...", flush=True)
        self.segmenter = GroundedSAM2Segmenter()
        print("Grounded SAM2 ready", flush=True)

        if load_depth:
            self.ensure_depth_ready()

    def ensure_depth_ready(self) -> None:
        if self.depth_estimator is not None:
            return
        print("loading UniDepthV2...", flush=True)
        self.depth_estimator = DepthEstimator()
        print("perception models ready", flush=True)

    def run(self, image: Image.Image) -> PerceptionOutput:
        self.ensure_depth_ready()
        return PerceptionOutput(
            depth_meters=self.depth_estimator.estimate(image),
            segmentation=self.segmenter.segment(image, self.text_prompt),
            image=image,
        )

    def decide(
        self, image: Image.Image, stop_distance_m: float = DEFAULT_STOP_DISTANCE_M
    ) -> tuple[StopDecision, PerceptionOutput]:
        perception = self.run(image)
        decision = process_perception(perception, stop_distance_m)
        return decision, perception


def run_live_loop(
    judge: StopJudge,
    sim: bool,
    stop_distance_m: float,
    min_interval_s: float,
    stop_signal_path: Path,
    output_dir: Path | None = None,
    save_viz: bool = False,
    cmd_vel_topic: str | None = None,
) -> None:
    import rclpy
    from isaacsim_controller import IsaacSimPublisher

    rclpy.init()
    node = IsaacSimPublisher(sim=sim, cmd_vel_topic=cmd_vel_topic)
    print("LIVE_JUDGE_READY", flush=True)
    frame_idx = 0
    try:
        while True:
            loop_start = time.time()
            image = node.get_latest_image_pil(fresh=True)
            if image is None:
                print("No camera frame received; retrying...")
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, min_interval_s - elapsed))
                continue

            decision, perception = judge.decide(image, stop_distance_m)
            if decision.distance_m is not None:
                write_target_distance(decision.distance_m, stop_signal_path.parent / ".target_distance")
            if save_viz and output_dir is not None:
                save_perception_visualizations(perception, stem=f"live_{frame_idx}", output_dir=output_dir)

            if decision.should_stop:
                print(
                    f"[STOP] Target at {decision.distance_m:.3f} m "
                    f"(threshold {stop_distance_m} m) - halting."
                )
                center_offset = calculate_offset(perception.segmentation, image)
                trigger_stop(
                    decision.distance_m,
                    stop_signal_path,
                    center_offset=center_offset,
                )
                node.stop()
                break

            frame_idx += 1
            elapsed = time.time() - loop_start
            sleep_s = max(0.0, min_interval_s - elapsed)
            if sleep_s:
                time.sleep(sleep_s)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop judge perception pipeline")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--text-prompt", default="purple boxes.")
    parser.add_argument("--live", action="store_true", help="Run on live ROS camera feed")
    parser.add_argument("--sim", action="store_true", help="Use Isaac sim ROS topics (default: real Go2)")
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=DEFAULT_STOP_DISTANCE_M,
        help="Stop when median masked depth is below this (meters)",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_S,
        help="Minimum seconds between judge loops",
    )
    parser.add_argument(
        "--stop-signal-file",
        type=Path,
        default=DEFAULT_STOP_SIGNAL_PATH,
        help="Shared stop signal file for OmniVLA-edge",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save debug JPGs each live loop")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved visualizations (default: run_images2/runN)",
    )
    parser.add_argument(
        "--scan-first",
        action="store_true",
        help="Run 360° SAM2 scan with the already-loaded segmenter before the live loop",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        default=None,
        help="Override cmd_vel topic for scan / live camera node",
    )
    args = parser.parse_args()

    print("loading perception models...", flush=True)
    # With --scan-first, load SAM2 only first; UniDepth waits until after the scan
    # so we don't pay for a second SAM2 load and avoid GPU contention during scan.
    judge = StopJudge(text_prompt=args.text_prompt, load_depth=not args.scan_first)
    print("STOP_JUDGE_READY", flush=True)

    if args.live:
        clear_stop(args.stop_signal_file)
        run_dir = None
        if args.save_viz:
            run_dir = args.output_dir or next_run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"output dir: {run_dir}", flush=True)

        if args.scan_first:
            from scan_surround import scan_surround

            print("[stop_judge] Starting 360° scan with preloaded SAM2...", flush=True)
            if not scan_surround(
                args.text_prompt,
                sim=args.sim,
                cmd_vel_topic=args.cmd_vel_topic,
                output_dir=run_dir,
                segmenter=judge.segmenter,
            ):
                print("[stop_judge] Target not found after 360 scan; exiting.", flush=True)
                sys.exit(1)
            print("SCAN_DONE", flush=True)

        judge.ensure_depth_ready()
        print("entering live perception loop", flush=True)
        print(f"stop if median masked depth < {args.stop_distance} m", flush=True)
        run_live_loop(
            judge=judge,
            sim=args.sim,
            stop_distance_m=args.stop_distance,
            min_interval_s=args.min_interval,
            stop_signal_path=args.stop_signal_file,
            output_dir=run_dir if args.save_viz else None,
            save_viz=args.save_viz,
            cmd_vel_topic=args.cmd_vel_topic,
        )
        return

    judge.ensure_depth_ready()

    run_dir = args.output_dir or next_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {run_dir}")
    decision, perception = judge.decide(Image.open(args.image), args.stop_distance)
    saved = save_perception_visualizations(perception, stem=args.image.stem, output_dir=run_dir)
    print(f"saved {len(saved)} visualizations to {run_dir}")
    for path in saved:
        print(f"  {path.name}")

    depth = perception.depth_meters.float().cpu().numpy()
    seg = perception.segmentation
    print(f"image: {args.image}")
    print(f"depth shape: {tuple(perception.depth_meters.shape)}")
    print(f"depth range (m): min={depth.min():.3f} max={depth.max():.3f}")
    print(f"detections: {len(seg.labels)}")
    for label, score in zip(seg.labels, seg.scores):
        print(f"  {label}: {score:.3f}")
    print(f"should_stop: {decision.should_stop} distance_m: {decision.distance_m}")


if __name__ == "__main__":
    main()
