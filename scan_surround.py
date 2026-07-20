"""Open-loop surround scan until SAM2 sees the target (or ~360°)."""

import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent
OMNIVLA_INFERENCE = ROOT.parent / "omni-VLA" / "OmniVLA" / "inference"
sys.path.insert(0, str(OMNIVLA_INFERENCE))
sys.path.insert(0, str(ROOT / "segmentation_implementation"))

from isaacsim_controller import IsaacSimPublisher
from run_grounded_sam2 import GroundedSAM2Segmenter

ANGULAR_VEL = 0.5
CHUNK_S = 0.6
MAX_YAW = 2 * math.pi
PUBLISH_INTERVAL = 0.1


def _wait_for_image(node: IsaacSimPublisher, timeout_sec: float = 5.0):
    """Wait for a fresh camera frame using the node's spin thread only.

    Do not call get_latest_image_pil() here: it spin_onces on the main thread
    while IsaacSimPublisher already spins in a background thread, which breaks
    cmd_vel delivery. center_target avoids that by never grabbing images.
    """
    node.latest_image = None
    deadline = time.monotonic() + timeout_sec
    while node.latest_image is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if node.latest_image is None:
        return None
    return PILImage.fromarray(node.latest_image, mode="RGB")


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.45):
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.any(axis=-1)
    if m.shape[:2] != out.shape[:2]:
        # Nearest-neighbor resize without OpenCV.
        ys = (np.linspace(0, m.shape[0] - 1, out.shape[0])).astype(int)
        xs = (np.linspace(0, m.shape[1] - 1, out.shape[1])).astype(int)
        m = m[ys][:, xs]
    color_arr = np.array(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * color_arr
    return out.astype(np.uint8)


def _save_scan_frame(
    output_dir: Path,
    frame_idx: int,
    image: PILImage.Image,
    result,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scan_{frame_idx}"
    image_path = output_dir / f"{stem}.jpg"
    image.save(image_path, quality=95)

    if not result.scores:
        return

    rgb = np.array(image.convert("RGB"))
    colors = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 64)]
    for i, mask in enumerate(result.masks):
        overlay = _overlay_mask(rgb, mask, colors[i % len(colors)])
        PILImage.fromarray(overlay).save(
            output_dir / f"{stem}_mask_{i}.jpg", quality=95
        )


def scan_surround(
    text_prompt: str,
    sim: bool = False,
    cmd_vel_topic: str | None = None,
    output_dir: Path | None = None,
    segmenter: GroundedSAM2Segmenter | None = None,
) -> bool:
    """Turn in place until SAM2 detects the target. Return True if found.

    Pass a preloaded ``segmenter`` to skip the expensive model load.
    """
    owns_segmenter = segmenter is None
    if owns_segmenter:
        print("[scan] Loading Grounded SAM2...", flush=True)
        segmenter = GroundedSAM2Segmenter()
        print("[scan] Models ready.", flush=True)
    else:
        print("[scan] Reusing preloaded Grounded SAM2.", flush=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scan] Saving frames to {output_dir}", flush=True)

    # Match center_target: init → publisher → discover sleep → publish loop.
    rclpy.init()
    node = IsaacSimPublisher(sim=sim, cmd_vel_topic=cmd_vel_topic)
    try:
        time.sleep(1.0)  # Allow the new ROS publisher to discover its subscriber.
        print(f"[scan] Searching for target on {node.cmd_vel_topic}.", flush=True)

        turned = 0.0
        frame_idx = 0
        while turned < MAX_YAW:
            image = _wait_for_image(node)
            if image is None:
                print("[scan] No camera frame; retrying...", flush=True)
                continue

            result = segmenter.segment(image, text_prompt)
            if output_dir is not None:
                _save_scan_frame(output_dir, frame_idx, image, result)
            frame_idx += 1

            if result.scores:
                print(
                    f"[scan] Target detected: {list(zip(result.labels, result.scores))}",
                    flush=True,
                )
                return True

            print(
                f"[scan] Not found; turning {CHUNK_S:.1f}s "
                f"({turned:.2f}/{MAX_YAW:.2f} rad)",
                flush=True,
            )
            # Same publish pattern as center_target.
            turn_end = time.monotonic() + CHUNK_S
            while time.monotonic() < turn_end:
                node.publish_velocity(0.0, ANGULAR_VEL)
                time.sleep(PUBLISH_INTERVAL)
            turned += ANGULAR_VEL * CHUNK_S
            time.sleep(1.0)

        print("[scan] Full 360° with no detection.", flush=True)
        return False
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
