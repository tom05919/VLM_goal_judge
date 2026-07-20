#!/usr/bin/env python3
"""Send live robot images to a remote OmniVLA server and execute its commands."""

import argparse
import io
import json
import math
import sys
import threading
import time
from pathlib import Path

import rclpy
import zmq
from PIL import Image

ROOT = Path(__file__).resolve().parent
OMNIVLA_INFERENCE = ROOT.parent / "omni-VLA" / "OmniVLA" / "inference"
sys.path.insert(0, str(OMNIVLA_INFERENCE))

from isaacsim_controller import IsaacSimPublisher
from stop_signal import DEFAULT_STOP_SIGNAL_PATH, is_stop_requested

MAX_LINEAR = 0.3
MAX_ANGULAR = 0.4


def _encode_jpeg(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _wait_for_fresh_image(node: IsaacSimPublisher, timeout_s: float = 5.0):
    """Wait without spinning; IsaacSimPublisher already owns a spin thread."""
    node.latest_image = None
    deadline = time.monotonic() + timeout_s
    while node.latest_image is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if node.latest_image is None:
        return None
    return Image.fromarray(node.latest_image.copy(), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote OmniVLA robot client")
    parser.add_argument("--endpoint", default="tcp://localhost:5555")
    parser.add_argument("--text-prompt", default="go to fire extinguisher")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--cmd-vel-topic", default=None)
    parser.add_argument(
        "--stop-signal-file",
        type=Path,
        default=DEFAULT_STOP_SIGNAL_PATH,
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(args.endpoint)

    rclpy.init()
    node = IsaacSimPublisher(sim=args.sim, cmd_vel_topic=args.cmd_vel_topic)
    estop = threading.Event()

    def watch_estop() -> None:
        try:
            while not estop.is_set():
                if input().strip().lower() in ("", "q"):
                    print("[REMOTE] Failsafe stop requested.", flush=True)
                    estop.set()
                    node.stop()
        except EOFError:
            pass

    threading.Thread(target=watch_estop, daemon=True).start()
    print(f"[REMOTE] Connected to {args.endpoint}", flush=True)

    try:
        while not estop.is_set():
            if is_stop_requested(args.stop_signal_file):
                print("[REMOTE] Stop judge requested halt.", flush=True)
                break

            image = _wait_for_fresh_image(node)
            if image is None:
                print("[REMOTE] No camera frame; retrying.", flush=True)
                continue

            request = json.dumps({"prompt": args.text_prompt}).encode()
            socket.send_multipart([request, _encode_jpeg(image)])
            deadline = time.monotonic() + args.timeout_ms / 1000.0
            while socket.poll(timeout=100, flags=zmq.POLLIN) == 0:
                if estop.is_set() or is_stop_requested(args.stop_signal_file):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for OmniVLA server at {args.endpoint}"
                    )
            if estop.is_set() or is_stop_requested(args.stop_signal_file):
                break
            reply = socket.recv_json()

            if "error" in reply:
                raise RuntimeError(f"OmniVLA server error: {reply['error']}")

            linear = float(reply["linear"])
            angular = float(reply["angular"])
            if not math.isfinite(linear) or not math.isfinite(angular):
                raise ValueError("Server returned a non-finite velocity")

            linear = max(0.0, min(MAX_LINEAR, linear))
            angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))

            if estop.is_set() or is_stop_requested(args.stop_signal_file):
                break
            node.publish_velocity(linear, angular)
    except zmq.Again as exc:
        raise TimeoutError(
            f"Timed out waiting for OmniVLA server at {args.endpoint}"
        ) from exc
    finally:
        estop.set()
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
