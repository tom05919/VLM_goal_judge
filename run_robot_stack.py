#!/usr/bin/env python3
"""Launch OmniVLA-edge and the live stop judge together."""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OMNIVLA_ROOT = ROOT.parent / "omni-VLA" / "OmniVLA"
OMNIVLA_INFERENCE = OMNIVLA_ROOT / "inference"
STOP_SIGNAL_PATH = ROOT / ".navigation_stop"
CONDA_SH = Path("/root/miniforge3/etc/profile.d/conda.sh")
ROS_SETUP = Path("/opt/ros/humble/setup.bash")

from stop_signal import (
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_STOP_DISTANCE_M,
    clear_stop,
    trigger_stop,
)


def _stream_output(
    proc: subprocess.Popen,
    prefix: str,
    ready_event: threading.Event | None = None,
) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")
        if ready_event is not None and "STOP_JUDGE_READY" in line:
            ready_event.set()


def _conda_run(
    env_name: str,
    script: Path,
    extra_args: list[str],
    workdir: Path | None = None,
    source_ros: bool = False,
) -> subprocess.Popen:
    workdir = workdir or script.parent
    cmd_parts = ["python", str(script), *extra_args]
    ros_source = ""
    if source_ros:
        if ROS_SETUP.is_file():
            ros_source = f"source {shlex.quote(str(ROS_SETUP))} && "
        else:
            print(f"[stack] WARNING: {ROS_SETUP} not found; rclpy may be missing", file=sys.stderr)
    cmd = (
        f"{ros_source}"
        f"source {shlex.quote(str(CONDA_SH))} && "
        f"conda activate {shlex.quote(env_name)} && "
        f"cd {shlex.quote(str(workdir))} && "
        f"{' '.join(shlex.quote(part) for part in cmd_parts)}"
    )
    return subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniVLA-edge + live stop judge")
    parser.add_argument("--sim", action="store_true", help="Use Isaac sim ROS topics (default: real Go2)")
    parser.add_argument("--text-prompt", default="purple boxes.")
    parser.add_argument("--stop-distance", type=float, default=DEFAULT_STOP_DISTANCE_M)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S)
    parser.add_argument("--stop-signal-file", type=Path, default=STOP_SIGNAL_PATH)
    parser.add_argument(
        "--cmd-vel-topic",
        default=None,
        help="Override cmd_vel topic (use /cmd_vel_out if robot.launch has teleop:=false)",
    )
    args = parser.parse_args()

    if not CONDA_SH.is_file():
        print(f"conda not found at {CONDA_SH}", file=sys.stderr)
        sys.exit(1)

    clear_stop(args.stop_signal_file)

    mode_args = ["--sim"] if args.sim else []
    shared_stop_args = ["--stop-signal-file", str(args.stop_signal_file)]

    omnivla_args = [*mode_args, *shared_stop_args]
    if args.cmd_vel_topic:
        omnivla_args.extend(["--cmd-vel-topic", args.cmd_vel_topic])
    stop_judge_args = [
        "--live",
        "--save-viz",
        *mode_args,
        "--text-prompt",
        args.text_prompt,
        "--stop-distance",
        str(args.stop_distance),
        "--min-interval",
        str(args.min_interval),
        *shared_stop_args,
    ]

    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    def shutdown(signum=None, frame=None):
        print("\n[stack] Shutting down...")
        trigger_stop(0.0, args.stop_signal_file)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    stop_judge_ready = threading.Event()

    print("[stack] Starting stop judge (perception env)...")
    stop_proc = _conda_run(
        "perception",
        ROOT / "stop_judge.py",
        stop_judge_args,
        workdir=ROOT,
        source_ros=True,
    )
    procs.append(stop_proc)
    stop_thread = threading.Thread(
        target=_stream_output,
        args=(stop_proc, "stop_judge", stop_judge_ready),
        daemon=True,
    )
    threads.append(stop_thread)
    stop_thread.start()

    print("[stack] Waiting for stop judge models to load...")
    if not stop_judge_ready.wait(timeout=600):
        print("[stack] WARNING: stop judge not ready after 10 min; starting OmniVLA anyway")
    else:
        print("[stack] Stop judge ready.")

    print("[stack] Starting OmniVLA-edge (real robot)...")
    omnivla_proc = _conda_run(
        "sim",
        OMNIVLA_INFERENCE / "run_omnivla_edge.py",
        omnivla_args,
        workdir=OMNIVLA_ROOT,
        source_ros=True,
    )
    procs.append(omnivla_proc)
    omnivla_thread = threading.Thread(
        target=_stream_output, args=(omnivla_proc, "omnivla"), daemon=True
    )
    threads.append(omnivla_thread)
    omnivla_thread.start()

    while True:
        for proc in procs:
            if proc.poll() is not None:
                print(f"[stack] Process exited with code {proc.returncode}")
                shutdown()
        time.sleep(2)


if __name__ == "__main__":
    main()
