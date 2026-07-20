#!/usr/bin/env python3
"""Launch local or remote OmniVLA navigation with the live stop judge."""

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
sys.path.insert(0, str(OMNIVLA_INFERENCE))

from center_target import center_target
from stop_signal import (
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_STOP_DISTANCE_M,
    clear_stop,
    is_stop_requested,
    read_center_offset,
    trigger_stop,
)


def _next_run_dir() -> Path:
    """Create run_images2/run1, run2, ... for scan + stop_judge visualizations."""
    import re

    base = ROOT / "run_images2"
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


def _stream_output(
    proc: subprocess.Popen,
    prefix: str,
    ready_event: threading.Event | None = None,
) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")
        if ready_event is not None and "SCAN_DONE" in line:
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
    parser = argparse.ArgumentParser(description="Run OmniVLA + live stop judge")
    parser.add_argument("--sim", action="store_true", help="Use Isaac sim ROS topics (default: real Go2)")
    parser.add_argument(
        "--text-prompt",
        default="purple boxes.",
        help="SAM2 prompt for scan_surround + stop_judge (OmniVLA uses its own prompt)",
    )
    parser.add_argument("--stop-distance", type=float, default=DEFAULT_STOP_DISTANCE_M)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S)
    parser.add_argument("--stop-signal-file", type=Path, default=STOP_SIGNAL_PATH)
    parser.add_argument(
        "--navigation",
        choices=("edge", "remote"),
        default="edge",
        help="Run OmniVLA-edge locally or use a remote full-model server",
    )
    parser.add_argument(
        "--server-endpoint",
        default="tcp://localhost:5555",
        help="Remote OmniVLA endpoint used with --navigation remote",
    )
    parser.add_argument(
        "--server-timeout-ms",
        type=int,
        default=120_000,
        help="Remote send/receive timeout in milliseconds",
    )
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

    run_dir = _next_run_dir()
    print(f"[stack] Run folder: {run_dir}")

    mode_args = ["--sim"] if args.sim else []
    shared_stop_args = ["--stop-signal-file", str(args.stop_signal_file)]

    omnivla_args = [*mode_args, *shared_stop_args]
    if args.cmd_vel_topic:
        omnivla_args.extend(["--cmd-vel-topic", args.cmd_vel_topic])
    stop_judge_args = [
        "--live",
        "--save-viz",
        "--scan-first",
        "--output-dir",
        str(run_dir),
        *mode_args,
        "--text-prompt",
        args.text_prompt,
        "--stop-distance",
        str(args.stop_distance),
        "--min-interval",
        str(args.min_interval),
        *shared_stop_args,
    ]
    if args.cmd_vel_topic:
        stop_judge_args.extend(["--cmd-vel-topic", args.cmd_vel_topic])

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

    print("[stack] Starting stop judge (load models + 360° scan)...")
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

    print("[stack] Waiting for models + scan to finish...")
    if not stop_judge_ready.wait(timeout=900):
        print("[stack] WARNING: scan not done after 15 min; starting OmniVLA anyway")
    else:
        print("[stack] Scan complete; starting navigation.")

    if args.navigation == "edge":
        navigation_script = OMNIVLA_INFERENCE / "run_omnivla_edge.py"
        navigation_env = "sim"
        navigation_args = omnivla_args
        navigation_label = "omnivla-edge"
        source_ros = False
    else:
        navigation_script = ROOT / "server_client.py"
        navigation_env = "perception"
        navigation_args = [
            *omnivla_args,
            "--endpoint",
            args.server_endpoint,
            "--timeout-ms",
            str(args.server_timeout_ms),
        ]
        navigation_label = "omnivla-remote"
        source_ros = True

    print(f"[stack] Starting {navigation_label} navigation...")
    omnivla_proc = _conda_run(
        navigation_env,
        navigation_script,
        navigation_args,
        workdir=OMNIVLA_ROOT if args.navigation == "edge" else ROOT,
        source_ros=source_ros,
    )
    procs.append(omnivla_proc)
    omnivla_thread = threading.Thread(
        target=_stream_output,
        args=(omnivla_proc, navigation_label),
        daemon=True,
    )
    threads.append(omnivla_thread)
    omnivla_thread.start()

    while True:
        if is_stop_requested(args.stop_signal_file):
            break
        for proc in procs:
            if proc.poll() is not None:
                print(f"[stack] Process exited with code {proc.returncode}")
                shutdown()
        time.sleep(0.5)

    print("[stack] Stop requested; waiting for navigation processes to exit...")
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)

    center_offset = read_center_offset(args.stop_signal_file)
    if center_offset is None:
        print("[stack] No target offset was saved; skipping centering.")
        return

    print(f"[stack] Centering target from offset {center_offset:.1f}px...")
    center_target(
        center_offset,
        sim=args.sim,
        cmd_vel_topic=args.cmd_vel_topic,
    )
    print("[stack] Target centering complete.")


if __name__ == "__main__":
    main()
