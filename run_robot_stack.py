#!/usr/bin/env python3
"""Launch local or remote OmniVLA navigation with the live stop judge.

Prefer ``python go2_nav.py`` for interactive use. This module exposes
``StackConfig`` / ``launch_stack`` for the CLI and a slim ``__main__``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OMNIVLA_ROOT = ROOT.parent / "omni-VLA" / "OmniVLA"
OMNIVLA_INFERENCE = OMNIVLA_ROOT / "inference"
STOP_SIGNAL_PATH = ROOT / ".navigation_stop"

from stop_signal import (
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_STOP_DISTANCE_M,
    clear_stop,
    is_stop_requested,
    read_center_offset,
    trigger_stop,
)

DEFAULT_SAM_PROMPT = "purple boxes."
DEFAULT_VLA_PROMPT = "go to the human with white shirt"
DEFAULT_ENDPOINT = "tcp://localhost:5555"


def resolve_conda_sh() -> Path:
    """Locate conda.sh without hardcoding /root/miniforge3."""
    override = os.environ.get("GO2_CONDA_BASE")
    if override:
        candidate = Path(override).expanduser() / "etc" / "profile.d" / "conda.sh"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"GO2_CONDA_BASE={override!r} but conda.sh not found at {candidate}"
        )

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidate = Path(conda_exe).resolve().parent.parent / "etc" / "profile.d" / "conda.sh"
        if candidate.is_file():
            return candidate

    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX")
    if mamba_root:
        candidate = Path(mamba_root) / "etc" / "profile.d" / "conda.sh"
        if candidate.is_file():
            return candidate

    try:
        base = subprocess.check_output(
            ["conda", "info", "--base"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        candidate = Path(base) / "etc" / "profile.d" / "conda.sh"
        if candidate.is_file():
            return candidate
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    raise FileNotFoundError(
        "Could not find conda.sh. Set GO2_CONDA_BASE to your conda/miniforge root "
        "(the directory that contains etc/profile.d/conda.sh)."
    )


@dataclass
class StackConfig:
    sam_prompt: str = DEFAULT_SAM_PROMPT
    vla_prompt: str = DEFAULT_VLA_PROMPT
    navigation: str = "edge"  # edge | full | remote
    endpoint: str = DEFAULT_ENDPOINT
    server_timeout_ms: int = 120_000
    sim: bool = False
    stop_distance: float = DEFAULT_STOP_DISTANCE_M
    min_interval: float = DEFAULT_MIN_INTERVAL_S
    stop_signal_file: Path = field(default_factory=lambda: STOP_SIGNAL_PATH)
    cmd_vel_topic: str | None = None


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
    conda_sh: Path,
    workdir: Path | None = None,
) -> subprocess.Popen:
    """Activate a conda env and run a script. RoboStack-only — no /opt/ros."""
    workdir = workdir or script.parent
    cmd_parts = ["python", str(script), *extra_args]
    cmd = (
        f"source {shlex.quote(str(conda_sh))} && "
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


def _navigation_spec(config: StackConfig, shared_nav_args: list[str]) -> tuple[str, Path, list[str], Path, str]:
    """Return (env, script, args, workdir, label) for the chosen navigation backend."""
    if config.navigation == "edge":
        return (
            "sim",
            OMNIVLA_INFERENCE / "run_omnivla_edge.py",
            [*shared_nav_args, "--text-prompt", config.vla_prompt],
            OMNIVLA_ROOT,
            "omnivla-edge",
        )
    if config.navigation == "full":
        return (
            "omnivla",
            OMNIVLA_INFERENCE / "run_omnivla.py",
            [
                *shared_nav_args,
                "--mode",
                "local",
                "--text-prompt",
                config.vla_prompt,
            ],
            OMNIVLA_ROOT,
            "omnivla-full",
        )
    if config.navigation == "remote":
        return (
            "perception",
            ROOT / "server_client.py",
            [
                *shared_nav_args,
                "--text-prompt",
                config.vla_prompt,
                "--endpoint",
                config.endpoint,
                "--timeout-ms",
                str(config.server_timeout_ms),
            ],
            ROOT,
            "omnivla-remote",
        )
    raise ValueError(f"Unknown navigation mode: {config.navigation!r}")


def launch_stack(config: StackConfig) -> None:
    """Run stop_judge (scan + live) then the selected navigation backend."""
    if config.navigation not in ("edge", "full", "remote"):
        raise ValueError(f"navigation must be edge|full|remote, got {config.navigation!r}")

    conda_sh = resolve_conda_sh()
    clear_stop(config.stop_signal_file)

    run_dir = _next_run_dir()
    print(f"[stack] Run folder: {run_dir}")
    print(f"[stack] SAM prompt: {config.sam_prompt!r}")
    print(f"[stack] VLA prompt: {config.vla_prompt!r}")
    print(f"[stack] Navigation: {config.navigation}")

    mode_args = ["--sim"] if config.sim else []
    shared_stop_args = ["--stop-signal-file", str(config.stop_signal_file)]
    shared_nav_args = [*mode_args, *shared_stop_args]
    if config.cmd_vel_topic:
        shared_nav_args.extend(["--cmd-vel-topic", config.cmd_vel_topic])

    stop_judge_args = [
        "--live",
        "--save-viz",
        "--scan-first",
        "--output-dir",
        str(run_dir),
        *mode_args,
        "--text-prompt",
        config.sam_prompt,
        "--stop-distance",
        str(config.stop_distance),
        "--min-interval",
        str(config.min_interval),
        *shared_stop_args,
    ]
    if config.cmd_vel_topic:
        stop_judge_args.extend(["--cmd-vel-topic", config.cmd_vel_topic])

    procs: list[subprocess.Popen] = []

    def shutdown(signum=None, frame=None):
        print("\n[stack] Shutting down...")
        trigger_stop(0.0, config.stop_signal_file)
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
        conda_sh=conda_sh,
        workdir=ROOT,
    )
    procs.append(stop_proc)
    stop_thread = threading.Thread(
        target=_stream_output,
        args=(stop_proc, "stop_judge", stop_judge_ready),
        daemon=True,
    )
    stop_thread.start()

    print("[stack] Waiting for models + scan to finish...")
    if not stop_judge_ready.wait(timeout=900):
        print("[stack] WARNING: scan not done after 15 min; starting OmniVLA anyway")
    else:
        print("[stack] Scan complete; starting navigation.")

    nav_env, nav_script, nav_args, nav_workdir, nav_label = _navigation_spec(
        config, shared_nav_args
    )
    print(f"[stack] Starting {nav_label} navigation...")
    omnivla_proc = _conda_run(
        nav_env,
        nav_script,
        nav_args,
        conda_sh=conda_sh,
        workdir=nav_workdir,
    )
    procs.append(omnivla_proc)
    omnivla_thread = threading.Thread(
        target=_stream_output,
        args=(omnivla_proc, nav_label),
        daemon=True,
    )
    omnivla_thread.start()

    while True:
        if is_stop_requested(config.stop_signal_file):
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

    center_offset = read_center_offset(config.stop_signal_file)
    if center_offset is None:
        print("[stack] No target offset was saved; skipping centering.")
        return

    print(f"[stack] Centering target from offset {center_offset:.1f}px...")
    # Lazy import: center_target pulls rclpy; keep go2_nav --help ROS-free.
    if str(OMNIVLA_INFERENCE) not in sys.path:
        sys.path.insert(0, str(OMNIVLA_INFERENCE))
    from center_target import center_target

    center_target(
        center_offset,
        sim=config.sim,
        cmd_vel_topic=config.cmd_vel_topic,
    )
    print("[stack] Target centering complete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run OmniVLA + live stop judge (prefer: python go2_nav.py)",
    )
    parser.add_argument(
        "--sam",
        "--text-prompt",
        dest="sam_prompt",
        default=DEFAULT_SAM_PROMPT,
        help="SAM2 prompt for scan_surround + stop_judge",
    )
    parser.add_argument(
        "--vla",
        dest="vla_prompt",
        default=DEFAULT_VLA_PROMPT,
        help="OmniVLA language navigation prompt",
    )
    parser.add_argument("--sim", action="store_true", help="Use Isaac sim ROS topics")
    parser.add_argument("--stop-distance", type=float, default=DEFAULT_STOP_DISTANCE_M)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S)
    parser.add_argument("--stop-signal-file", type=Path, default=STOP_SIGNAL_PATH)
    parser.add_argument(
        "--navigation",
        choices=("edge", "full", "remote"),
        default="edge",
        help="edge=local OmniVLA-edge; full=local full model; remote=ZeroMQ client",
    )
    parser.add_argument(
        "--server-endpoint",
        default=DEFAULT_ENDPOINT,
        help="Remote OmniVLA endpoint used with --navigation remote",
    )
    parser.add_argument("--server-timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--cmd-vel-topic",
        default=None,
        help="Override cmd_vel topic (use /cmd_vel_out if robot.launch has teleop:=false)",
    )
    args = parser.parse_args(argv)

    config = StackConfig(
        sam_prompt=args.sam_prompt,
        vla_prompt=args.vla_prompt,
        navigation=args.navigation,
        endpoint=args.server_endpoint,
        server_timeout_ms=args.server_timeout_ms,
        sim=args.sim,
        stop_distance=args.stop_distance,
        min_interval=args.min_interval,
        stop_signal_file=args.stop_signal_file,
        cmd_vel_topic=args.cmd_vel_topic,
    )
    launch_stack(config)


if __name__ == "__main__":
    main()
