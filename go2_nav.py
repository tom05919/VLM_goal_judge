#!/usr/bin/env python3
"""Friendly CLI for the Go2 stop-judge + OmniVLA navigation stack.

Everyday use:
  python go2_nav.py              # interactive wizard
  python go2_nav.py run --sam ... --vla ... --nav edge
  python go2_nav.py serve        # GPU host: full OmniVLA ZeroMQ server
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OMNIVLA_ROOT = ROOT.parent / "omni-VLA"
OMNIVLA_INFERENCE = OMNIVLA_ROOT / "inference"

from run_robot_stack import (
    DEFAULT_ENDPOINT,
    DEFAULT_SAM_PROMPT,
    DEFAULT_VLA_PROMPT,
    STOP_SIGNAL_PATH,
    StackConfig,
    launch_stack,
    resolve_conda_sh,
)
from stop_signal import DEFAULT_MIN_INTERVAL_S, DEFAULT_STOP_DISTANCE_M


def _prompt(message: str, default: str) -> str:
    raw = input(f"{message} [{default}]: ").strip()
    return raw if raw else default


def _prompt_choice(message: str, options: list[tuple[str, str]], default_key: str) -> str:
    print(message)
    key_by_index: dict[str, str] = {}
    for i, (key, label) in enumerate(options, start=1):
        marker = " (default)" if key == default_key else ""
        print(f"  [{i}] {key:6} — {label}{marker}")
        key_by_index[str(i)] = key
    default_index = next(
        str(i) for i, (key, _) in enumerate(options, start=1) if key == default_key
    )
    raw = input(f"Choice [{default_index}]: ").strip() or default_index
    if raw in key_by_index:
        return key_by_index[raw]
    if raw in {key for key, _ in options}:
        return raw
    print(f"Invalid choice {raw!r}; using {default_key!r}.", file=sys.stderr)
    return default_key


def run_wizard() -> StackConfig:
    print()
    print("  Go2 navigation stack")
    print("  --------------------")
    sam = _prompt("  SAM2 detect prompt", DEFAULT_SAM_PROMPT)
    vla = _prompt("  OmniVLA language prompt", DEFAULT_VLA_PROMPT)
    print()
    nav = _prompt_choice(
        "  Navigation model:",
        [
            ("edge", "OmniVLA-edge on this PC (sim env)"),
            ("full", "full OmniVLA on this PC (omnivla env)"),
            ("remote", "full OmniVLA via ZeroMQ (GPU host)"),
        ],
        "edge",
    )
    endpoint = DEFAULT_ENDPOINT
    if nav == "remote":
        endpoint = _prompt("  Server endpoint", DEFAULT_ENDPOINT)
    print()
    platform = _prompt_choice(
        "  Platform:",
        [
            ("real", "real Go2"),
            ("sim", "Isaac Sim"),
        ],
        "real",
    )
    print()
    print("  Starting… (scan → stop judge → navigation)")
    print()
    return StackConfig(
        sam_prompt=sam,
        vla_prompt=vla,
        navigation=nav,
        endpoint=endpoint,
        sim=(platform == "sim"),
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sam", default=DEFAULT_SAM_PROMPT, help="SAM2 detect prompt")
    parser.add_argument("--vla", default=DEFAULT_VLA_PROMPT, help="OmniVLA language prompt")
    parser.add_argument(
        "--nav",
        "--navigation",
        dest="navigation",
        choices=("edge", "full", "remote"),
        default="edge",
    )
    parser.add_argument("--endpoint", "--server-endpoint", dest="endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--server-timeout-ms", type=int, default=120_000)
    parser.add_argument("--sim", action="store_true", help="Use Isaac sim ROS topics")
    parser.add_argument("--stop-distance", type=float, default=DEFAULT_STOP_DISTANCE_M)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S)
    parser.add_argument("--stop-signal-file", type=Path, default=STOP_SIGNAL_PATH)
    parser.add_argument("--cmd-vel-topic", default=None)


def _config_from_run_args(args: argparse.Namespace) -> StackConfig:
    return StackConfig(
        sam_prompt=args.sam,
        vla_prompt=args.vla,
        navigation=args.navigation,
        endpoint=args.endpoint,
        server_timeout_ms=args.server_timeout_ms,
        sim=args.sim,
        stop_distance=args.stop_distance,
        min_interval=args.min_interval,
        stop_signal_file=args.stop_signal_file,
        cmd_vel_topic=args.cmd_vel_topic,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Run full OmniVLA as a ZeroMQ server (GPU host). Activate omnivla env."""
    conda_sh = resolve_conda_sh()
    script = OMNIVLA_INFERENCE / "run_omnivla.py"
    if not script.is_file():
        raise SystemExit(f"Missing {script}")

    # --vla is fallback only; live prompt comes from the client each request.
    extra = [
        "--mode",
        "serve",
        "--bind",
        args.bind,
        "--text-prompt",
        args.vla,
    ]
    cmd = (
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate omnivla && "
        f"cd {shlex.quote(str(OMNIVLA_ROOT))} && "
        f"python {shlex.quote(str(script))} "
        f"{' '.join(shlex.quote(p) for p in extra)}"
    )
    print(
        "[go2_nav] Serving full OmniVLA "
        f"(bind={args.bind}; --vla is fallback only — client sends live prompt).",
        flush=True,
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    raise SystemExit(subprocess.call(["bash", "-lc", cmd], env=env))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="go2_nav.py",
        description="Go2 navigation stack (stop judge + OmniVLA)",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Non-interactive stack launch")
    _add_run_arguments(run_parser)

    serve_parser = sub.add_parser("serve", help="GPU host: run full OmniVLA ZeroMQ server")
    serve_parser.add_argument("--bind", default="tcp://*:5555")
    serve_parser.add_argument(
        "--vla",
        default=DEFAULT_VLA_PROMPT,
        help="Fallback language prompt if client omits metadata prompt",
    )

    # No subcommand and no flags → interactive wizard.
    if not argv:
        launch_stack(run_wizard())
        return

    # Allow `go2_nav.py --sam ...` as alias for `run` without requiring the subcommand.
    if argv[0] not in ("run", "serve", "-h", "--help"):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if args.command == "serve":
        cmd_serve(args)
        return
    if args.command == "run":
        launch_stack(_config_from_run_args(args))
        return
    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
