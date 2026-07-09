"""Shared latched STOP signal between stop_judge and OmniVLA-edge."""

from pathlib import Path

DEFAULT_STOP_SIGNAL_PATH = Path(__file__).resolve().parent / ".navigation_stop"
DEFAULT_STOP_DISTANCE_M = 1.0
DEFAULT_MIN_INTERVAL_S = 1.0


def clear_stop(path: Path | None = None) -> None:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    if signal_path.exists():
        signal_path.unlink()


def trigger_stop(distance_m: float, path: Path | None = None) -> None:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    signal_path.write_text(f"STOP\n{distance_m}\n", encoding="utf-8")


def is_stop_requested(path: Path | None = None) -> bool:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    if not signal_path.exists():
        return False
    return signal_path.read_text(encoding="utf-8").strip().startswith("STOP")
