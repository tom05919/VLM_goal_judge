"""Shared latched STOP signal between stop_judge and OmniVLA-edge."""

from pathlib import Path

DEFAULT_STOP_SIGNAL_PATH = Path(__file__).resolve().parent / ".navigation_stop"
DEFAULT_TARGET_DISTANCE_PATH = DEFAULT_STOP_SIGNAL_PATH.parent / ".target_distance"
DEFAULT_STOP_DISTANCE_M = 0.8
DEFAULT_MIN_INTERVAL_S = 1.0


def clear_stop(path: Path | None = None) -> None:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    if signal_path.exists():
        signal_path.unlink()
    distance_path = signal_path.parent / ".target_distance"
    if distance_path.exists():
        distance_path.unlink()


def write_target_distance(distance_m: float, path: Path | None = None) -> None:
    (path or DEFAULT_TARGET_DISTANCE_PATH).write_text(f"{distance_m}\n", encoding="utf-8")


def read_target_distance(path: Path | None = None) -> float | None:
    distance_path = path or DEFAULT_TARGET_DISTANCE_PATH
    if not distance_path.exists():
        return None
    try:
        return float(distance_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def trigger_stop(distance_m: float, path: Path | None = None) -> None:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    signal_path.write_text(f"STOP\n{distance_m}\n", encoding="utf-8")


def is_stop_requested(path: Path | None = None) -> bool:
    signal_path = path or DEFAULT_STOP_SIGNAL_PATH
    if not signal_path.exists():
        return False
    return signal_path.read_text(encoding="utf-8").strip().startswith("STOP")
