# goal_stop_judge

Orchestrator for the Go2 stop-judge + OmniVLA navigation stack.

**Full run guide:** [`../README.md`](../README.md)  
**Everyday command:** `python go2_nav.py`

## Local scripts

| Script | Role |
|--------|------|
| `go2_nav.py` | Interactive CLI / `run` / `serve` (preferred entry) |
| `run_robot_stack.py` | Library + slim flags; launches stop judge + nav |
| `stop_judge.py` | Scan + live distance stop (SAM2 + UniDepth) |
| `scan_surround.py` | 360° scan helper used by stop judge |
| `server_client.py` | Remote full OmniVLA via ZeroMQ |
| `center_target.py` | Post-stop yaw centering (spawned by the stack) |
| `stop_signal.py` | Shared stop-file helpers |

## Environments

Recreate from manifests in [`envs/`](envs/) (`install_*_env.sh`, `VERSIONS.md`).
Do not mix `sim` / `perception` / `omnivla`, and do not source `/opt/ros` with RoboStack.

## Out of scope

[`VLM_implementation/`](VLM_implementation/) is experimental and is **not** used by
`go2_nav.py` / `run_robot_stack.py`.
