# VLM Semantic Stop Judge Testing Protocol

## Core Experimental Question

Given OmniVLA-edge as a fixed language-conditioned navigation policy, can a VLM-based judge correctly determine when the robot should stop?

The project does **not** evaluate whether OmniVLA can navigate to language-specified goals in general. That has already been demonstrated by the OmniVLA paper. This benchmark evaluates the missing **semantic termination layer**.

---

## Main Hypotheses

| ID | Hypothesis | Required Evidence |
|---|---|---|
| H1 | OmniVLA-edge can reach or approach language-specified goals but lacks reliable autonomous stopping. | No-judge baseline reaches goal but requires timeout/manual stop or continues past goal. |
| H2 | A fixed-rate VLM judge improves semantic stopping compared to timeout and heuristic stopping. | Higher correct-stop rate and lower late-stop/manual-intervention rate. |
| H3 | An adaptive/event-triggered VLM judge preserves stop quality while reducing VLM calls. | Similar correct-stop rate to fixed-rate VLM judge with fewer VLM calls per trial. |
| H4 | Stop failures can be separated into base-policy failures, judge failures, scheduler failures, and system failures. | Trial annotations and failure taxonomy. |

---

# 1. System Variants to Test

| Variant ID | Name | Description | Purpose | Final Paper? |
|---|---|---|---|---|
| V0 | Oracle human stop | Human labels exact completion point from video/logs. Not a deployed method. | Upper-bound analysis for stop timing. | Analysis only |
| V1 | OmniVLA + no judge / timeout | OmniVLA runs until timeout or manual stop. | Shows missing-stop problem. | Yes |
| V2 | OmniVLA + heuristic stop | Stop using low-level signals such as velocity, waypoint magnitude, or distance traveled. | Tests whether a non-semantic stop rule is enough. | Yes |
| V3 | OmniVLA + fixed-rate VLM judge | VLM checks completion every fixed interval. | Tests whether VLM judging solves stopping. | Yes |
| V4 | OmniVLA + adaptive VLM judge | Cheap scheduler decides when to query VLM. | Main method. Tests query efficiency. | Yes |
| V5 | High-frequency VLM judge | VLM checks very frequently, e.g. every 1 sec. | Optional upper bound for fixed-rate VLM performance. | Optional / pilot only |

---

# 2. Task Suite

Use tasks where completion is visually decidable. Avoid vague tasks like "check for danger" unless using obvious proxy objects.

| Task ID | Task Type | Example Command | Success Condition | False Stop Condition | Late Stop Condition | Notes |
|---|---|---|---|---|---|---|
| T1 | Doorway approach | "Go to the doorway." | Robot stops within 0.5–1.0 m of doorway threshold. | Stops >1.5 m before doorway while path remains open. | Reaches doorway but continues >3 sec or travels >1.0 m past threshold. | Cleanest semantic stop task. |
| T2 | Room entry | "Enter the room." | Robot crosses doorway threshold and stops inside room. | Stops before crossing threshold. | Enters room but keeps moving too far, exits, or requires manual stop. | Tests semantic boundary crossing. |
| T3 | Hallway endpoint | "Go to the end of the hallway." | Robot stops near visible end of hallway or point where forward progress ends. | Stops halfway with clear path ahead. | Reaches end region but keeps pushing/turning/oscillating. | Tests geometric completion. |
| T4 | Object approach | "Go to the backpack." | Robot stops within 0.5–1.5 m of target object with target visible. | Stops before target is reached or at wrong object. | Reaches inspection distance but passes object or continues moving. | Tests object-language grounding for stopping. |
| T5 | Landmark/sign target | "Find the exit sign." | Target sign/landmark is visible and robot stops near inspection distance. | Stops without visual evidence of target. | Sees/reaches target but continues searching. | First-responder flavored but still visually checkable. |
| T6 | Negative / absent target | "Find the red backpack." when no red backpack exists. | Robot should not mark complete without evidence. | Stops and reports complete despite absent target. | N/A | Optional stress test for hallucinated completion. |

---

# 3. Trial Types

| Trial Type ID | Trial Type | What Is Being Tested | Inputs | Outputs | Use For |
|---|---|---|---|---|---|
| TT1 | Offline VLM judge benchmark | Whether the VLM can classify completion from images/clips. | Command + image or short image history. | complete / incomplete / uncertain / unsafe. | Prompt selection, threshold tuning, VLM sanity check. |
| TT2 | Sim integration sanity | Whether full loop runs safely and logs correctly before real robot. | Sim camera + OmniVLA + judge + stop wrapper. | Sim trajectory, judge calls, stop decision. | Debugging only. |
| TT3 | Real no-judge baseline | Whether OmniVLA reaches goals but fails to stop. | Real camera + OmniVLA. | Goal-reached timestamp, manual stop, timeout. | Missing-stop evidence. |
| TT4 | Heuristic stop baseline | Whether low-level non-semantic rules can stop reliably. | OmniVLA outputs, velocity, odom, waypoint/action magnitude. | Stop decision from heuristic. | Reviewer control baseline. |
| TT5 | Fixed-rate VLM judge | Whether VLM judge can stop correctly. | Command + current image every K sec. | VLM completion output and stop decision. | Main comparison. |
| TT6 | Fixed-rate sweep | Which VLM query rate is reasonable. | Same as TT5 at multiple rates. | Stop quality vs VLM calls. | Pilot tuning. |
| TT7 | Adaptive scheduler pilot | Which adaptive triggers are useful. | Cheap motion/visual signals + VLM judge. | Query timing and stop decision. | Scheduler selection. |
| TT8 | Main real-robot benchmark | Final comparison of V1–V4. | Locked tasks, locked prompts, locked scheduler. | Metrics and paper results. | Main paper evidence. |
| TT9 | Stress / robustness trials | Whether judge fails under harder conditions. | Target absent, lighting change, clutter, alternate hallway. | Failure modes and robustness metrics. | Optional extra paper strength. |

---

# 4. Required Trial Counts

## Recommended Main Plan

| Phase | Trial Type | Tasks | Variants / Conditions | Trials per Condition | Total Trials / Samples | Sim / Real Split | Purpose |
|---|---|---:|---:|---:|---:|---|---|
| P0 | Offline VLM judge benchmark | 5–6 task types | Prompt/input variants | N/A | 100–300 labeled samples | 70–90% real frames, 10–30% sim frames | Select prompt, confidence threshold, and input format. |
| P1 | Sim integration sanity | 3 tasks | V1, V3, V4 | 2–3 | 18–27 sim trials | 100% sim | Verify logging, stop wrapper, judge integration, and safety before real robot. |
| P2 | Real no-judge baseline | 5 tasks | V1 | 5 | 25 real trials | 100% real | Prove missing-stop behavior and establish OmniVLA reach rate in your setup. |
| P3 | Heuristic baseline pilot | 3 tasks | V2 heuristic variants | 3 | 9–18 real trials | 100% real | Choose the simplest defensible heuristic baseline. |
| P4 | Fixed-rate VLM sweep | 3 tasks | 1s, 2s, 5s query intervals | 3 | 27 real trials | 100% real | Pick fixed-rate baseline for final comparison. |
| P5 | Adaptive scheduler pilot | 3 tasks | 2–3 adaptive schedulers | 3 | 18–27 real trials | 100% real | Pick final adaptive scheduler. |
| P6 | Main benchmark | 5 tasks | V1, V2, V3, V4 | 5 | 100 real trials | 100% real | Main paper result. |
| P7 | Optional stress tests | 2–3 tasks | V1, V3, V4 | 3 | 18–27 real trials | 100% real | Test target absence, lighting, clutter, or held-out hallway. |

---

## Minimum Acceptable Plan

Use this if hardware time is tight.

| Phase | Trial Type | Tasks | Variants / Conditions | Trials per Condition | Total Trials / Samples | Sim / Real Split | Purpose |
|---|---|---:|---:|---:|---:|---|---|
| P0-min | Offline VLM judge benchmark | 4–5 task types | Prompt/input variants | N/A | 50–100 labeled samples | Mostly real frames | Basic VLM judge validation. |
| P1-min | Sim sanity | 2–3 tasks | V1, V3, V4 | 2 | 12–18 sim trials | 100% sim | Integration check. |
| P2-min | Real no-judge baseline | 4 tasks | V1 | 5 | 20 real trials | 100% real | Missing-stop evidence. |
| P3-min | Main benchmark | 4 tasks | V1, V3, V4 | 5 | 60 real trials | 100% real | Minimum publishable comparison. |
| P4-min | Optional heuristic check | 2–3 tasks | V2 | 3 | 6–9 real trials | 100% real | Add if reviewer-control baseline is needed. |

---

## Strong Plan

Use this if the system is stable and time allows.

| Phase | Trial Type | Tasks | Variants / Conditions | Trials per Condition | Total Trials / Samples | Sim / Real Split | Purpose |
|---|---|---:|---:|---:|---:|---|---|
| P0-strong | Offline VLM judge benchmark | 6 task types | Prompt/input variants | N/A | 300–500 labeled samples | 80% real, 20% sim | Strong offline judge evaluation. |
| P1-strong | Sim integration and ablation | 5 tasks | V1, V2, V3, V4 | 5 | 100 sim trials | 100% sim | Controlled ablation and debugging. |
| P2-strong | Main real benchmark | 5 tasks | V1, V2, V3, V4 | 10 | 200 real trials | 100% real | Strong final result. |
| P3-strong | Held-out environment benchmark | 3 tasks | V1, V3, V4 | 5 | 45 real trials | 100% real | Tests location transfer. |
| P4-strong | Stress tests | 3 tasks | V1, V3, V4 | 5 | 45 real trials | 100% real | Tests lighting, clutter, absent targets. |

---

# 5. Offline VLM Judge Benchmark

## Dataset Construction

| Category | Percentage Target | Description |
|---|---:|---|
| Clearly incomplete | 25% | Goal not yet visible or robot clearly far from target. |
| Near-complete but not complete | 25% | Target visible, but robot should not stop yet. |
| Complete | 25% | Human would say command is satisfied. |
| After-complete / overshot | 15% | Robot already passed correct stop point. |
| Ambiguous / uncertain / unsafe | 10% | Poor visibility, occlusion, collision risk, unclear target. |

## Input Variants to Test

| Input Variant | Description | Purpose |
|---|---|---|
| I1 | Current frame only | Fastest and simplest. |
| I2 | Initial frame + current frame | Helps with progress understanding. |
| I3 | Last 3 frames | Helps with threshold-crossing tasks like entering a room. |
| I4 | Current frame + short textual history | Tests whether history improves decisions without many images. |

## Prompt Variants to Test

| Prompt Variant | Output Format | Expected Use |
|---|---|---|
| P1 | yes / no | Baseline only; likely too brittle. |
| P2 | complete / incomplete / uncertain | Better for avoiding forced decisions. |
| P3 | complete / incomplete / uncertain / unsafe | Recommended deployment format. |
| P4 | strict JSON with status, confidence, evidence, should_stop | Final candidate. |

## Offline Metrics

| Metric | Definition | Why It Matters |
|---|---|---|
| Completion precision | Of frames marked complete, how many are truly complete. | Measures premature-stop risk. |
| Completion recall | Of truly complete frames, how many are detected. | Measures missed-stop risk. |
| False complete rate | Fraction of non-complete frames marked complete. | Most safety-critical offline metric. |
| Uncertain rate | Fraction of frames marked uncertain. | Measures judge usefulness. |
| JSON parse failure rate | Fraction of responses not parseable. | Deployment robustness. |
| Average VLM latency | Mean inference time per VLM call. | Determines feasible query rate. |
| Confidence calibration | Whether high confidence corresponds to correctness. | Helps set thresholds. |

---

# 6. Online Real-Robot Metrics

| Metric | Symbol / Field | Definition | Primary Use |
|---|---|---|---|
| Reach rate | `reach_rate` | Fraction of trials where OmniVLA reaches the goal region at any point. | Separates navigation failure from judge failure. |
| Correct stop rate | `correct_stop_rate` | Fraction of trials where robot stops at the correct goal region. | Main task-level success metric. |
| Conditional correct stop rate | `correct_stop_given_reach` | Correct stops among trials where OmniVLA reached the goal. | Isolates VLM judge performance. |
| False stop rate | `false_stop_rate` | Robot stops before command is complete. | Measures premature stopping. |
| Late stop rate | `late_stop_rate` | Robot reaches goal but stops too late. | Measures missing-stop behavior. |
| Stop delay | `t_stop - t_goal_complete` | Seconds between human-labeled completion and robot stop. | Direct stop timing metric. |
| Overshoot distance | `distance_after_goal_complete` | Distance traveled after goal completion before stopping. | Physical version of stop delay. |
| Timeout rate | `timeout_rate` | Trial ends by timeout. | Measures failure to terminate. |
| Manual intervention rate | `manual_intervention_rate` | Human had to stop or correct robot. | Deployment reliability. |
| Collision / safety-stop rate | `safety_stop_rate` | Trial triggered safety stop or collision event. | Safety. |
| VLM calls per trial | `num_vlm_calls` | Number of VLM judge queries. | Query-efficiency metric. |
| Mean VLM latency | `vlm_latency_mean` | Average VLM inference time. | Compute cost. |
| Max VLM latency | `vlm_latency_max` | Worst observed VLM inference time. | Real-time risk. |
| Total trial time | `trial_duration` | Start to stop/timeout/manual intervention. | Efficiency. |

---

# 7. Stop Timing Labels

Each trial should contain these timestamps.

| Field | Definition |
|---|---|
| `t_start` | Trial start time. |
| `t_goal_first_visible` | First time the target/goal becomes visually visible, if applicable. |
| `t_goal_complete` | Human-labeled time when the command is satisfied. |
| `t_first_vlm_complete` | First VLM response marked complete. |
| `t_stop_command` | Time system issues stop command. |
| `t_robot_stopped` | Time robot physically stops moving. |
| `t_manual_intervention` | Time human intervenes, if applicable. |
| `t_timeout` | Timeout time, if applicable. |

Derived fields:

| Field | Formula |
|---|---|
| `stop_delay` | `t_robot_stopped - t_goal_complete` |
| `judge_delay` | `t_first_vlm_complete - t_goal_complete` |
| `actuation_delay` | `t_robot_stopped - t_stop_command` |
| `post_completion_distance` | distance traveled after `t_goal_complete` |

---

# 8. Completion Judge Output Schema

Use a structured output.

```json
{
  "confidence": 0.0,
  "evidence": "short visual reason",
  "should_stop": true
}
```

Recommended deployment rule:

| Judge Output | Control Action |
|---|---|
| `unsafe` | Stop immediately. |
| `complete`, high confidence | Take final verification frame or require second complete judgment. |
| `complete`, medium confidence | Continue slowly or query again soon. |
| `uncertain` | Continue and query again soon. |
| `incomplete` | Continue. |

---

# 9. Fixed-Rate VLM Sweep

Test during pilot.

| Rate ID | Query Interval | Expected Behavior | Keep for Final? |
|---|---:|---|---|
| F1 | 1 sec | Best stop timing, highest compute cost. | Usually pilot only |
| F2 | 2 sec | Good balance if VLM latency is manageable. | Candidate |
| F3 | 5 sec | Lower compute, possible late stops. | Candidate |
| F4 | 10 sec | Very low compute, likely too late for short tasks. | Pilot only |

Final fixed-rate baseline should use one selected rate, likely 2 sec or 5 sec.

---

# 10. Adaptive Scheduler Variants

Test during pilot, then choose one.

| Scheduler ID | Trigger Set | Description | Expected Use |
|---|---|---|---|
| A1 | Time only | Equivalent to fixed-rate baseline. | Baseline |
| A2 | Time + distance | Query if enough time passed or robot moved far enough. | Simple adaptive candidate |
| A3 | Time + distance + uncertainty | Query sooner after uncertain VLM output. | Recommended candidate |
| A4 | Time + distance + uncertainty + visual change | Query when scene changes significantly. | Optional if visual-change metric is stable |
| A5 | Time + waypoint/action magnitude | Query more often when OmniVLA outputs shrink or become unstable. | Optional if action magnitude is meaningful |

Recommended final adaptive scheduler:

```text
A3: time + distance + uncertainty
```

Example configuration:

```text
base query interval: 5 sec
minimum query interval: 1 sec
distance trigger: 1 m since last VLM call
uncertainty trigger: query again after 1 sec
completion trigger: require final verification
unsafe trigger: stop immediately
```

---

# 11. Baseline Details

## V1: No Judge / Timeout

| Parameter | Recommended Setting |
|---|---|
| Timeout | 30–60 sec depending on task length |
| Stop condition | timeout or manual stop |
| Purpose | Show missing semantic termination |

## V2: Heuristic Stop

Possible heuristics:

| Heuristic | Rule |
|---|---|
| Velocity threshold | Stop if commanded velocity is below threshold for N seconds. |
| Waypoint/action magnitude | Stop if OmniVLA action magnitude is small for N consecutive ticks. |
| Distance traveled | Stop after expected distance for task. |
| No-progress heuristic | Stop if odometry progress is small for N seconds. |

Recommended final heuristic:

```text
velocity/action-magnitude threshold + N-second persistence
```

This is simple, reproducible, and gives reviewers a non-VLM comparison.

## V3: Fixed-Rate VLM Judge

| Parameter | Recommended Setting |
|---|---|
| Query interval | Selected from pilot, likely 2 sec or 5 sec |
| Stop rule | complete + confirmation |
| Purpose | Tests whether VLM judging solves semantic stopping |

## V4: Adaptive VLM Judge

| Parameter | Recommended Setting |
|---|---|
| Scheduler | time + distance + uncertainty |
| Stop rule | complete + confirmation |
| Purpose | Tests query-efficient semantic stopping |

---

# 12. Trial Logging Requirements

Every trial must save the following.

| Category | Required Fields |
|---|---|
| Trial metadata | `trial_id`, date, operator, environment, task, command, method |
| System config | OmniVLA version, VLM model, prompt version, scheduler config, stop thresholds |
| Sensor logs | camera frames, odometry, LiDAR/proximity if available |
| VLA logs | raw OmniVLA outputs, converted commands, command timestamps |
| VLM logs | query timestamps, input image paths, raw responses, parsed status, confidence, evidence |
| Stop logs | stop decision time, stop reason, robot stopped time |
| Human annotation | goal reached, `t_goal_complete`, success/failure, false stop, late stop, notes |
| Safety logs | collision, safety stop, manual intervention, timeout |
| Media | trial video, keyframe images |

---

# 13. Annotation Rules

| Label | Definition |
|---|---|
| `reach_success` | Robot physically reaches the goal region at any point, regardless of stopping. |
| `correct_stop` | Robot stops in the correct region after the task is complete. |
| `false_stop` | Robot stops before the task is complete. |
| `late_stop` | Robot reaches the goal but continues too long or too far before stopping. |
| `timeout_failure` | Trial ends by timeout without correct stop. |
| `manual_intervention` | Human stops or redirects robot. |
| `base_policy_failure` | OmniVLA does not reach the goal. |
| `judge_failure` | OmniVLA reaches the goal, but VLM judge fails to stop correctly. |
| `scheduler_failure` | VLM judge could have stopped correctly, but scheduler queried too late or too rarely. |
| `system_failure` | Failure due to latency, ROS issue, dropped frames, safety stop, or hardware problem. |

Recommended annotation procedure:

| Step | Action |
|---|---|
| 1 | Watch trial video. |
| 2 | Mark `t_goal_complete` if robot reaches goal. |
| 3 | Mark `t_robot_stopped`. |
| 4 | Label stop as correct, false, late, timeout, or manual intervention. |
| 5 | Assign primary failure category. |
| 6 | Save notes and representative keyframes. |

For stronger rigor, have a second person annotate 20–30% of trials.

---

# 14. Main Results Tables to Produce

## Table 1: Overall Performance

| Method | Reach Rate | Correct Stop Rate | False Stop Rate | Late Stop Rate | Timeout Rate | Manual Intervention Rate |
|---|---:|---:|---:|---:|---:|---:|
| V1: No judge / timeout |  |  |  |  |  |  |
| V2: Heuristic stop |  |  |  |  |  |  |
| V3: Fixed-rate VLM judge |  |  |  |  |  |  |
| V4: Adaptive VLM judge |  |  |  |  |  |  |

## Table 2: Judge-Specific Performance

Only include trials where OmniVLA reached the goal.

| Method | Conditional Correct Stop Rate | Mean Stop Delay | Median Stop Delay | Mean Overshoot Distance | False Complete Count |
|---|---:|---:|---:|---:|---:|
| V2: Heuristic stop |  |  |  |  |  |
| V3: Fixed-rate VLM judge |  |  |  |  |  |
| V4: Adaptive VLM judge |  |  |  |  |  |

## Table 3: Compute / Query Efficiency

| Method | Mean VLM Calls / Trial | Mean VLM Latency | Max VLM Latency | Mean Trial Duration | Correct Stop per VLM Call |
|---|---:|---:|---:|---:|---:|
| V3: Fixed-rate VLM judge |  |  |  |  |  |
| V4: Adaptive VLM judge |  |  |  |  |  |

## Table 4: Per-Task Results

| Task | Method | Reach Rate | Correct Stop Rate | False Stop Rate | Mean Stop Delay | Mean VLM Calls |
|---|---|---:|---:|---:|---:|---:|
| T1 Doorway | V1 |  |  |  |  |  |
| T1 Doorway | V2 |  |  |  |  |  |
| T1 Doorway | V3 |  |  |  |  |  |
| T1 Doorway | V4 |  |  |  |  |  |
| T2 Room Entry | V1 |  |  |  |  |  |
| T2 Room Entry | V2 |  |  |  |  |  |
| T2 Room Entry | V3 |  |  |  |  |  |
| T2 Room Entry | V4 |  |  |  |  |  |

---

# 15. Failure Taxonomy Table

| Failure Category | Definition | Example |
|---|---|---|
| Base-policy failure | OmniVLA does not reach goal. | Robot moves toward wrong object. |
| Premature VLM stop | Judge marks complete too early. | Stops halfway to doorway. |
| Missed completion | Judge keeps saying incomplete after goal reached. | Robot reaches backpack but continues. |
| VLM uncertainty failure | Judge remains uncertain until timeout. | Doorway visible but judge never commits. |
| Scheduler late query | VLM was not queried near completion. | Robot passes goal before next judge call. |
| Heuristic false stop | Low-level heuristic stops without semantic completion. | Velocity drops near obstacle, but task incomplete. |
| Visual ambiguity | Camera view insufficient for judgment. | Target partially occluded. |
| Latency/system failure | Delay or dropped frames cause bad stop timing. | VLM response arrives too late. |
| Safety stop | Obstacle/collision risk ends trial. | Robot approaches wall too closely. |

---

# 16. Final Benchmark Recommendation

Use this as the main paper benchmark unless hardware time is severely limited.

| Component | Final Choice |
|---|---|
| Main tasks | T1 doorway, T2 room entry, T3 hallway end, T4 object approach, T5 sign/landmark |
| Main variants | V1 no judge, V2 heuristic stop, V3 fixed-rate VLM judge, V4 adaptive VLM judge |
| Trials | 5 trials per task per variant |
| Total main trials | 100 real-robot trials |
| Offline judge dataset | 100–300 labeled frames/clips |
| Pilot trials | 36–81 real trials depending on how much tuning is needed |
| Sim usage | Integration/safety/debugging only, not primary evidence |
| Real usage | All main benchmark results |
| Primary metric | Conditional correct stop rate |
| Secondary metrics | false stop rate, stop delay, overshoot distance, VLM calls, latency, manual intervention |
| Main comparison | fixed-rate VLM judge vs adaptive VLM judge |
| Main expected result | adaptive judge uses fewer VLM calls with similar stop correctness |

---

# 17. Paper Claim Supported by This Protocol

The intended final claim is:

> Treating OmniVLA-edge as a fixed language-conditioned navigation policy, we evaluate the missing semantic termination layer needed for autonomous deployment. A VLM-based completion judge improves stop correctness over timeout and low-level heuristic baselines, and an adaptive event-triggered judge reduces VLM calls while preserving semantic stopping performance.

---

# 18. Things Not to Spend Time Testing

| Avoid | Reason |
|---|---|
| Many different VLAs | OmniVLA is fixed; your contribution is the judge. |
| Full onboard Jetson deployment before results | Useful engineering, not needed for core contribution. |
| Fine-tuning the VLM early | Prompting/offline evaluation should come first. |
| Vague commands like "check for danger" | Hard to define completion without extra sensors/proxies. |
| Too many scheduler variants in final paper | Makes story messy. Use pilots to choose one. |
| Success-only evaluation | Hides whether failures come from navigation or stopping. |
| Sim-only final benchmark | Weakens claim about physical Go2 deployment. |

---

# 19. Required Files / Artifacts

| Artifact | Path |
|---|---|
| Evaluation protocol | `docs/LOCKED_evaluation_protocol.md` |
| Task definitions | `docs/task_suite.md` |
| Offline labels | `annotations/completion_labels_v0.csv` |
| Main trial annotations | `annotations/main_trial_annotations.csv` |
| Trial runner | `src/trial_runner.py` |
| Completion judge | `src/completion_monitor.py` |
| Fixed scheduler | `src/fixed_rate_completion_scheduler.py` |
| Adaptive scheduler | `src/adaptive_completion_scheduler.py` |
| Stop confirmation | `src/stop_confirmation.py` |
| Baseline scripts | `scripts/run_baseline_no_judge.sh`, `scripts/run_heuristic_stop.sh` |
| VLM scripts | `scripts/run_fixed_vlm_judge.sh`, `scripts/run_adaptive_vlm_judge.sh` |
| Main results | `results/main_results_clean.csv` |
| Failure taxonomy | `results/failure_taxonomy.md` |
| Figures | `figures/` |
| Videos | `videos/` |

---

# 20. Daily Rule During Testing

Only run a trial if it contributes to one of these:

1. Selecting the final VLM judge prompt.
2. Selecting the fixed-rate baseline.
3. Selecting the adaptive scheduler.
4. Measuring missing-stop behavior.
5. Measuring correct-stop behavior.
6. Measuring query efficiency.
7. Explaining failures.

Do not run random demos that cannot be logged, labeled, or used in the paper.
