# LeKiwi Nav Env Codebase Audit Report

## 1. Overview

Comprehensive audit of the LeKiwi Navigation RL Pipeline codebase (25 Python files, 2 JSON catalogs, pipeline documentation). The codebase implements a sim-to-real transfer pipeline for a 3-wheeled omnidirectional (Kiwi drive) mobile robot with a 6-DOF arm, using Isaac Lab/Isaac Sim for RL training (PPO) and behavioral cloning.

**Pipeline flow:**
1. `build_object_catalog.py` -> object catalog
2. `calibrate_real_robot.py` -> real robot measurements
3. `tune_sim_dynamics.py` -> sim dynamics tuning (SysID)
4. `replay_in_sim.py` -> calibration verification
5. `check_calibration_gate.py` -> RMSE quality gate
6. `train_lekiwi.py` (PPO) / `train_bc.py` (BC) -> policy training
7. `collect_demos.py` -> expert demo collection
8. `convert_hdf5_to_lerobot_v3.py` -> VLA dataset conversion
9. `record_teleop.py` / `teleop_dual_logger.py` / `sim_action_receiver_logger.py` -> teleop data collection

---

## 2. Issues Found

### 2.1 [MEDIUM] `record_teleop.py` -- Ros2TeleopSubscriber MRO / super().__init__ problem

**File:** `record_teleop.py:139-143`

```python
class Ros2TeleopSubscriber(Node, TeleopInputBase):
    def __init__(self, arm_topic, wheel_topic, M_inv, wheel_radius):
        super().__init__("teleop_recorder")
```

`Ros2TeleopSubscriber` inherits from both `Node` (ROS2) and `TeleopInputBase`. The `super().__init__("teleop_recorder")` call follows Python MRO, so it calls `Node.__init__("teleop_recorder")`. However, `TeleopInputBase.__init__` is never called. This is not currently a bug because `TeleopInputBase.__init__` is the default `object.__init__`, but it is fragile -- if someone adds initialization logic to `TeleopInputBase`, it will be silently skipped.

### 2.2 [LOW] `leader_to_home_tcp_rest_matched_with_keyboard_base.py` -- Hardcoded paths and IPs

**File:** `leader_to_home_tcp_rest_matched_with_keyboard_base.py:27-29`

```python
DEFAULT_HOME_TAILSCALE_IP = "100.91.14.65"
DEFAULT_HOME_TCP_PORT = 15002
DEFAULT_LEADER_PORT = "COM8"
```

These are developer-specific defaults that will not work on other machines. This is a usability issue, not a bug, since they are overridable via CLI args.

### 2.3 [LOW] `extract_kiwi_geometry_from_usd.py` -- Hardcoded user-specific paths

**File:** `extract_kiwi_geometry_from_usd.py:36-42, 51`

```python
raise RuntimeError(
    "...export LD_LIBRARY_PATH=/home/yubin11/isaacsim/..."
)
# ...
default="/home/yubin11/Downloads/lekiwi_robot.usd"
```

The error message and default robot USD path contain a hardcoded home directory `/home/yubin11/`. This should use `LEKIWI_USD_PATH` environment variable like `lekiwi_robot_cfg.py` does.

### 2.4 [LOW] `lekiwi_robot_cfg.py` -- Hardcoded USD path fallback

**File:** `lekiwi_robot_cfg.py:74`

```python
usd_path=os.environ.get("LEKIWI_USD_PATH", "/home/yubin11/Downloads/lekiwi_robot.usd"),
```

The fallback path `/home/yubin11/Downloads/lekiwi_robot.usd` is user-specific. On any other machine without `LEKIWI_USD_PATH` set, this will fail silently with a confusing USD-not-found error.

### 2.5 [INFO] `sim_real_calibration_test.py` -- Hardcoded geometry constants vs config

**File:** `sim_real_calibration_test.py:75-83`

```python
WHEEL_RADIUS = 0.049
BASE_RADIUS = 0.1085
```

These constants are duplicated from `lekiwi_robot_cfg.py` rather than imported. If `lekiwi_robot_cfg.py` values are updated (e.g., by `extract_kiwi_geometry_from_usd.py --apply_to_cfg`), the calibration test script will be out of sync. This is by design (Script Editor environment can't easily import), but should be noted.

### 2.6 [INFO] `calibration_common.py` -- `align_and_compare` does not handle empty input gracefully

**File:** `calibration_common.py:234-260`

If `real_t` or `sim_t` have length < 2, the function may produce degenerate results (e.g., `t_end=0` leading to `np.linspace(0, 1e-6, n)` interpolation). There's no explicit validation.

### 2.7 [INFO] Observation dimension compatibility between BC and PPO

When training BC on 33D observations and then warm-starting PPO with 37D observations (multi-object mode), `load_bc_into_policy()` in `train_lekiwi.py:138-150` correctly detects the shape mismatch at the first layer and skips it with a warning. This is safe but means the warm-start provides no benefit for the first layer in this cross-dimension scenario.

### 2.8 [INFO] `teleop_dual_logger.py` -- Potential desktop TCP forwarding latency

**File:** `teleop_dual_logger.py:57-58`

```python
parser.add_argument("--connect_timeout_s", type=float, default=0.05)
parser.add_argument("--send_timeout_s", type=float, default=0.01)
```

The very low connect/send timeouts (50ms/10ms) are aggressive. On networks with even slight latency (e.g., Tailscale VPN), frequent reconnection cycles could occur, potentially causing brief data gaps in the sim-side logging.

---

## 3. Pipeline Consistency Analysis

### 3.1 Joint Name / Motor ID Mappings -- CONSISTENT

All files consistently use the same mappings:
- `SIM_JOINT_TO_REAL_MOTOR_ID` in `calibration_common.py` matches across all calibration scripts
- `ARM_JOINT_NAMES` and `WHEEL_JOINT_NAMES` in `lekiwi_robot_cfg.py` are consistently imported
- The 6 arm joint names and 3 wheel joint names are used identically across all 25 files

### 3.2 Wheel Angle Ordering -- CONSISTENT

- `WHEEL_JOINT_NAMES = [axle_2 (FL), axle_1 (FR), axle_0 (Back)]`
- `WHEEL_ANGLES_DEG = [-30.0, -150.0, 90.0]` (same order: FL, FR, Back)
- `sim_real_calibration_test.py` uses a dict keyed by joint name: `{axle_0: 90, axle_1: -150, axle_2: -30}` -- same values, just dict-ordered
- All Kiwi IK computations use the same formula: `M[i] = [cos(theta_i), sin(theta_i), L]`

### 3.3 Geometry Constants -- CONSISTENT

- `WHEEL_RADIUS = 0.049 m` and `BASE_RADIUS = 0.1085 m` are consistent across all files
- `extract_kiwi_geometry_from_usd.py` can update these from USD, and the pipeline supports overrides via `--wheel_radius` / `--base_radius` args

### 3.4 Action Space (9D) -- CONSISTENT

- `action[0:3]` = body velocity command (vx, vy, wz), normalized by max_lin_vel and max_ang_vel
- `action[3:9]` = arm joint position targets (6 joints)
- This 9D format is consistent across: `lekiwi_nav_env.py`, `collect_demos.py`, `record_teleop.py`, `train_lekiwi.py`, `train_bc.py`, `convert_hdf5_to_lerobot_v3.py`

### 3.5 Observation Space (33D/37D) -- CONSISTENT

The 33D base observation layout matches the README documentation exactly:
- `[0:2]` target_xy_body, `[2:4]` object_xy_body, `[4:6]` home_xy_body
- `[6:10]` phase_onehot (4), `[10]` object_visible, `[11]` object_grasped
- `[12:15]` base_lin_vel_body (3), `[15:18]` base_ang_vel_body (3)
- `[18:24]` arm_joint_pos (6), `[24:30]` arm_joint_vel (6), `[30:33]` wheel_vel (3)
- Multi-object adds `[33:36]` bbox_normalized + `[36]` category_normalized = 37D

### 3.6 HDF5 Data Format -- CONSISTENT

`collect_demos.py` output format matches `convert_hdf5_to_lerobot_v3.py` input expectations:
- `episode_k/actions` (T, 9) float32
- `episode_k/robot_state` (T, 9) float32 = arm_pos(6) + wheel_vel(3)
- `episode_k/obs` (T, 33 or 37) float32
- `episode_k/subtask_ids` (T,) int32
- `episode_k/images/base_rgb` and `wrist_rgb` optional

### 3.7 Calibration Pipeline Data Flow -- CONSISTENT

1. `calibrate_real_robot.py` outputs `calibration_latest.json` with:
   - `wheel_radius.encoder_log`, `wheel_radius.command`
   - `base_radius.encoder_log`, `base_radius.command`
   - `arm_sysid.tests` with cmd/pos trajectories
   - `joint_ranges` for arm limits

2. `tune_sim_dynamics.py` reads this and outputs `tuned_dynamics.json` with:
   - `best_params` (damping, friction, armature, etc.)
   - `best_eval` (sequences, mean_rmse_rad)
   - `command_transform` (linear_map, lin_scale, ang_scale, wz_sign)

3. `replay_in_sim.py` can read both calibration JSON and tuned dynamics JSON

4. `check_calibration_gate.py` can verify both replay reports and tuning reports

5. `sim_real_command_transform.py` correctly implements invertible transforms with `INVERSE_LINEAR_MAP`

### 3.8 Command Transform -- CONSISTENT

The `real_to_sim` / `sim_to_real` functions in `sim_real_command_transform.py` correctly handle:
- Linear map application and inversion
- Scale factors with proper division for inverse
- Zero-check with `ValueError` for non-invertible transforms
- The transform is correctly applied in `tune_sim_dynamics.py` and `replay_in_sim.py`

### 3.9 Phase State Machine -- CORRECT

The SEARCH -> APPROACH -> GRASP -> RETURN state machine in `lekiwi_nav_env.py:946-1069` is logically correct:
- SEARCH -> APPROACH: triggered by object detection (visibility)
- APPROACH -> SEARCH: fallback when object visibility is lost
- APPROACH -> GRASP: when object distance < approach_thresh
- GRASP timeout -> APPROACH: for retry mechanism
- GRASP -> RETURN: physics-based (gripper + contact + distance) or proximity-based
- RETURN success: when home_dist < return_thresh while object is grasped

### 3.10 Encoder Unit Handling -- CONSISTENT

All scripts that handle encoder data support the same unit options (`auto`, `rad`, `deg`, `m100`/`m100_100`) and use consistent auto-detection heuristics:
- `calibrate_real_robot.py`, `replay_in_sim.py`, `tune_sim_dynamics.py`, `record_teleop.py`, `sim_action_receiver_logger.py`, `build_arm_limits_real2sim.py`
- `EncoderCalibrationMapper` in `calibration_common.py` correctly converts normalized STS3215 values to radians

---

## 4. Architecture Quality Assessment

### Strengths

1. **Centralized constants**: `lekiwi_robot_cfg.py` and `calibration_common.py` effectively centralize shared constants and utilities, minimizing duplication
2. **Robust fallbacks**: Most scripts handle missing optional inputs gracefully (e.g., auto-detection of encoder units, optional dynamics JSON)
3. **Pipeline gating**: `check_calibration_gate.py` provides automated quality assurance before proceeding to training
4. **Multi-object extensibility**: The 33D -> 37D obs extension is backward compatible
5. **Cross-platform teleop**: Support for both ROS2 and TCP direct modes with automatic fallback
6. **Comprehensive calibration**: The 6-step calibration pipeline (wheel_radius -> base_radius -> arm_sysid -> tune -> replay -> gate) is thorough

### Potential Improvements (not bugs)

1. **Test coverage**: No unit tests exist in the repository. Key functions like `kiwi_ik_np`, `align_and_compare`, `real_to_sim`/`sim_to_real`, `EncoderCalibrationMapper.normalized_to_rad` would benefit from unit tests.
2. **Configuration management**: Multiple hardcoded user-specific paths (`/home/yubin11/...`) should be replaced with environment variables or config files.
3. **Error propagation in `sim_real_calibration_test.py`**: Being a Script Editor script, it uses `async` stepping and `asyncio` patterns that are harder to debug. Error handling in the measurement loop could be more explicit.

---

## 5. Summary

| Category | Count | Severity |
|----------|-------|----------|
| Bugs | 0 | - |
| Medium issues | 1 | MRO/init in record_teleop.py |
| Low issues | 3 | Hardcoded paths |
| Info/notes | 4 | Design observations |

**Overall assessment**: The codebase is well-structured and internally consistent. No critical bugs or pipeline contradictions were found. The main issues are usability-related (hardcoded developer-specific paths) rather than correctness issues. The pipeline data flow from calibration through training to deployment is logically sound, with proper format consistency maintained across all 25 scripts.
