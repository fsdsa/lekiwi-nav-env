#!/usr/bin/env python3
"""
LeKiwi 실측 캘리브레이션 스크립트.

실제 로봇(STS3215 encoder) 데이터를 기반으로 Sim2Real 파라미터를 측정한다.

측정 항목:
  1) wheel_radius: 직진 이동거리 vs 바퀴 각도 변화
  2) base_radius: 제자리 회전각 vs 바퀴 각도 변화
  3) joint_range: 관절 min/max
  3-1) joint_range_single: 단일 관절 min/max (기존 값 유지 + 지정 관절만 갱신)
  4) rest_position: REST 자세 평균
  5) trajectory: replay용 시계열 기록
  6) arm_sysid: 관절별 step/chirp 명령-응답 기록 ({cmd, pos})

Usage:
  # client 모드(원격 host 연결): arm/rest 위주 (all은 arm_sysid 포함)
  python calibrate_real_robot.py --mode all --connection_mode client --remote_ip 192.168.1.46

  # direct 모드(라즈베리파이 로컬): wheel/base 포함 전체 권장 (all은 arm_sysid 포함)
  python calibrate_real_robot.py --mode all --connection_mode direct --robot_port /dev/ttyACM0

  # arm 6축 range만 측정 (duration<=0 이면 Enter로 시작/종료)
  python calibrate_real_robot.py --mode joint_range --connection_mode direct \
    --robot_port /dev/ttyACM0 --joint_range_duration 0

  # 단일 관절 range만 측정 (기존 wheel/base 값 유지)
  python calibrate_real_robot.py --mode joint_range_single --connection_mode direct \
    --robot_port /dev/ttyACM0 --joint_key arm_gripper.pos --joint_range_duration 0

  python calibrate_real_robot.py --mode wheel_radius
  python calibrate_real_robot.py --mode base_radius --encoder_unit rad
  python calibrate_real_robot.py --mode record_trajectory --record_duration 30
  python calibrate_real_robot.py --mode arm_sysid

Notes:
  - calibration_latest.json은 모드별 실행 시 누적 업데이트된다.
    (예: wheel_radius 후 arm_sysid 실행 시 기존 값 유지 + arm_sysid 추가)
  - arm_sysid는 --arm_sysid_sample_hz (기본 50Hz)를 사용한다.
  - joint_range / joint_range_single은 측정 중 arm torque를 자동으로 OFF/ON 처리한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from calibration_common import (
    REAL_ARM_ID_TO_SIM_JOINT,
    extract_motor_id_candidates as _extract_motor_id_candidates,
    infer_sim_wheel_from_real_key as _infer_sim_wheel_from_real_key,
    normalize_key,
)

REAL_ARM_IDS = {
    motor_id
    for motor_id in REAL_ARM_ID_TO_SIM_JOINT
}


# =============================================================================
# Robot adapter
# =============================================================================


def _flatten_numeric_dict(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dict/list into numeric key-value pairs."""
    out: dict[str, float] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric_dict(v, key))
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten_numeric_dict(v, key))
        return out

    if isinstance(obj, (int, float, np.integer, np.floating)):
        out[prefix] = float(obj)

    return out


class RobotAdapter:
    """LeKiwi client wrapper with API fallbacks."""

    def __init__(self, remote_ip: str, client_id: str, connection_mode: str, robot_port: str):
        self.remote_ip = remote_ip
        self.client_id = client_id
        self.connection_mode = connection_mode
        self.robot_port = robot_port
        self.robot = None

    def connect(self):
        if self.connection_mode == "direct":
            self._connect_direct()
            return
        self._connect_client()

    def _connect_client(self):
        errors: list[str] = []
        candidates = [
            ("lerobot.robots.lekiwi", "LeKiwiClient", "LeKiwiClientConfig"),
            ("lerobot.common.robots.lekiwi", "LeKiwiClient", "LeKiwiClientConfig"),
        ]
        for mod_name, cls_name, cfg_name in candidates:
            try:
                module = __import__(mod_name, fromlist=[cls_name, cfg_name])
                Client = getattr(module, cls_name)
                Config = getattr(module, cfg_name)
                cfg = Config(remote_ip=self.remote_ip, id=self.client_id)
                self.robot = Client(cfg)
                self.robot.connect()
                print(f"  Connected via {mod_name}")
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mod_name}: {type(exc).__name__}: {exc}")

        raise RuntimeError(
            "LeKiwiClient import/connect 실패. LeRobot SDK 환경을 확인하세요.\n"
            + "\n".join(errors)
        )

    def _connect_direct(self):
        errors: list[str] = []
        candidates = [
            ("lerobot.robots.lekiwi", "LeKiwi", "LeKiwiConfig"),
            ("lerobot.common.robots.lekiwi", "LeKiwi", "LeKiwiConfig"),
        ]
        for mod_name, cls_name, cfg_name in candidates:
            try:
                module = __import__(mod_name, fromlist=[cls_name, cfg_name])
                RobotCls = getattr(module, cls_name)
                Config = getattr(module, cfg_name)
                cfg = Config(id=self.client_id, port=self.robot_port, cameras={})
                self.robot = RobotCls(cfg)
                # calibration 프롬프트 방지 (기존 calibration 파일 사용)
                self.robot.connect(calibrate=False)
                print(f"  Connected DIRECT via {mod_name} (port={self.robot_port})")
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mod_name}: {type(exc).__name__}: {exc}")

        raise RuntimeError(
            "LeKiwi direct import/connect 실패. 라즈베리파이 로컬 환경/포트를 확인하세요.\n"
            + "\n".join(errors)
        )

    def disconnect(self):
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def set_arm_torque(self, enabled: bool) -> bool:
        """Best-effort torque toggle for arm motors (direct mode only)."""
        if self.robot is None:
            return False
        bus = getattr(self.robot, "bus", None)
        arm_motors = getattr(self.robot, "arm_motors", None)
        if bus is None or not arm_motors:
            return False
        try:
            if enabled:
                bus.enable_torque(arm_motors)
            else:
                bus.disable_torque(arm_motors)
            return True
        except Exception as exc:  # noqa: BLE001
            mode = "enable" if enabled else "disable"
            print(f"  warning: failed to {mode} arm torque: {exc}")
            return False

    def get_observation(self) -> dict[str, float]:
        if self.robot is None:
            return {}

        obs = None
        for name in ("get_observation", "get_observations"):
            fn = getattr(self.robot, name, None)
            if callable(fn):
                obs = fn()
                break
        if obs is None:
            raise RuntimeError("robot observation API를 찾지 못했습니다.")

        if isinstance(obs, dict):
            flat = _flatten_numeric_dict(obs)
            # direct 모드에서는 wheel encoder를 추가 노출한다.
            if self.connection_mode == "direct":
                try:
                    wheel_pos = self.robot.bus.sync_read("Present_Position", self.robot.base_motors)
                    wheel_vel = self.robot.bus.sync_read("Present_Velocity", self.robot.base_motors)
                    for name, val in wheel_pos.items():
                        flat[f"{name}.pos"] = float(val)
                    for name, val in wheel_vel.items():
                        flat[f"{name}.vel_raw"] = float(val)
                except Exception:  # noqa: BLE001
                    pass
            return flat
        return {}

    def send_action(self, action: dict):
        if self.robot is None:
            return

        # LeKiwi send_action은 x.vel / y.vel / theta.vel 키를 항상 기대한다.
        act = dict(action)
        vx = act.get("x.vel", act.get("base.linear_velocity_x.pos", act.get("base.vx", 0.0)))
        vy = act.get("y.vel", act.get("base.linear_velocity_y.pos", act.get("base.vy", 0.0)))
        wz = act.get("theta.vel", act.get("base.angular_velocity_z.pos", act.get("base.wz", 0.0)))
        act["x.vel"] = float(vx)
        act["y.vel"] = float(vy)
        act["theta.vel"] = float(wz)

        # Some LeRobot versions expect arm position targets on every call.
        # Base-only action can trigger empty-motor writes without this fallback.
        try:
            obs = self.get_observation()
            arm_pos_keys: list[str] = []
            for key in _select_arm_keys(obs):
                nk = key.lower()
                if (".vel" in nk) or ("velocity" in nk):
                    continue
                if (".pos" in nk) or ("position" in nk):
                    arm_pos_keys.append(key)
            if arm_pos_keys and not any(k in act for k in arm_pos_keys):
                for key in arm_pos_keys:
                    val = obs.get(key)
                    if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                        act[key] = float(val)
        except Exception:  # noqa: BLE001
            pass

        fn = getattr(self.robot, "send_action", None)
        if callable(fn):
            fn(act)
            return

        fn = getattr(self.robot, "set_action", None)
        if callable(fn):
            fn(act)
            return

        raise RuntimeError("robot action API(send_action/set_action)를 찾지 못했습니다.")


# =============================================================================
# Key extraction
# =============================================================================


def _select_wheel_keys(obs: dict[str, float]) -> list[str]:
    keys = list(obs.keys())

    def _prio(name: str) -> tuple[int, str]:
        n = name.lower()
        if ".pos" in n or "position" in n:
            return (0, n)
        if ".vel" in n or "velocity" in n:
            return (2, n)
        return (1, n)

    # priority 0: infer by known wheel motor IDs / tokens, one key per sim wheel
    key_per_sim: dict[str, str] = {}
    for key in sorted(keys, key=_prio):
        sim_joint = _infer_sim_wheel_from_real_key(key)
        if sim_joint is not None and sim_joint not in key_per_sim:
            key_per_sim[sim_joint] = key
    ordered_sim = ["axle_2_joint", "axle_1_joint", "axle_0_joint"]  # FL, FR, Back
    if all(sj in key_per_sim for sj in ordered_sim):
        return [key_per_sim[sj] for sj in ordered_sim]

    # priority: exact axle names
    exact = [k for k in keys if k in {"axle_0_joint", "axle_1_joint", "axle_2_joint"}]
    if len(exact) >= 3:
        return sorted(exact)

    # fallback patterns
    patterns = ["axle", "wheel", "base.wheel"]
    out = [k for k in keys if any(p in k.lower() for p in patterns)]
    # keep stable ordering
    return sorted(out)[:3]


def _wheel_cos_factor_from_key(key: str) -> float | None:
    """Return |cos(theta_i)| for known Kiwi wheel keys.

    Kiwi wheel geometry:
      axle_2 (front-left):  theta = -30 deg
      axle_1 (front-right): theta = -150 deg
      axle_0 (back):        theta = 90 deg
    """
    sim_joint = _infer_sim_wheel_from_real_key(key)
    if sim_joint == "axle_2_joint":  # FL
        return abs(math.cos(math.radians(-30.0)))
    if sim_joint == "axle_1_joint":  # FR
        return abs(math.cos(math.radians(-150.0)))
    if sim_joint == "axle_0_joint":  # Back
        return abs(math.cos(math.radians(90.0)))
    return None


def _select_arm_keys(obs: dict[str, float]) -> list[str]:
    keys = list(obs.keys())
    arm_name_tokens = ["shoulder", "elbow", "wrist", "gripper", "arm_"]
    out = []
    for k in keys:
        nk = k.lower()
        ids = _extract_motor_id_candidates(k)

        # Semantic arm names are accepted directly.
        if any(tok in nk for tok in arm_name_tokens):
            out.append(k)
            continue

        # STS3215-style generic keys must be filtered by motor-id to avoid wheel(7/8/9).
        if "sts3215" in nk:
            if any(mid in REAL_ARM_IDS for mid in ids):
                out.append(k)
            continue

        # Generic numeric fallback: include only known arm IDs.
        if any(mid in REAL_ARM_IDS for mid in ids):
            out.append(k)
    # keep a stable order and prefer smaller motor-id suffixes when available
    def _arm_key_sort_key(key: str) -> tuple[int, str]:
        ids = _extract_motor_id_candidates(key)
        tail = ids[-1] if ids else 10**9
        return (tail, key)

    return sorted(set(out), key=_arm_key_sort_key)


# =============================================================================
# Unit handling
# =============================================================================


def infer_angle_unit(deltas: list[float], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice
    abs_vals = [abs(v) for v in deltas if np.isfinite(v)]
    if not abs_vals:
        return "rad"
    p95_abs = float(np.percentile(abs_vals, 95))
    if 60.0 <= p95_abs <= 110.0:
        return "m100"
    return "deg" if p95_abs > 20.0 else "rad"


def infer_encoder_unit_from_series(series: dict[str, np.ndarray], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice
    vals_all = []
    for vals in series.values():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals_all.append(arr)
    if not vals_all:
        return "rad"
    all_v = np.concatenate(vals_all)
    p95_abs = float(np.percentile(np.abs(all_v), 95))
    if 60.0 <= p95_abs <= 110.0:
        return "m100"
    return "deg" if p95_abs > 20.0 else "rad"


def to_radians(value: float, unit: str) -> float:
    if unit == "deg":
        return math.radians(value)
    if unit == "m100":
        return value * (math.pi / 100.0)
    return value


def to_unwrapped_radians(values: np.ndarray, unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rad = arr.copy()
    if unit == "deg":
        rad = np.deg2rad(rad)
    elif unit == "m100":
        rad = rad * (np.pi / 100.0)
    return np.unwrap(rad)


def radians_to_unit(value_rad: float, unit: str) -> float:
    if unit == "deg":
        return float(np.rad2deg(value_rad))
    if unit == "m100":
        return float(value_rad * 100.0 / np.pi)
    return float(value_rad)


# =============================================================================
# Measurements
# =============================================================================


def _make_base_action(vx: float, vy: float, wz: float) -> dict[str, float]:
    # LeKiwi canonical body-velocity keys
    return {
        "x.vel": float(vx),
        "y.vel": float(vy),
        "theta.vel": float(wz),
    }


def _stop_base(robot: RobotAdapter):
    action = _make_base_action(0.0, 0.0, 0.0)
    for _ in range(5):
        try:
            robot.send_action(action)
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: failed to send stop action: {exc}")
            break
        time.sleep(0.05)


def _make_arm_action_safe(
    robot: RobotAdapter, joint_key: str, target: float, obs: dict[str, float] | None = None
) -> dict[str, float]:
    """Build safe arm action: keep all current arm joints, update only target joint."""
    cur_obs = obs if isinstance(obs, dict) else robot.get_observation()
    arm_keys = _select_arm_keys(cur_obs)
    action: dict[str, float] = {}
    for k in arm_keys:
        v = cur_obs.get(k)
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            action[k] = float(v)
    action[str(joint_key)] = float(target)
    action["x.vel"] = 0.0
    action["y.vel"] = 0.0
    action["theta.vel"] = 0.0
    return action


def _infer_arm_unit_from_obs(obs: dict[str, float], arm_keys: list[str], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice
    vals = [abs(float(obs.get(k, 0.0))) for k in arm_keys if np.isfinite(obs.get(k, np.nan))]
    if not vals:
        return "rad"
    return "deg" if float(np.median(vals)) > 20.0 else "rad"


def measure_wheel_radius(robot: RobotAdapter, encoder_unit: str, vx_cmd: float, duration: float, sample_hz: float):
    print("\n" + "=" * 60)
    print("  WHEEL RADIUS 측정")
    print("=" * 60)
    print("  로봇 시작 위치를 표시한 뒤 Enter")
    input()

    obs0 = robot.get_observation()
    wheel_keys = _select_wheel_keys(obs0)
    if len(wheel_keys) < 3:
        print(f"  wheel key 탐지 실패: {wheel_keys}")
        if "x.vel" in obs0 and "theta.vel" in obs0:
            print("  현재 observation은 body velocity만 포함하며 wheel encoder가 없습니다.")
            print("  해결: --connection_mode direct 로 라즈베리파이 로컬에서 실행하세요.")
        return None

    start = {k: obs0[k] for k in wheel_keys}
    log = []

    t0 = time.time()
    while time.time() - t0 < duration:
        robot.send_action(_make_base_action(vx_cmd, 0.0, 0.0))
        obs = robot.get_observation()
        snap = {k: float(obs.get(k, np.nan)) for k in wheel_keys}
        log.append({"t": time.time() - t0, "encoders": snap})
        time.sleep(1.0 / sample_hz)

    _stop_base(robot)
    time.sleep(0.3)

    wheel_series_raw: dict[str, np.ndarray] = {}
    for k in wheel_keys:
        vals = [entry.get("encoders", {}).get(k, np.nan) for entry in log]
        wheel_series_raw[k] = np.asarray(vals, dtype=np.float64)

    unit = infer_encoder_unit_from_series(wheel_series_raw, encoder_unit)
    deltas_rad = {}
    deltas = {}
    for k, arr in wheel_series_raw.items():
        arr_u = to_unwrapped_radians(arr, unit)
        if arr_u.size < 2:
            continue
        d_rad = float(arr_u[-1] - arr_u[0])
        deltas_rad[k] = d_rad
        deltas[k] = radians_to_unit(d_rad, unit)

    dist_cm = float(input("  실측 이동 거리(cm): "))
    dist_m = dist_cm / 100.0

    # Kiwi IK 보정:
    #   delta_theta_i = cos(theta_i) * vx * T / r
    #   d = vx * T  =>  r_i = d * |cos(theta_i)| / |delta_theta_i|
    #
    # vx-only 직진에서 back wheel(theta=90deg)는 거의 0이므로 제외해야 한다.
    candidates = []
    skipped = []
    for key, raw_delta in deltas.items():
        delta_rad = abs(float(deltas_rad.get(key, to_radians(raw_delta, unit))))
        cos_factor = _wheel_cos_factor_from_key(key)
        if cos_factor is None:
            skipped.append((key, "unknown_wheel_key"))
            continue
        if cos_factor < 1e-3:
            skipped.append((key, "cos(theta)≈0 (not excited by vx-only)"))
            continue
        if delta_rad < 1e-4:
            skipped.append((key, "encoder_delta_too_small"))
            continue
        r_i = dist_m * cos_factor / delta_rad
        candidates.append(
            {
                "wheel_key": key,
                "encoder_delta_raw": float(raw_delta),
                "encoder_delta_rad": float(delta_rad),
                "abs_cos_theta": float(cos_factor),
                "wheel_radius_m": float(r_i),
            }
        )

    if not candidates:
        print("  유효한 바퀴 샘플이 없습니다. wheel key 매핑/엔코더 단위를 확인하세요.")
        print(f"  wheel keys: {wheel_keys}")
        print(f"  skipped: {skipped}")
        return None

    wheel_radius = float(np.median([c["wheel_radius_m"] for c in candidates]))
    mean_angle_valid = float(np.mean([c["encoder_delta_rad"] for c in candidates]))

    print(f"  unit={unit}, valid_wheels={len(candidates)}, mean_valid_angle={mean_angle_valid:.5f} rad")
    for c in candidates:
        print(
            f"    {c['wheel_key']:<20} "
            f"delta={c['encoder_delta_rad']:.5f}rad "
            f"|cos|={c['abs_cos_theta']:.3f} "
            f"r_i={c['wheel_radius_m']:.6f}m"
        )
    if skipped:
        print("  skipped wheels:")
        for key, reason in skipped:
            print(f"    - {key}: {reason}")
    print(f"  wheel_radius (median) = {wheel_radius:.6f} m")

    return {
        "wheel_radius_m": wheel_radius,
        "encoder_unit": unit,
        "measured_distance_m": dist_m,
        "mean_wheel_angle_rad": mean_angle_valid,
        "encoder_unwrapped": True,
        "command": {
            "vx": float(vx_cmd),
            "vy": 0.0,
            "wz": 0.0,
            "wz_unit": "deg_per_s",
            "duration_s": float(duration),
            "sample_hz": float(sample_hz),
        },
        "wheel_keys": wheel_keys,
        "wheel_deltas": deltas,
        "wheel_radius_candidates": candidates,
        "skipped_wheels": skipped,
        "encoder_log": log,
    }


def measure_base_radius(
    robot: RobotAdapter,
    known_wheel_radius: float | None,
    encoder_unit: str,
    wz_cmd: float,
    duration: float,
    sample_hz: float,
):
    print("\n" + "=" * 60)
    print("  BASE RADIUS 측정")
    print("=" * 60)
    print("  로봇 회전 기준선을 맞춘 뒤 Enter")
    input()

    obs0 = robot.get_observation()
    wheel_keys = _select_wheel_keys(obs0)
    if len(wheel_keys) < 3:
        print(f"  wheel key 탐지 실패: {wheel_keys}")
        if "x.vel" in obs0 and "theta.vel" in obs0:
            print("  현재 observation은 body velocity만 포함하며 wheel encoder가 없습니다.")
            print("  해결: --connection_mode direct 로 라즈베리파이 로컬에서 실행하세요.")
        return None

    start = {k: obs0[k] for k in wheel_keys}
    log = []

    t0 = time.time()
    while time.time() - t0 < duration:
        robot.send_action(_make_base_action(0.0, 0.0, wz_cmd))
        obs = robot.get_observation()
        snap = {k: float(obs.get(k, np.nan)) for k in wheel_keys}
        log.append({"t": time.time() - t0, "encoders": snap})
        time.sleep(1.0 / sample_hz)

    _stop_base(robot)
    time.sleep(0.3)

    wheel_series_raw: dict[str, np.ndarray] = {}
    for k in wheel_keys:
        vals = [entry.get("encoders", {}).get(k, np.nan) for entry in log]
        wheel_series_raw[k] = np.asarray(vals, dtype=np.float64)

    unit = infer_encoder_unit_from_series(wheel_series_raw, encoder_unit)
    deltas = {}
    deltas_rad = {}
    for k, arr in wheel_series_raw.items():
        arr_u = to_unwrapped_radians(arr, unit)
        if arr_u.size < 2:
            continue
        d_rad = float(arr_u[-1] - arr_u[0])
        deltas_rad[k] = d_rad
        deltas[k] = radians_to_unit(d_rad, unit)

    angles = [abs(v) for v in deltas_rad.values()]
    mean_wheel_angle = float(np.mean(angles)) if angles else 0.0

    rot_deg = float(input("  실측 총 회전각(도): "))
    total_rot = math.radians(rot_deg)

    if total_rot < 1e-3 or mean_wheel_angle < 1e-3:
        print("  측정값이 너무 작습니다.")
        return None

    if known_wheel_radius is None:
        ratio = mean_wheel_angle / total_rot
        print(f"  wheel_radius 미제공: L/r = {ratio:.6f}")
        return {
            "ratio_L_over_r": ratio,
            "encoder_unit": unit,
            "encoder_unwrapped": True,
            "total_rotation_rad": total_rot,
            "mean_wheel_angle_rad": mean_wheel_angle,
            "command": {
                "vx": 0.0,
                "vy": 0.0,
                "wz": float(wz_cmd),
                "wz_unit": "deg_per_s",
                "duration_s": float(duration),
                "sample_hz": float(sample_hz),
            },
            "wheel_keys": wheel_keys,
            "wheel_deltas": deltas,
            "encoder_log": log,
        }

    base_radius = known_wheel_radius * mean_wheel_angle / total_rot
    print(f"  unit={unit}, base_radius={base_radius:.6f} m")

    return {
        "base_radius_m": base_radius,
        "encoder_unit": unit,
        "encoder_unwrapped": True,
        "total_rotation_rad": total_rot,
        "mean_wheel_angle_rad": mean_wheel_angle,
        "wheel_radius_used_m": known_wheel_radius,
        "command": {
            "vx": 0.0,
            "vy": 0.0,
            "wz": float(wz_cmd),
            "wz_unit": "deg_per_s",
            "duration_s": float(duration),
            "sample_hz": float(sample_hz),
        },
        "wheel_keys": wheel_keys,
        "wheel_deltas": deltas,
        "encoder_log": log,
    }


def measure_joint_ranges(robot: RobotAdapter, duration: float, sample_hz: float):
    print("\n" + "=" * 60)
    print("  JOINT RANGE 측정")
    print("=" * 60)
    torque_off = robot.set_arm_torque(False)
    if torque_off:
        print("  arm torque OFF (수동으로 관절을 움직일 수 있습니다)")

    print("  Enter를 누르면 기록을 시작합니다.")
    if duration > 0:
        print(f"  기록 시작 후 {duration:.1f}초 동안 각 관절을 최대/최소 끝까지 천천히 왕복하세요.")
    else:
        print("  기록 시작 후 각 관절을 충분히 움직인 다음, 종료 Enter를 눌러 마칩니다.")
    input("  준비되면 Enter: ")

    t0 = time.time()
    frames: list[dict] = []
    try:
        if duration > 0:
            print(f"  recording... ({duration:.1f}s)")
            while time.time() - t0 < duration:
                obs = robot.get_observation()
                frames.append({"t": time.time() - t0, "positions": obs})
                time.sleep(1.0 / sample_hz)
        else:
            stop_evt = threading.Event()

            def _wait_stop():
                input("  종료하려면 Enter: ")
                stop_evt.set()

            threading.Thread(target=_wait_stop, daemon=True).start()
            print("  recording... (manual stop)")
            while not stop_evt.is_set():
                obs = robot.get_observation()
                frames.append({"t": time.time() - t0, "positions": obs})
                time.sleep(1.0 / sample_hz)
    finally:
        if torque_off:
            robot.set_arm_torque(True)
            print("  arm torque ON")

    if not frames:
        return None

    keys = sorted({k for fr in frames for k in fr["positions"].keys()})
    ranges = {}
    for k in keys:
        vals = [fr["positions"].get(k) for fr in frames if k in fr["positions"]]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if not vals:
            continue
        ranges[k] = {"min": float(np.min(vals)), "max": float(np.max(vals))}

    arm_keys = _select_arm_keys(frames[0]["positions"])
    print(f"  arm-like keys: {arm_keys[:12]}")

    return {
        "joint_ranges": ranges,
        "raw_readings": frames,
    }


def measure_single_joint_range(robot: RobotAdapter, joint_key: str, duration: float, sample_hz: float):
    print("\n" + "=" * 60)
    print("  SINGLE JOINT RANGE 측정")
    print("=" * 60)
    print(f"  target joint: {joint_key}")
    torque_off = robot.set_arm_torque(False)
    if torque_off:
        print("  arm torque OFF (수동으로 관절을 움직일 수 있습니다)")

    print("  Enter를 누르면 기록을 시작합니다.")
    if duration > 0:
        print(f"  기록 시작 후 {duration:.1f}초 동안 해당 관절만 최대/최소 끝까지 천천히 왕복하세요.")
    else:
        print("  기록 시작 후 해당 관절을 충분히 움직인 다음, 종료 Enter를 눌러 마칩니다.")
    input("  준비되면 Enter: ")

    t0 = time.time()
    trace: list[dict[str, float]] = []
    try:
        if duration > 0:
            print(f"  recording... ({duration:.1f}s)")
            while time.time() - t0 < duration:
                obs = robot.get_observation()
                val = obs.get(joint_key)
                if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                    trace.append({"t": float(time.time() - t0), "value": float(val)})
                time.sleep(1.0 / sample_hz)
        else:
            stop_evt = threading.Event()

            def _wait_stop():
                input("  종료하려면 Enter: ")
                stop_evt.set()

            threading.Thread(target=_wait_stop, daemon=True).start()
            print("  recording... (manual stop)")
            while not stop_evt.is_set():
                obs = robot.get_observation()
                val = obs.get(joint_key)
                if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                    trace.append({"t": float(time.time() - t0), "value": float(val)})
                time.sleep(1.0 / sample_hz)
    finally:
        if torque_off:
            robot.set_arm_torque(True)
            print("  arm torque ON")

    if not trace:
        print(f"  warning: '{joint_key}' 값을 읽지 못했습니다.")
        return None

    vals = [x["value"] for x in trace]
    rmin = float(np.min(vals))
    rmax = float(np.max(vals))
    print(f"  result: min={rmin:.6f}, max={rmax:.6f} ({len(trace)} samples)")

    return {
        "joint_key": joint_key,
        "range": {"min": rmin, "max": rmax},
        "duration_s": float(duration),
        "sample_hz": float(sample_hz),
        "num_samples": len(trace),
        "trace": trace,
    }


def measure_rest_position(robot: RobotAdapter, samples: int, sample_hz: float):
    print("\n" + "=" * 60)
    print("  REST POSITION 측정")
    print("=" * 60)
    print("  로봇을 REST 자세로 맞춘 뒤 Enter")
    input()

    readings = []
    for _ in range(samples):
        readings.append(robot.get_observation())
        time.sleep(1.0 / sample_hz)

    keys = sorted({k for r in readings for k in r.keys()})
    rest = {}
    for k in keys:
        vals = [r[k] for r in readings if k in r]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if vals:
            rest[k] = float(np.mean(vals))

    return {"rest_positions": rest}


def record_trajectory(robot: RobotAdapter, duration: float, fps: float):
    print("\n" + "=" * 60)
    print("  TRAJECTORY 기록")
    print("=" * 60)
    print("  텔레옵을 시작한 뒤 Enter")
    input()

    traj = []
    t0 = time.time()
    while time.time() - t0 < duration:
        obs = robot.get_observation()
        traj.append({"t": time.time() - t0, "positions": obs})
        time.sleep(1.0 / fps)

    print(f"  recorded frames: {len(traj)}")
    return {
        "trajectory": traj,
        "fps": fps,
        "duration": duration,
        "num_frames": len(traj),
    }


def measure_arm_sysid(
    robot: RobotAdapter,
    encoder_unit: str,
    sample_hz: float,
    step_hold_s: float,
    step_values_deg: list[float],
    chirp_duration_s: float,
    chirp_amp_deg: float,
    chirp_f0_hz: float,
    chirp_f1_hz: float,
):
    print("\n" + "=" * 60)
    print("  ARM SYSID 측정")
    print("=" * 60)
    print("  팔 주변 안전 확인 후 Enter")
    input()

    obs0 = robot.get_observation()
    arm_keys = _select_arm_keys(obs0)
    if not arm_keys:
        print("  arm key 탐지 실패")
        return None

    unit = _infer_arm_unit_from_obs(obs0, arm_keys, encoder_unit)
    to_unit = 1.0 if unit == "deg" else math.pi / 180.0
    step_values = [float(v) * to_unit for v in step_values_deg]
    chirp_amp = float(chirp_amp_deg) * to_unit

    print(f"  arm keys: {arm_keys}")
    print(f"  unit={unit}, step_values({unit})={step_values}, chirp_amp({unit})={chirp_amp:.4f}")

    tests = []
    settle_s = 0.4

    for joint_key in arm_keys:
        base_obs = robot.get_observation()
        if joint_key not in base_obs:
            continue
        base = float(base_obs[joint_key])
        print(f"\n  joint={joint_key} base={base:.5f} {unit}")

        # step tests
        for step_delta in step_values:
            target = base + step_delta
            t0 = time.time()
            t_log, cmd_log, pos_log = [], [], []
            while time.time() - t0 < step_hold_s:
                robot.send_action(_make_arm_action_safe(robot, joint_key, target))
                obs = robot.get_observation()
                t_log.append(float(time.time() - t0))
                cmd_log.append(float(target))
                pos_log.append(float(obs.get(joint_key, np.nan)))
                time.sleep(1.0 / sample_hz)

            tests.append(
                {
                    "joint_key": joint_key,
                    "type": "step",
                    "unit": unit,
                    "base": float(base),
                    "step_delta": float(step_delta),
                    "target": float(target),
                    "sample_hz": float(sample_hz),
                    "t": t_log,
                    "cmd": cmd_log,
                    "pos": pos_log,
                }
            )
            # return to base
            r0 = time.time()
            while time.time() - r0 < settle_s:
                robot.send_action(_make_arm_action_safe(robot, joint_key, base))
                time.sleep(1.0 / sample_hz)

        # chirp test
        t0 = time.time()
        t_log, cmd_log, pos_log = [], [], []
        k = (chirp_f1_hz - chirp_f0_hz) / max(chirp_duration_s, 1e-6)
        while time.time() - t0 < chirp_duration_s:
            tt = float(time.time() - t0)
            phase = 2.0 * math.pi * (chirp_f0_hz * tt + 0.5 * k * tt * tt)
            target = base + chirp_amp * math.sin(phase)
            robot.send_action(_make_arm_action_safe(robot, joint_key, target))
            obs = robot.get_observation()
            t_log.append(tt)
            cmd_log.append(float(target))
            pos_log.append(float(obs.get(joint_key, np.nan)))
            time.sleep(1.0 / sample_hz)

        tests.append(
            {
                "joint_key": joint_key,
                "type": "chirp",
                "unit": unit,
                "base": float(base),
                "chirp_amp": float(chirp_amp),
                "chirp_f0_hz": float(chirp_f0_hz),
                "chirp_f1_hz": float(chirp_f1_hz),
                "sample_hz": float(sample_hz),
                "t": t_log,
                "cmd": cmd_log,
                "pos": pos_log,
            }
        )
        # return to base
        r0 = time.time()
        while time.time() - r0 < settle_s:
            robot.send_action(_make_arm_action_safe(robot, joint_key, base))
            time.sleep(1.0 / sample_hz)

    return {
        "unit": unit,
        "arm_keys": arm_keys,
        "sample_hz": float(sample_hz),
        "step_hold_s": float(step_hold_s),
        "step_values_deg": [float(v) for v in step_values_deg],
        "chirp_duration_s": float(chirp_duration_s),
        "chirp_amp_deg": float(chirp_amp_deg),
        "chirp_f0_hz": float(chirp_f0_hz),
        "chirp_f1_hz": float(chirp_f1_hz),
        "tests": tests,
    }


# =============================================================================
# Save
# =============================================================================


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_results(results: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"calibration_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    # latest 포인터
    latest = out_dir / "calibration_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n  saved: {path}")

    if results.get("wheel_radius"):
        print(f"  WHEEL_RADIUS = {results['wheel_radius']['wheel_radius_m']:.6f}")
    if results.get("base_radius") and results["base_radius"].get("base_radius_m") is not None:
        print(f"  BASE_RADIUS  = {results['base_radius']['base_radius_m']:.6f}")
    if results.get("arm_sysid"):
        num_tests = len(results["arm_sysid"].get("tests", []))
        print(f"  ARM_SYSID tests = {num_tests}")

    return path


def load_latest_results(output_dir: str) -> dict[str, Any]:
    """Load existing calibration_latest.json for incremental update.

    If the file does not exist (or is invalid), returns an empty dict.
    """
    latest = Path(output_dir) / "calibration_latest.json"
    if not latest.is_file():
        return {}
    try:
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: failed to read existing latest calibration: {exc}")
    return {}


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="LeKiwi 실측 캘리브레이션")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "wheel_radius",
            "base_radius",
            "joint_range",
            "joint_range_single",
            "rest_position",
            "record_trajectory",
            "arm_sysid",
        ],
    )
    parser.add_argument("--remote_ip", type=str, default="192.168.1.42")
    parser.add_argument("--client_id", type=str, default="calibration")
    parser.add_argument("--connection_mode", type=str, default="client", choices=["client", "direct"])
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--output_dir", type=str, default="calibration")
    parser.add_argument("--encoder_unit", type=str, default="auto", choices=["auto", "rad", "deg", "m100"])

    parser.add_argument("--vx_cmd", type=float, default=0.15)
    # LeKiwi API의 theta.vel는 deg/s
    parser.add_argument("--wz_cmd", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample_hz", type=float, default=20.0)

    parser.add_argument("--joint_range_duration", type=float, default=10.0)
    parser.add_argument(
        "--joint_key",
        type=str,
        default="arm_gripper.pos",
        help="target joint key for --mode joint_range_single",
    )
    parser.add_argument("--rest_samples", type=int, default=20)

    parser.add_argument("--record_duration", type=float, default=30.0)
    parser.add_argument("--record_fps", type=float, default=25.0)
    parser.add_argument("--arm_step_hold_s", type=float, default=1.0)
    parser.add_argument("--arm_step_values_deg", type=str, default="-10,-5,5,10")
    parser.add_argument("--arm_chirp_duration_s", type=float, default=4.0)
    parser.add_argument("--arm_chirp_amp_deg", type=float, default=8.0)
    parser.add_argument("--arm_chirp_f0_hz", type=float, default=0.2)
    parser.add_argument("--arm_chirp_f1_hz", type=float, default=2.0)
    parser.add_argument(
        "--arm_sysid_sample_hz",
        type=float,
        default=50.0,
        help="sample rate for arm_sysid mode (default: 50Hz)",
    )

    args = parser.parse_args()

    robot = RobotAdapter(
        remote_ip=args.remote_ip,
        client_id=args.client_id,
        connection_mode=args.connection_mode,
        robot_port=args.robot_port,
    )
    robot.connect()

    # Incremental update: preserve previous calibration blocks unless overwritten in this run.
    results: dict[str, Any] = load_latest_results(args.output_dir)
    results.update({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "remote_ip": args.remote_ip,
        "connection_mode": args.connection_mode,
    })

    try:
        if args.mode in ("all", "rest_position"):
            results["rest_position"] = measure_rest_position(
                robot,
                samples=args.rest_samples,
                sample_hz=args.sample_hz,
            )

        if args.mode in ("all", "wheel_radius"):
            results["wheel_radius"] = measure_wheel_radius(
                robot,
                encoder_unit=args.encoder_unit,
                vx_cmd=args.vx_cmd,
                duration=args.duration,
                sample_hz=args.sample_hz,
            )

        if args.mode in ("all", "base_radius"):
            wr = None
            if isinstance(results.get("wheel_radius"), dict):
                wr = results["wheel_radius"].get("wheel_radius_m")
            if wr is None:
                wr_in = input("  wheel_radius(m)을 입력 (없으면 Enter): ").strip()
                wr = float(wr_in) if wr_in else None

            results["base_radius"] = measure_base_radius(
                robot,
                known_wheel_radius=wr,
                encoder_unit=args.encoder_unit,
                wz_cmd=args.wz_cmd,
                duration=args.duration,
                sample_hz=args.sample_hz,
            )

        if args.mode in ("all", "joint_range"):
            results["joint_ranges"] = measure_joint_ranges(
                robot,
                duration=args.joint_range_duration,
                sample_hz=args.sample_hz,
            )

        if args.mode == "joint_range_single":
            single = measure_single_joint_range(
                robot,
                joint_key=args.joint_key,
                duration=args.joint_range_duration,
                sample_hz=args.sample_hz,
            )
            if single is not None:
                joint_ranges_block = results.get("joint_ranges")
                if not isinstance(joint_ranges_block, dict):
                    joint_ranges_block = {}

                ranges_map = joint_ranges_block.get("joint_ranges")
                if not isinstance(ranges_map, dict):
                    ranges_map = {}
                ranges_map[args.joint_key] = single["range"]
                joint_ranges_block["joint_ranges"] = ranges_map

                single_runs = joint_ranges_block.get("single_joint_runs")
                if not isinstance(single_runs, list):
                    single_runs = []
                single_runs.append(single)
                joint_ranges_block["single_joint_runs"] = single_runs

                results["joint_ranges"] = joint_ranges_block

        if args.mode == "record_trajectory":
            results["trajectory"] = record_trajectory(
                robot,
                duration=args.record_duration,
                fps=args.record_fps,
            )

        if args.mode in ("all", "arm_sysid"):
            step_values_deg = []
            for tok in args.arm_step_values_deg.split(","):
                tok = tok.strip()
                if tok:
                    step_values_deg.append(float(tok))
            if not step_values_deg:
                step_values_deg = [-10.0, -5.0, 5.0, 10.0]

            results["arm_sysid"] = measure_arm_sysid(
                robot,
                encoder_unit=args.encoder_unit,
                sample_hz=args.arm_sysid_sample_hz,
                step_hold_s=args.arm_step_hold_s,
                step_values_deg=step_values_deg,
                chirp_duration_s=args.arm_chirp_duration_s,
                chirp_amp_deg=args.arm_chirp_amp_deg,
                chirp_f0_hz=args.arm_chirp_f0_hz,
                chirp_f1_hz=args.arm_chirp_f1_hz,
            )

        save_results(results, args.output_dir)

    except KeyboardInterrupt:
        print("\n  interrupted")
        _stop_base(robot)
    finally:
        _stop_base(robot)
        robot.disconnect()
        print("  disconnected")


if __name__ == "__main__":
    main()
