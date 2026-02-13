#!/usr/bin/env python3
"""
LeKiwi 실측 캘리브레이션 스크립트.

실제 로봇(STS3215 encoder) 데이터를 기반으로 Sim2Real 파라미터를 측정한다.

측정 항목:
  1) wheel_radius: 직진 이동거리 vs 바퀴 각도 변화
  2) base_radius: 제자리 회전각 vs 바퀴 각도 변화
  3) joint_range: 관절 min/max
  4) rest_position: REST 자세 평균
  5) trajectory: replay용 시계열 기록
  6) arm_sysid: 관절별 step/chirp 명령-응답 기록 ({cmd, pos})

Usage:
  # client 모드(원격 host 연결): arm/rest 위주
  python calibrate_real_robot.py --mode all --connection_mode client --remote_ip 192.168.1.46

  # direct 모드(라즈베리파이 로컬): wheel/base 포함 전체 권장
  python calibrate_real_robot.py --mode all --connection_mode direct --robot_port /dev/ttyACM0

  python calibrate_real_robot.py --mode wheel_radius
  python calibrate_real_robot.py --mode base_radius --encoder_unit rad
  python calibrate_real_robot.py --mode record_trajectory --record_duration 30
  python calibrate_real_robot.py --mode arm_sysid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

# Canonical mapping (left: sim joint name, right: real motor ID)
SIM_JOINT_TO_REAL_MOTOR_ID = {
    "axle_0_joint": 8,
    "axle_1_joint": 9,
    "axle_2_joint": 7,
    "STS3215_03a_v1_4_Revolute_57": 6,
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": 5,
    "STS3215_03a_v1_3_Revolute_53": 4,
    "STS3215_03a_v1_2_Revolute_51": 3,
    "STS3215_03a_v1_1_Revolute_49": 2,
    "STS3215_03a_v1_Revolute_45": 1,
}
REAL_WHEEL_ID_TO_SIM_JOINT = {
    motor_id: sim_joint
    for sim_joint, motor_id in SIM_JOINT_TO_REAL_MOTOR_ID.items()
    if sim_joint.startswith("axle_")
}
REAL_ARM_IDS = {
    motor_id
    for sim_joint, motor_id in SIM_JOINT_TO_REAL_MOTOR_ID.items()
    if sim_joint.startswith("STS3215_")
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
        last_exc = None
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
                last_exc = exc

        raise RuntimeError(
            "LeKiwiClient import/connect 실패. LeRobot SDK 환경을 확인하세요.\n"
            f"last error: {last_exc}"
        )

    def _connect_direct(self):
        last_exc = None
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
                last_exc = exc

        raise RuntimeError(
            "LeKiwi direct import/connect 실패. 라즈베리파이 로컬 환경/포트를 확인하세요.\n"
            f"last error: {last_exc}"
        )

    def disconnect(self):
        if self.robot is None:
            return
        try:
            self.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

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


def normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _extract_motor_id_candidates(key: str) -> list[int]:
    nums = re.findall(r"\d+", key)
    return [int(n) for n in nums]


def _infer_sim_wheel_from_real_key(key: str) -> str | None:
    """Infer sim wheel joint from real observation key."""
    k = normalize_key(key)

    # 1) explicit axle names
    if "axle2" in k:
        return "axle_2_joint"
    if "axle1" in k:
        return "axle_1_joint"
    if "axle0" in k:
        return "axle_0_joint"

    # 2) semantic tokens
    if "frontleft" in k:
        return "axle_2_joint"
    if "frontright" in k:
        return "axle_1_joint"
    if "back" in k or "rear" in k:
        return "axle_0_joint"
    if "baseleftwheel" in k or "leftwheel" in k:
        return "axle_2_joint"
    if "baserightwheel" in k or "rightwheel" in k:
        return "axle_1_joint"
    if "basebackwheel" in k:
        return "axle_0_joint"

    # 3) motor id mapping (7/8/9)
    ids = _extract_motor_id_candidates(key)
    for mid in reversed(ids):  # prefer tail id (e.g., STS3215_7 -> 7)
        if mid in REAL_WHEEL_ID_TO_SIM_JOINT:
            return REAL_WHEEL_ID_TO_SIM_JOINT[mid]

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
    if user_choice in ("rad", "deg"):
        return user_choice
    abs_vals = [abs(v) for v in deltas if np.isfinite(v)]
    if not abs_vals:
        return "rad"
    median = float(np.median(abs_vals))
    # 직진/회전 짧은 측정에서 deg는 보통 수십~수백, rad는 대개 0~20
    return "deg" if median > 20.0 else "rad"


def to_radians(value: float, unit: str) -> float:
    if unit == "deg":
        return math.radians(value)
    return value


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
        robot.send_action(action)
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
    if user_choice in ("rad", "deg"):
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

    obs1 = robot.get_observation()
    end = {k: obs1.get(k, start[k]) for k in wheel_keys}

    deltas = {k: float(end[k] - start[k]) for k in wheel_keys}
    unit = infer_angle_unit(list(deltas.values()), encoder_unit)

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
        delta_rad = abs(to_radians(raw_delta, unit))
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
        "command": {
            "vx": float(vx_cmd),
            "vy": 0.0,
            "wz": 0.0,
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

    obs1 = robot.get_observation()
    end = {k: obs1.get(k, start[k]) for k in wheel_keys}
    deltas = {k: float(end[k] - start[k]) for k in wheel_keys}

    unit = infer_angle_unit(list(deltas.values()), encoder_unit)
    angles = [abs(to_radians(v, unit)) for v in deltas.values()]
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
            "total_rotation_rad": total_rot,
            "mean_wheel_angle_rad": mean_wheel_angle,
            "command": {
                "vx": 0.0,
                "vy": 0.0,
                "wz": float(wz_cmd),
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
        "total_rotation_rad": total_rot,
        "mean_wheel_angle_rad": mean_wheel_angle,
        "wheel_radius_used_m": known_wheel_radius,
        "command": {
            "vx": 0.0,
            "vy": 0.0,
            "wz": float(wz_cmd),
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
    print("  관절을 최대/최소까지 천천히 움직인 뒤 Enter")
    input()

    t0 = time.time()
    frames: list[dict] = []

    while time.time() - t0 < duration:
        obs = robot.get_observation()
        frames.append({"t": time.time() - t0, "positions": obs})
        time.sleep(1.0 / sample_hz)

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


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="LeKiwi 실측 캘리브레이션")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "wheel_radius", "base_radius", "joint_range", "rest_position", "record_trajectory", "arm_sysid"],
    )
    parser.add_argument("--remote_ip", type=str, default="192.168.1.42")
    parser.add_argument("--client_id", type=str, default="calibration")
    parser.add_argument("--connection_mode", type=str, default="client", choices=["client", "direct"])
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--output_dir", type=str, default="calibration")
    parser.add_argument("--encoder_unit", type=str, default="auto", choices=["auto", "rad", "deg"])

    parser.add_argument("--vx_cmd", type=float, default=0.15)
    # LeKiwi API의 theta.vel는 deg/s
    parser.add_argument("--wz_cmd", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample_hz", type=float, default=20.0)

    parser.add_argument("--joint_range_duration", type=float, default=10.0)
    parser.add_argument("--rest_samples", type=int, default=20)

    parser.add_argument("--record_duration", type=float, default=30.0)
    parser.add_argument("--record_fps", type=float, default=25.0)
    parser.add_argument("--arm_step_hold_s", type=float, default=1.0)
    parser.add_argument("--arm_step_values_deg", type=str, default="-10,-5,5,10")
    parser.add_argument("--arm_chirp_duration_s", type=float, default=4.0)
    parser.add_argument("--arm_chirp_amp_deg", type=float, default=8.0)
    parser.add_argument("--arm_chirp_f0_hz", type=float, default=0.2)
    parser.add_argument("--arm_chirp_f1_hz", type=float, default=2.0)

    args = parser.parse_args()

    robot = RobotAdapter(
        remote_ip=args.remote_ip,
        client_id=args.client_id,
        connection_mode=args.connection_mode,
        robot_port=args.robot_port,
    )
    robot.connect()

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "remote_ip": args.remote_ip,
        "connection_mode": args.connection_mode,
    }

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

        if args.mode == "record_trajectory":
            results["trajectory"] = record_trajectory(
                robot,
                duration=args.record_duration,
                fps=args.record_fps,
            )

        if args.mode == "arm_sysid":
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
                sample_hz=args.sample_hz,
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
