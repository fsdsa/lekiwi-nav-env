"""Shared calibration helpers for LeKiwi scripts.

This module centralizes constants and small utilities that were duplicated
across calibrate/replay/tune/arm-limit tools.
"""

from __future__ import annotations

import re

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
REAL_ARM_ID_TO_SIM_JOINT = {
    motor_id: sim_joint
    for sim_joint, motor_id in SIM_JOINT_TO_REAL_MOTOR_ID.items()
    if sim_joint.startswith("STS3215_")
}

REAL_ARM_KEY_TO_SIM_JOINT = {
    "arm_shoulder_pan.pos": "STS3215_03a_v1_Revolute_45",
    "arm_shoulder_lift.pos": "STS3215_03a_v1_1_Revolute_49",
    "arm_elbow_flex.pos": "STS3215_03a_v1_2_Revolute_51",
    "arm_wrist_flex.pos": "STS3215_03a_v1_3_Revolute_53",
    "arm_wrist_roll.pos": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "arm_gripper.pos": "STS3215_03a_v1_4_Revolute_57",
}
SIM_JOINT_TO_REAL_ARM_KEY = {
    sim_joint: real_key for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items()
}
REAL_ARM_KEY_TO_MOTOR_NAME = {
    "arm_shoulder_pan.pos": "arm_shoulder_pan",
    "arm_shoulder_lift.pos": "arm_shoulder_lift",
    "arm_elbow_flex.pos": "arm_elbow_flex",
    "arm_wrist_flex.pos": "arm_wrist_flex",
    "arm_wrist_roll.pos": "arm_wrist_roll",
    "arm_gripper.pos": "arm_gripper",
}

_ARM_TOKEN_TO_SIM_JOINT = {
    "armshoulderpan": "STS3215_03a_v1_Revolute_45",
    "armshoulderlift": "STS3215_03a_v1_1_Revolute_49",
    "armelbowflex": "STS3215_03a_v1_2_Revolute_51",
    "armwristflex": "STS3215_03a_v1_3_Revolute_53",
    "armwristroll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "armgripper": "STS3215_03a_v1_4_Revolute_57",
    "shoulderpan": "STS3215_03a_v1_Revolute_45",
    "shoulderlift": "STS3215_03a_v1_1_Revolute_49",
    "elbow": "STS3215_03a_v1_2_Revolute_51",
    "wristflex": "STS3215_03a_v1_3_Revolute_53",
    "wristroll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "gripper": "STS3215_03a_v1_4_Revolute_57",
}


def normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def extract_motor_id_candidates(key: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", str(key))]


def is_gripper_key(key: str) -> bool:
    return "gripper" in normalize_key(key)


class EncoderCalibrationMapper:
    """Convert LeRobot normalized STS3215 commands (m100_100/0_100) into radians."""

    def __init__(self, by_motor_name: dict[str, dict[str, float]], ticks_per_turn: int = 4096):
        self.by_motor_name = by_motor_name
        self.by_motor_id: dict[int, dict[str, float]] = {}
        for cfg in by_motor_name.values():
            mid = int(cfg.get("id", -1))
            if mid > 0:
                self.by_motor_id[mid] = cfg
        self.ticks_per_turn = int(ticks_per_turn)

    @classmethod
    def from_payload(cls, payload: dict) -> "EncoderCalibrationMapper | None":
        if not isinstance(payload, dict):
            return None
        by_motor_name: dict[str, dict[str, float]] = {}
        for motor_name, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            try:
                motor_id = int(raw.get("id"))
                drive_mode = int(raw.get("drive_mode", 0))
                range_min = float(raw.get("range_min"))
                range_max = float(raw.get("range_max"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(range_min) or not np.isfinite(range_max):
                continue
            if abs(range_max - range_min) < 1e-9:
                continue
            by_motor_name[str(motor_name)] = {
                "id": float(motor_id),
                "drive_mode": float(drive_mode),
                "range_min": float(range_min),
                "range_max": float(range_max),
            }
        if not by_motor_name:
            return None
        return cls(by_motor_name=by_motor_name)

    def _get_cfg_for_real_key(self, real_key: str) -> dict[str, float] | None:
        motor_name = REAL_ARM_KEY_TO_MOTOR_NAME.get(real_key)
        if motor_name and motor_name in self.by_motor_name:
            return self.by_motor_name[motor_name]
        sim_joint = REAL_ARM_KEY_TO_SIM_JOINT.get(real_key)
        if sim_joint is None:
            return None
        motor_id = SIM_JOINT_TO_REAL_MOTOR_ID.get(sim_joint)
        if motor_id is None:
            return None
        return self.by_motor_id.get(int(motor_id))

    def _raw_to_rad(self, raw_pos: float, cfg: dict[str, float]) -> float:
        range_min = float(cfg["range_min"])
        range_max = float(cfg["range_max"])
        mid = 0.5 * (range_min + range_max)
        max_res = max(1.0, float(self.ticks_per_turn - 1))
        deg = (float(raw_pos) - mid) * 360.0 / max_res
        return float(np.deg2rad(deg))

    def normalized_to_rad(self, real_key: str, value: float) -> float | None:
        cfg = self._get_cfg_for_real_key(real_key)
        if cfg is None:
            return None

        range_min = float(cfg["range_min"])
        range_max = float(cfg["range_max"])
        drive_mode = int(cfg["drive_mode"]) != 0
        lo = min(range_min, range_max)
        hi = max(range_min, range_max)

        if is_gripper_key(real_key):
            v = float(np.clip(value, 0.0, 100.0))
            if drive_mode:
                v = 100.0 - v
            raw = (v / 100.0) * (hi - lo) + lo
        else:
            v = float(np.clip(value, -100.0, 100.0))
            if drive_mode:
                v = -v
            raw = ((v + 100.0) / 200.0) * (hi - lo) + lo

        return self._raw_to_rad(raw, cfg)

    def limits_rad_for_real_key(self, real_key: str) -> tuple[float, float] | None:
        if is_gripper_key(real_key):
            low = self.normalized_to_rad(real_key, 0.0)
            high = self.normalized_to_rad(real_key, 100.0)
        else:
            low = self.normalized_to_rad(real_key, -100.0)
            high = self.normalized_to_rad(real_key, 100.0)
        if low is None or high is None:
            return None
        lo = float(min(low, high))
        hi = float(max(low, high))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
            return None
        return (lo, hi)


def infer_sim_wheel_from_real_key(key: str) -> str | None:
    nk = normalize_key(key)
    if "axle2" in nk or "frontleft" in nk or "baseleftwheel" in nk or "leftwheel" in nk:
        return "axle_2_joint"
    if "axle1" in nk or "frontright" in nk or "baserightwheel" in nk or "rightwheel" in nk:
        return "axle_1_joint"
    if "axle0" in nk or "back" in nk or "rear" in nk or "basebackwheel" in nk:
        return "axle_0_joint"
    for motor_id in reversed(extract_motor_id_candidates(key)):
        if motor_id in REAL_WHEEL_ID_TO_SIM_JOINT:
            return REAL_WHEEL_ID_TO_SIM_JOINT[motor_id]
    return None


def infer_sim_arm_from_real_key(key: str) -> str | None:
    nk = normalize_key(key)
    for token, sim_joint in _ARM_TOKEN_TO_SIM_JOINT.items():
        if token in nk:
            return sim_joint
    if "joint0" in nk or "arm0" in nk:
        return "STS3215_03a_v1_Revolute_45"
    if "joint1" in nk or "arm1" in nk:
        return "STS3215_03a_v1_1_Revolute_49"
    if "joint2" in nk or "arm2" in nk:
        return "STS3215_03a_v1_2_Revolute_51"
    if "joint3" in nk or "arm3" in nk:
        return "STS3215_03a_v1_3_Revolute_53"
    if "joint4" in nk or "arm4" in nk:
        return "STS3215_03a_Wrist_Roll_v1_Revolute_55"
    if "joint5" in nk or "arm5" in nk:
        return "STS3215_03a_v1_4_Revolute_57"
    for motor_id in reversed(extract_motor_id_candidates(key)):
        if motor_id in REAL_ARM_ID_TO_SIM_JOINT:
            return REAL_ARM_ID_TO_SIM_JOINT[motor_id]
    return None


def kiwi_ik_np(
    vx: float,
    vy: float,
    wz: float,
    wheel_radius: float,
    base_radius: float,
    wheel_angles_rad: list[float] | tuple[float, ...],
) -> np.ndarray:
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in wheel_angles_rad], dtype=np.float64)
    return m.dot(np.array([vx, vy, wz], dtype=np.float64)) / max(float(wheel_radius), 1e-6)


def align_and_compare(
    real_t: np.ndarray,
    real_series: np.ndarray,
    sim_t: np.ndarray,
    sim_series: np.ndarray,
) -> dict:
    real_t = np.asarray(real_t, dtype=np.float64).reshape(-1)
    real_series = np.asarray(real_series, dtype=np.float64).reshape(-1)
    sim_t = np.asarray(sim_t, dtype=np.float64).reshape(-1)
    sim_series = np.asarray(sim_series, dtype=np.float64).reshape(-1)

    if real_t.size != real_series.size:
        raise ValueError(f"real_t/real_series size mismatch: {real_t.size} vs {real_series.size}")
    if sim_t.size != sim_series.size:
        raise ValueError(f"sim_t/sim_series size mismatch: {sim_t.size} vs {sim_series.size}")

    if real_t.size < 2 or sim_t.size < 2:
        raise ValueError("align_and_compare requires at least 2 samples for both real and sim series.")

    if (
        not np.all(np.isfinite(real_t))
        or not np.all(np.isfinite(real_series))
        or not np.all(np.isfinite(sim_t))
        or not np.all(np.isfinite(sim_series))
    ):
        raise ValueError("align_and_compare received non-finite values.")

    # np.interp는 x가 단조 증가해야 하므로 정렬 및 중복 타임스탬프 제거.
    real_order = np.argsort(real_t)
    real_t = real_t[real_order]
    real_series = real_series[real_order]
    sim_order = np.argsort(sim_t)
    sim_t = sim_t[sim_order]
    sim_series = sim_series[sim_order]

    real_keep = np.concatenate(([True], np.diff(real_t) > 1e-9))
    sim_keep = np.concatenate(([True], np.diff(sim_t) > 1e-9))
    real_t = real_t[real_keep]
    real_series = real_series[real_keep]
    sim_t = sim_t[sim_keep]
    sim_series = sim_series[sim_keep]

    if real_t.size < 2 or sim_t.size < 2:
        raise ValueError("align_and_compare needs at least 2 unique timestamps for both real and sim.")

    t_end = min(float(real_t[-1]), float(sim_t[-1]))
    n = max(5, min(len(real_t), len(sim_t)))
    t_common = np.linspace(0.0, max(t_end, 1e-6), n)

    rr = np.interp(t_common, real_t, real_series)
    ss = np.interp(t_common, sim_t, sim_series)

    rr = np.unwrap(rr)
    ss = np.unwrap(ss)
    rr = rr - rr[0]
    ss = ss - ss[0]

    err = ss - rr
    return {
        "time_s": t_common,
        "real_delta_rad": rr,
        "sim_delta_rad": ss,
        "mae_rad": float(np.mean(np.abs(err))),
        "rmse_rad": float(np.sqrt(np.mean(err**2))),
        "max_err_rad": float(np.max(np.abs(err))),
    }


def align_and_compare_best_polarity(
    real_t: np.ndarray,
    real_series: np.ndarray,
    sim_t: np.ndarray,
    sim_series: np.ndarray,
) -> dict:
    aligned = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=sim_series)
    aligned["sim_polarity"] = 1
    flipped = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=-sim_series)
    flipped["sim_polarity"] = -1
    return flipped if flipped["rmse_rad"] < aligned["rmse_rad"] else aligned
