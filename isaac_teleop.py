#!/usr/bin/env python3
# leader_to_home_tcp_rest_matched_with_keyboard_base.py
# Windows: SO100 Leader -> TCP(JSON lines) to Home
#  - arm: name/position (same as before)
#  - base: {"vx","vy","wz"}  (key-state based, sent EVERY tick)

import json
import math
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

# ---------- keyboard (pynput) ----------
try:
    from pynput import keyboard
    PYNPUT_OK = True
except Exception:
    keyboard = None
    PYNPUT_OK = False

FPS = 20

HOME_TAILSCALE_IP = "100.91.14.65"
HOME_TCP_PORT = 15002

JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]

# Measured: Isaac Sim joint positions at Play with no input
SIM_REST_RAD6 = [
    -0.001634,   # shoulder_pan
    -0.002328,   # shoulder_lift
    0.098572,    # elbow_flex
    0.004954,    # wrist_flex
    0.009319,    # wrist_roll
    -0.000285,   # gripper
]

# Measured: Leader arm positions when matching Isaac Sim rest pose
LEADER_REST_RAD6 = [
    0.042740,    # shoulder_pan (2.45 deg)
    -1.521986,   # shoulder_lift (-87.20 deg)
    1.739000,    # elbow_flex (99.64 deg)
    1.201714,    # wrist_flex (68.85 deg)
    -0.926154,   # wrist_roll (-53.06 deg)
    0.215753,    # gripper (18.09 deg)
]

SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

SMOOTH_ALPHA = 0.12
MAX_SPEED_RAD_S = 2.0
RAMP_SECONDS = 0.8

# ---------- base keymap ----------
LINEAR_SPEED = 0.25     # m/s
ANGULAR_SPEED = 2.0     # rad/s
SPEED_STEP = 0.02       # m/s step
ANG_STEP = 0.5          # rad/s step
MIN_LINEAR = 0.02
MIN_ANG = 0.5
MAX_LINEAR = 0.5
MAX_ANG = 10.0

pressed: Dict[str, bool] = {}
prev_pressed: Dict[str, bool] = {}

def _norm_key_char(k) -> Optional[str]:
    try:
        if hasattr(k, "char") and k.char is not None:
            return str(k.char).lower()
    except Exception:
        pass
    return None

def on_press(k):
    kc = _norm_key_char(k)
    if kc:
        pressed[kc] = True

def on_release(k):
    kc = _norm_key_char(k)
    if kc:
        pressed[kc] = False

def _connect_tcp() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((HOME_TAILSCALE_IP, HOME_TCP_PORT))
    s.settimeout(None)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s

def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def extract_leader_deg(raw_action: Any) -> List[float]:
    deg6 = [0.0] * 6
    if isinstance(raw_action, dict):
        def pick(name: str) -> float:
            for k, v in raw_action.items():
                k2 = str(k).strip().lower().replace(" ", "_").replace("-", "_")
                if k2.startswith("arm_"):
                    k2 = k2[4:]
                if k2.endswith(".pos"):
                    k2 = k2[:-4]
                if k2 == name:
                    try:
                        return float(v)
                    except Exception:
                        return 0.0
            return 0.0

        deg6[0] = pick("shoulder_pan")
        deg6[1] = pick("shoulder_lift")
        deg6[2] = pick("elbow_flex")
        deg6[3] = pick("wrist_flex")
        deg6[4] = pick("wrist_roll")
        deg6[5] = pick("gripper")
    return deg6

def leader_deg_to_rad6(deg6: List[float]) -> List[float]:
    rad6 = [math.radians(x) for x in deg6]
    rad6[0] = wrap_pi(rad6[0])
    return rad6

def rate_limit(cur: float, prev: float, max_speed: float, dt: float) -> float:
    if dt <= 0:
        return cur
    max_step = max_speed * dt
    d = cur - prev
    if d > max_step:
        return prev + max_step
    if d < -max_step:
        return prev - max_step
    return cur

def compute_base_cmd() -> Dict[str, float]:
    global LINEAR_SPEED, ANGULAR_SPEED, prev_pressed

    active = {k for k, v in pressed.items() if v}

    def rising(k: str) -> bool:
        return (k in active) and (not prev_pressed.get(k, False))

    if rising("r"):
        LINEAR_SPEED = min(MAX_LINEAR, LINEAR_SPEED + SPEED_STEP)
        ANGULAR_SPEED = min(MAX_ANG, ANGULAR_SPEED + ANG_STEP)
        print(f"[BASE] speed up -> linear={LINEAR_SPEED:.3f} m/s, ang={ANGULAR_SPEED:.3f} rad/s")
    if rising("f"):
        LINEAR_SPEED = max(MIN_LINEAR, LINEAR_SPEED - SPEED_STEP)
        ANGULAR_SPEED = max(MIN_ANG, ANGULAR_SPEED - ANG_STEP)
        print(f"[BASE] speed down -> linear={LINEAR_SPEED:.3f} m/s, ang={ANGULAR_SPEED:.3f} rad/s")

    prev_pressed = {k: (k in active) for k in ["w","a","s","d","z","x","r","f"]}

    vx = vy = wz = 0.0
    if "w" in active: vx += LINEAR_SPEED
    if "s" in active: vx -= LINEAR_SPEED
    if "a" in active: vy += LINEAR_SPEED
    if "d" in active: vy -= LINEAR_SPEED
    if "z" in active: wz += ANGULAR_SPEED
    if "x" in active: wz -= ANGULAR_SPEED

    return {"vx": float(vx), "vy": float(vy), "wz": float(wz)}

def main():
    leader = SO100Leader(SO100LeaderConfig(port="COM8", id="my_awesome_leader_arm"))
    leader.connect()
    if not leader.is_connected:
        raise RuntimeError("Leader arm not connected")

    if PYNPUT_OK:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("[KEY] pynput listener started")
    else:
        print("[KEY] pynput NOT available. base cmd will stay zero.")

    print(f"Target TCP = {HOME_TAILSCALE_IP}:{HOME_TCP_PORT}")
    print("SIM_REST_RAD6    =", [round(x, 6) for x in SIM_REST_RAD6])
    print("LEADER_REST_RAD6 =", [round(x, 6) for x in LEADER_REST_RAD6])
    print("SIGNS            =", SIGNS)

    sock: Optional[socket.socket] = None
    last_out: Optional[List[float]] = None
    last_t = time.time()
    t_stream_start: Optional[float] = None

    while True:
        if sock is None:
            try:
                sock = _connect_tcp()
                t_stream_start = time.time()
                print("CONNECTED")
            except Exception as e:
                print("connect failed:", e)
                time.sleep(1.0)
                continue

        tick_start = time.perf_counter()

        # ----- arm -----
        deg6 = extract_leader_deg(leader.get_action())
        leader_rad6 = leader_deg_to_rad6(deg6)

        delta = [leader_rad6[i] - LEADER_REST_RAD6[i] for i in range(6)]
        target = [SIM_REST_RAD6[i] + SIGNS[i] * delta[i] for i in range(6)]

        if t_stream_start is not None and RAMP_SECONDS > 0:
            a = min(max((time.time() - t_stream_start) / RAMP_SECONDS, 0.0), 1.0)
            target = [SIM_REST_RAD6[i] + a * (target[i] - SIM_REST_RAD6[i]) for i in range(6)]

        now = time.time()
        dt = now - last_t
        last_t = now

        if last_out is None:
            out = list(target)
        else:
            tmp = [rate_limit(target[i], last_out[i], MAX_SPEED_RAD_S, dt) for i in range(6)]
            a = SMOOTH_ALPHA
            out = [(1 - a) * last_out[i] + a * tmp[i] for i in range(6)]
        last_out = list(out)

        # ----- base -----
        base_cmd = compute_base_cmd() if PYNPUT_OK else {"vx": 0.0, "vy": 0.0, "wz": 0.0}

        payload = {
            "t": time.time(),
            "name": JOINT_NAMES,
            "position": out,
            "base": base_cmd,
            "src": "so100_leader+keyboard",
        }

        try:
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except Exception as e:
            print("send failed:", e)
            try:
                sock.close()
            except Exception:
                pass
            sock = None
            continue

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - tick_start), 0.0))

if __name__ == "__main__":
    main()
