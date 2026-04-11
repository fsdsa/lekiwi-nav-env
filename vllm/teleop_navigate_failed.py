#!/usr/bin/env python3
"""
⚠️ 레거시: direction-based navigate 실패 보충용. 목적지 기반 navigate 전환 후 폐기.

Navigate 실패 항목 IK 직접 명령 수집.
팔은 고정 (open/tucked pose), base는 IK로 자동 이동.

Usage:
    PYTHONUNBUFFERED=1 python vllm/teleop_navigate_failed.py \
      --scene_idx 1302 --scene_scale 0.6
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--scene_idx", type=int, default=1302)
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")
parser.add_argument("--scene_scale", type=float, default=0.6)
parser.add_argument("--scene_floor_z", type=float, default=None)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--output", type=str, default="demos/navigate_failed_teleop.hdf5")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=400)
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--listen_host", type=str, default="0.0.0.0")
parser.add_argument("--listen_port", type=int, default=15002)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch
import omni.replicator.core as rep

from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg, _TUCKED_GRIPPER_RAD
from procthor_scene import (
    SceneTaskLayout,
    apply_scene_task_layout,
    resolve_scene_usd,
)

# ═══════════════════════════════════════════════════════════════════════
#  실패 항목 스케줄
# ═══════════════════════════════════════════════════════════════════════

_NAV_OPEN_ARM_POSE = [-0.05, -0.20, -0.14, -0.87, 0.20, 0.20]

# (x, y, yaw_deg, arm_mode, direction)
FAILED_SCHEDULE = [
    (2.457, 9.165, -1.9, "open", "forward"),
    (2.457, 9.165, -1.9, "open", "backward"),
    (2.457, 9.165, -1.9, "open", "left"),
    (2.457, 9.165, -1.9, "open", "right"),
    (2.683, 8.78, 135.4, "open", "forward"),
    (2.683, 8.78, 135.4, "open", "backward"),
    (2.683, 8.78, 135.4, "open", "left"),
    (2.683, 8.78, 135.4, "open", "right"),
    (6.971, 8.048, -134.7, "open", "forward"),
    (6.971, 8.048, -134.7, "open", "backward"),
    (6.971, 8.048, -134.7, "open", "left"),
    (6.971, 8.048, -134.7, "open", "right"),
    (6.969, 8.038, -32.5, "tucked", "turn_right"),
    (6.969, 8.038, -32.5, "open", "forward"),
    (6.969, 8.038, -32.5, "open", "backward"),
    (6.969, 8.038, -32.5, "open", "left"),
    (6.969, 8.038, -32.5, "open", "right"),
    (7.679, 9.597, -5.9, "open", "forward"),
    (7.679, 9.597, -5.9, "open", "backward"),
    (7.679, 9.597, -5.9, "open", "left"),
    (7.679, 9.597, -5.9, "open", "right"),
    (6.968, 12.301, 1.5, "open", "forward"),
    (6.968, 12.301, 1.5, "open", "backward"),
    (6.968, 12.301, 1.5, "open", "left"),
    (6.968, 12.301, 1.5, "open", "right"),
    (4.048, 15.515, -99.6, "open", "forward"),
    (4.048, 15.515, -99.6, "open", "backward"),
    (4.048, 15.515, -99.6, "open", "left"),
    (4.048, 15.515, -99.6, "open", "right"),
    (8.12, 5.244, 140.5, "tucked", "turn_right"),
    (8.12, 5.244, 140.5, "open", "forward"),
    (8.12, 5.244, 140.5, "open", "backward"),
    (8.12, 5.244, 140.5, "open", "left"),
    (8.12, 5.244, 140.5, "open", "right"),
    (8.867, 5.172, -6.8, "open", "forward"),
    (8.867, 5.172, -6.8, "open", "backward"),
    (8.867, 5.172, -6.8, "open", "left"),
    (8.867, 5.172, -6.8, "open", "right"),
    (7.405, 3.055, -7.6, "open", "forward"),
    (7.405, 3.055, -7.6, "open", "backward"),
    (7.405, 3.055, -7.6, "open", "left"),
    (7.405, 3.055, -7.6, "open", "right"),
    (4.882, 3.551, 60.0, "open", "forward"),
    (4.882, 3.551, 60.0, "open", "backward"),
    (4.882, 3.551, 60.0, "open", "left"),
    (4.882, 3.551, 60.0, "open", "right"),
    (6.437, 2.332, 125.2, "open", "forward"),
    (6.437, 2.332, 125.2, "open", "backward"),
    (6.437, 2.332, 125.2, "open", "left"),
    (6.437, 2.332, 125.2, "open", "right"),
]

DIR_MAP = {
    "forward": [0, 1, 0],
    "backward": [0, -1, 0],
    "left": [-1, 0, 0],
    "right": [1, 0, 0],
    "turn_left": [0, 0, 1],
    "turn_right": [0, 0, -1],
}

# ═══════════════════════════════════════════════════════════════════════
#  텔레옵 입력 (record_teleop_scene.py에서 인라인 복사 — import 충돌 방지)
# ═══════════════════════════════════════════════════════════════════════

import json
import socket
import threading

from lekiwi_robot_cfg import ARM_JOINT_NAMES


class TcpTeleopSubscriber:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._arm_positions = np.zeros(6, dtype=np.float64)
        self._base_cmd = np.zeros(3, dtype=np.float64)
        self._stamp = 0.0
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self):
        while not self._stop.is_set():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self._host, self._port))
                server.listen(1)
                server.settimeout(1.0)
                print(f"  [TCP] Listening on {self._host}:{self._port}")
                while not self._stop.is_set():
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue
                    print(f"  [TCP] Client connected: {addr[0]}:{addr[1]}")
                    conn.settimeout(1.0)
                    buf = ""
                    with conn:
                        while not self._stop.is_set():
                            try:
                                pkt = conn.recv(4096)
                            except socket.timeout:
                                continue
                            except OSError:
                                break
                            if not pkt:
                                break
                            buf += pkt.decode("utf-8", errors="ignore")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                self._handle_line(line.strip())
                    print("  [TCP] Client disconnected")
            except OSError as ex:
                print(f"  [TCP] Socket error: {ex}")
                time.sleep(1.0)
            finally:
                try:
                    server.close()
                except Exception:
                    pass

    def _handle_line(self, line):
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        with self._lock:
            payload = msg.get("action", msg) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}
            names = msg.get("name", []) if isinstance(msg, dict) else []
            positions = msg.get("position", []) if isinstance(msg, dict) else []
            if isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions):
                name_to_pos = dict(zip(names, positions))
                for i, jn in enumerate(ARM_JOINT_NAMES):
                    if jn in name_to_pos:
                        self._arm_positions[i] = float(name_to_pos[jn])
            base = msg.get("base", {}) if isinstance(msg, dict) else {}
            if isinstance(base, dict):
                self._base_cmd[0] = float(base.get("vx", self._base_cmd[0]))
                self._base_cmd[1] = float(base.get("vy", self._base_cmd[1]))
                self._base_cmd[2] = float(base.get("wz", self._base_cmd[2]))
            self._base_cmd[0] = float(payload.get("x.vel", payload.get("base.vx", self._base_cmd[0])))
            self._base_cmd[1] = float(payload.get("y.vel", payload.get("base.vy", self._base_cmd[1])))
            self._base_cmd[2] = float(payload.get("theta.vel", payload.get("base.wz", self._base_cmd[2])))
            self._stamp = time.time()

    def get_latest(self):
        with self._lock:
            arm = self._arm_positions.copy()
            ik = self._base_cmd.copy()
            body_cmd = np.array([-ik[1], ik[0], ik[2]])
            active = (time.time() - self._stamp) < 1.0
        return arm, body_cmd, active

    def shutdown(self):
        self._stop.set()


def normalize_arm_positions_to_rad(arm_pos, unit_hint="auto"):
    return arm_pos, unit_hint

# ═══════════════════════════════════════════════════════════════════════
#  키보드
# ═══════════════════════════════════════════════════════════════════════

import termios
import tty

_orig_term = None

def _setup_keyboard():
    global _orig_term
    import sys
    _orig_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

def _restore_keyboard():
    if _orig_term:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _orig_term)

def _check_arrow_key():
    import select
    if select.select([sys.stdin], [], [], 0.0)[0]:
        ch = os.read(sys.stdin.fileno(), 3)
        if ch == b'\x1b[C': return 1   # →
        if ch == b'\x1b[D': return -1  # ←
    return 0


def get_state_9d(env):
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].cpu().numpy()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].cpu().numpy()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return np.concatenate([arm, [grip, bv[0], bv[1], wz]]).astype(np.float32)


def capture(env, cams):
    env.sim.render()
    base_ann, wrist_ann = cams
    base_out = base_ann.get_data()
    wrist_out = wrist_ann.get_data()
    if isinstance(base_out, dict):
        base_rgb = base_out.get("rgb", base_out.get("rgba"))
    else:
        base_rgb = base_out
    if isinstance(wrist_out, dict):
        wrist_rgb = wrist_out.get("rgb", wrist_out.get("rgba"))
    else:
        wrist_rgb = wrist_out
    if base_rgb is None or wrist_rgb is None:
        return None, None
    base_rgb = np.array(base_rgb)[..., :3]
    wrist_rgb = np.array(wrist_rgb)[..., :3]
    return base_rgb, wrist_rgb


def main():
    # Scene
    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    if scene_path is None:
        raise FileNotFoundError(f"Scene not found: idx={args.scene_idx}")

    from procthor_scene import _load_support_floor_z, SCENE_PRESETS
    preset = SCENE_PRESETS.get(args.scene_idx)
    floor_z = 0.0
    if preset and args.scene_floor_z is None:
        floor_z = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path)

    # Env
    cfg = Skill1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 1.0
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.force_tucked_pose = True
    cfg.episode_length_s = 3600.0
    cfg.scene_reference_usd = str(scene_path)
    cfg.scene_scale = args.scene_scale
    cfg.use_builtin_ground = True
    cfg.builtin_ground_z = floor_z * args.scene_scale
    cfg.sim.device = "cpu"
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    env = Skill1Env(cfg=cfg)
    env._original_tucked_pose = env._tucked_pose.clone()
    env._original_tucked_gripper = _TUCKED_GRIPPER_RAD

    # Cameras
    base_cam_path = (
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
    )
    wrist_cam_path = (
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera"
    )
    base_rp = rep.create.render_product(base_cam_path, (args.camera_width, args.camera_height))
    wrist_rp = rep.create.render_product(wrist_cam_path, (args.camera_width, args.camera_height))
    base_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    base_ann.attach([base_rp])
    wrist_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_ann.attach([wrist_rp])
    cams = (base_ann, wrist_ann)

    # IK 직접 명령 모드 (텔레옵 불필요)

    # HDF5
    output_path = args.output
    if args.resume and os.path.isfile(output_path):
        hdf5_file = h5py.File(output_path, "a")
        saved_count = sum(1 for k in hdf5_file.keys() if k.startswith("episode_"))
        print(f"  [Resume] {saved_count}개 에피소드에서 이어서")
    else:
        hdf5_file = h5py.File(output_path, "w")
        saved_count = 0

    schedule_idx = saved_count
    total = len(FAILED_SCHEDULE)

    print(f"\n{'='*60}")
    print(f"  Navigate 실패 항목 IK 자동 수집 [{saved_count}/{total}]")
    print(f"  {args.max_steps} step/에피소드, headless 가능")
    print(f"{'='*60}\n")

    try:
        while schedule_idx < total and simulation_app.is_running():
            entry = FAILED_SCHEDULE[schedule_idx]
            pos_x, pos_y, yaw_deg, arm_mode, dir_key = entry

            # Reset
            obs, _ = env.reset()

            # Arm pose
            import lekiwi_skill1_env
            if arm_mode == "open":
                env._tucked_pose = torch.tensor(
                    _NAV_OPEN_ARM_POSE[:5], dtype=torch.float32, device=env.device)
                lekiwi_skill1_env._TUCKED_GRIPPER_RAD = _NAV_OPEN_ARM_POSE[5]
            else:
                env._tucked_pose = env._original_tucked_pose.clone()
                lekiwi_skill1_env._TUCKED_GRIPPER_RAD = env._original_tucked_gripper

            # Scene layout
            scaled_floor_z = floor_z * args.scene_scale
            layout = SceneTaskLayout(
                robot_xy=(pos_x, pos_y),
                robot_yaw_rad=math.radians(yaw_deg),
                source_xy=(pos_x + 1.0, pos_y),
                source_yaw_rad=0.0,
                dest_xy=(pos_x - 1.0, pos_y),
                dest_yaw_rad=0.0,
                floor_z=scaled_floor_z,
            )
            apply_scene_task_layout(env, layout)
            for _ in range(30):
                env.sim.step()
                env.sim.render()

            # Direction command
            cmd_vec = DIR_MAP[dir_key]
            env._direction_cmd[0] = torch.tensor(cmd_vec, dtype=torch.float32, device=env.device)
            instruction = f"navigate {dir_key}"

            print(f"  [{schedule_idx+1}/{total}] pos=({pos_x:.2f},{pos_y:.2f}) "
                  f"arm={arm_mode} dir={dir_key}")

            # Episode buffer
            ep_data = {
                "base_rgb": [], "wrist_rgb": [],
                "state": [], "action": [],
                "instruction": instruction,
            }

            # IK 직접 명령: direction → body frame velocity
            speed = 0.5  # m/s (linear), rad/s (angular)
            body_vel = {
                "forward":    (0.0, speed, 0.0),
                "backward":   (0.0, -speed, 0.0),
                "left":       (-speed, 0.0, 0.0),
                "right":      (speed, 0.0, 0.0),
                "turn_left":  (0.0, 0.0, 1.0),
                "turn_right": (0.0, 0.0, -1.0),
            }[dir_key]

            _stuck_count = 0
            step = 0
            key = 0
            while step < args.max_steps and simulation_app.is_running():
                action_np = np.zeros(9, dtype=np.float32)
                # base command (body frame: vx=right, vy=forward, wz=CCW)
                action_np[6] = body_vel[0]
                action_np[7] = body_vel[1]
                action_np[8] = body_vel[2]

                # Arm = current state (env forces tucked/open pose)
                state_9d = get_state_9d(env)
                action_np[:6] = state_9d[:6]

                action_t = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

                base_rgb, wrist_rgb = capture(env, cams)
                obs, _, _, _, _ = env.step(action_t)
                step += 1

                if base_rgb is not None and wrist_rgb is not None:
                    ep_data["base_rgb"].append(base_rgb)
                    ep_data["wrist_rgb"].append(np.zeros_like(base_rgb))
                    ep_data["state"].append(get_state_9d(env))
                    ep_data["action"].append(action_np.copy())

                # 충돌 감지 (직선 이동)
                if step > 30 and dir_key in ("forward", "backward", "left", "right"):
                    actual_speed = float(np.linalg.norm(state_9d[6:8]))
                    if actual_speed < 0.02:
                        _stuck_count += 1
                        if _stuck_count > 20:
                            print(f"    [Stuck] → stop at step {step}")
                            break
                    else:
                        _stuck_count = 0

                if step % 60 == 0:
                    rpos = env.robot.data.root_pos_w[0].tolist()
                    print(f"    [step={step}] pos=({rpos[0]:.2f},{rpos[1]:.2f})")

            # Auto-save
            n = len(ep_data["state"])
            if n >= 10:
                grp = hdf5_file.create_group(f"episode_{saved_count}")
                grp.create_dataset("actions", data=np.array(ep_data["action"], dtype=np.float32))
                grp.create_dataset("robot_state", data=np.array(ep_data["state"], dtype=np.float32))
                img_grp = grp.create_group("images")
                img_grp.create_dataset("base_rgb", data=np.stack(ep_data["base_rgb"]),
                                       compression="gzip", compression_opts=4)
                img_grp.create_dataset("wrist_rgb", data=np.stack(ep_data["wrist_rgb"]),
                                       compression="gzip", compression_opts=4)
                grp.attrs["instruction"] = instruction
                grp.attrs["num_steps"] = n
                hdf5_file.flush()
                saved_count += 1
                print(f"    [Saved] episode_{saved_count-1}: {n} steps")

            schedule_idx += 1

    except KeyboardInterrupt:
        print("\n  중단 (Ctrl+C)")

    hdf5_file.close()
    print(f"\n  완료: {saved_count} 에피소드 → {output_path}")

    env.close()
    simulation_app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
