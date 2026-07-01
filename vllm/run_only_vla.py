#!/usr/bin/env python3
"""
VLA-only 전체 태스크 실행기 (end-to-end Pi0.5, VLM 없이)

이 스크립트는 `lekiwi_full_teleop_combined` 데이터셋으로 학습된 end-to-end
Pi0.5 체크포인트 (예: H100 64K)를 평가하기 위한 것이다. 해당 모델은
단일 instruction ("find the medicine bottle and place it next to the red cup")
만을 사용해 학습되었으므로 VLM orchestrator나 skill state machine이 필요 없다.

아키텍처:
    Isaac Sim (local GPU)
      └─ base_cam + wrist_cam + depth 렌더

    VLA 서버 (port 8002)
      └─ base_rgb + wrist_rgb + state + fixed_instruction → 9D action chunk

    매 step: env.step(action), safety depth check만

Usage:
    # 서버 기동 (only_vla end-to-end 064K 체크포인트 — 로컬 백업 경로):
    bash launch_servers.sh vla --checkpoint backup/h100_endtoend/064000/pretrained_model

    # 실행:
    python run_only_vla.py --headless
    # 또는 다른 instruction:
    python run_only_vla.py --instruction "find the bottle and place it on the table" --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Add own dir first (공용 유틸), then parent (lekiwi_skill*_env)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Only-VLA (end-to-end Pi0.5, no VLM)")

# VLA server
parser.add_argument("--vla_server", type=str, default="http://localhost:8002")
parser.add_argument("--jpeg_quality", type=int, default=80)

# Task (instruction is FIXED — no VLM classification)
parser.add_argument(
    "--instruction",
    type=str,
    default="find the medicine bottle and place it next to the red cup",
    help="Pi0.5에 전달할 고정 instruction. 학습 데이터와 동일해야 함.",
)

# Execution
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--max_total_steps", type=int, default=6000,
                    help="trial당 최대 step (6000 = 약 10분 at 10Hz)")
parser.add_argument("--n_use", type=int, default=50,
                    help="VLA chunk(50) 중 실제 사용할 action 개수. "
                         "50이면 chunk 전체 실행 후 재쿼리 (Receding Horizon). "
                         "작게 하면 반응성↑, 쿼리 오버헤드↑")
parser.add_argument("--action_log", type=str, default="",
                    help="매 step VLA action 기록 파일 (빈 문자열=비활성)")
parser.add_argument("--frame_save_dir", type=str, default="",
                    help="지정 시 매 N step 카메라 프레임 저장 (PNG)")
parser.add_argument("--frame_save_interval", type=int, default=5)

# Camera (VLA만, VLM 없음)
parser.add_argument("--vla_width", type=int, default=640)
parser.add_argument("--vla_height", type=int, default=400)
parser.add_argument("--render_wide", action="store_true", default=False,
                    help="base_rgb를 1280x800으로 렌더 후 downscale (VLM 호환용, 기본 640x400 직접 렌더)")

# Safety (depth 기반 단순 safety, VLM 없음)
parser.add_argument("--safety_dist", type=float, default=0.3,
                    help="depth < safety_dist면 y.vel(전진) 차단. 0이면 비활성")

# Env
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

# Scene (ProcTHOR)
parser.add_argument("--scene_idx", type=int, default=0)
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_scale", type=float, default=1.0)
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")

# Difficulty
parser.add_argument("--difficulty", type=str, default="easy",
                    choices=["easy", "middle", "hard"])

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import base64
import io

import numpy as np
import requests
import torch
from PIL import Image

import omni.replicator.core as rep


# ═══════════════════════════════════════════════════════════════════════
#  VLA Client (run_full_task.py와 동일 로직, n_use 파라미터 포함)
# ═══════════════════════════════════════════════════════════════════════

class VLAClient:
    """Pi0.5 VLA 서버 클라이언트. Action chunk 버퍼링 지원."""

    def __init__(self, server_url: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        self._action_buffer: list[list[float]] = []
        self._buffer_idx = 0
        self._last_latency = 0.0
        self._total_latency = 0.0
        self._call_count = 0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_action(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                     state_9d: list[float], instruction: str) -> list[list[float]]:
        t0 = time.perf_counter()
        payload = {
            "base_image_b64": self.encode_image(base_rgb),
            "wrist_image_b64": self.encode_image(wrist_rgb),
            "state": state_9d,
            "instruction": instruction,
        }
        try:
            resp = self._session.post(
                f"{self.server_url}/act", json=payload, timeout=10.0,
            )
            resp.raise_for_status()
            result = resp.json()
            self._last_latency = time.perf_counter() - t0
            self._total_latency += self._last_latency
            self._call_count += 1
            return result["actions"]
        except Exception as e:
            print(f"  [VLA] error: {e}")
            self._last_latency = time.perf_counter() - t0
            return []

    def get_action_9d(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                      state_9d: list[float], instruction: str,
                      n_use: int | None = None) -> np.ndarray:
        effective_max = len(self._action_buffer)
        if n_use is not None:
            effective_max = min(effective_max, n_use)

        if self._buffer_idx >= effective_max:
            self._action_buffer = self.query_action(
                base_rgb, wrist_rgb, state_9d, instruction
            )
            self._buffer_idx = 0
            if not self._action_buffer:
                return np.zeros(9, dtype=np.float32)

        raw = np.array(self._action_buffer[self._buffer_idx], dtype=np.float32)
        self._buffer_idx += 1
        if len(raw) >= 9:
            return raw[:9]
        return np.pad(raw, (0, 9 - len(raw)))

    def reset_buffer(self):
        self._action_buffer = []
        self._buffer_idx = 0

    def health_check(self) -> dict | None:
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=3.0)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    @property
    def latency(self) -> float:
        return self._last_latency

    @property
    def avg_latency(self) -> float:
        return self._total_latency / max(self._call_count, 1)

    @property
    def call_count(self) -> int:
        return self._call_count


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def get_depth_min(depth_image: np.ndarray) -> float | None:
    """전방 중앙 1/3 영역 min depth. (depth 이미지가 로봇 arm을 포함하는 경우
    false positive 가능하지만 단순 safety용)."""
    if depth_image is None:
        return None
    H, W = depth_image.shape[:2]
    center = depth_image[H // 3 : 2 * H // 3, W // 3 : 2 * W // 3]
    valid = (center > 0.10) & (center < 10.0)
    if valid.sum() < 10:
        return None
    return float(center[valid].min())


def get_state_9d(env) -> list[float]:
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].tolist()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].tolist()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return arm + [grip] + bv[:2] + [wz]


def downscale_for_vla(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if rgb is None:
        return None
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


# ═══════════════════════════════════════════════════════════════════════
#  Env setup (run_full_task.py를 간소화 — VLM render product 제거)
# ═══════════════════════════════════════════════════════════════════════

def setup_env(args):
    from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
    from procthor_scene import resolve_scene_usd

    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = 600.0

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    # 약병(source)=0.7, 컵(dest)=0.56 — 3개 파이프라인(VIVA/only-VLA/VLM+RL) task 통일
    cfg.object_scale = 0.7
    cfg.dest_object_scale = 0.56
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    if scene_path is not None:
        cfg.scene_reference_usd = str(scene_path)
        cfg.scene_scale = args.scene_scale
        cfg.use_builtin_ground = True
        from procthor_scene import _load_support_floor_z, SCENE_PRESETS
        preset = SCENE_PRESETS.get(args.scene_idx)
        floor_z = _load_support_floor_z(
            str(scene_path.resolve()), preset.support_floor_prim_path
        ) if preset else 0.0
        cfg.builtin_ground_z = floor_z * args.scene_scale - 0.1
        cfg.sim.device = "cpu"
        print(f"  [Scene] {scene_path}, floor_z={floor_z:.4f}, scale={args.scene_scale}, device=cpu")

    env = Skill2Env(cfg=cfg)

    # Ground 검정색 (시각만)
    import omni.usd as _ousd
    from pxr import UsdShade, Sdf, Gf
    _stage = _ousd.get_context().get_stage()
    _cube_vis = _stage.GetPrimAtPath("/World/ground/geometry/mesh")
    if not _cube_vis.IsValid():
        _cube_vis = _stage.GetPrimAtPath("/World/ground")
    if _cube_vis.IsValid():
        _mtl_path = "/World/Looks/BlackMatte"
        UsdShade.Material.Define(_stage, _mtl_path)
        _shader = UsdShade.Shader.Define(_stage, _mtl_path + "/Shader")
        _shader.CreateIdAttr("UsdPreviewSurface")
        _shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
        _shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        _mtl = UsdShade.Material.Get(_stage, _mtl_path)
        _mtl.CreateSurfaceOutput().ConnectToSource(_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(_cube_vis).Bind(_mtl)

    # 카메라 — VLA 직접 해상도 (640x400). render_wide면 1280x800 후 downscale.
    base_w = 1280 if args.render_wide else args.vla_width
    base_h = 800 if args.render_wide else args.vla_height

    base_rgb_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (base_w, base_h),
    )
    wrist_rgb_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera",
        (args.vla_width, args.vla_height),
    )
    DEPTH_W, DEPTH_H = 320, 200
    depth_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (DEPTH_W, DEPTH_H),
    )

    base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    base_rgb_annot.attach([base_rgb_rp])
    wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_rgb_annot.attach([wrist_rgb_rp])
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([depth_rp])

    print(f"  [Render] base={base_w}x{base_h} wrist={args.vla_width}x{args.vla_height} depth={DEPTH_W}x{DEPTH_H}")

    cams = {
        "base_rgb": base_rgb_annot,
        "wrist_rgb": wrist_rgb_annot,
        "depth": depth_annot,
    }
    return env, cams, scene_path


def capture(env, cams):
    env.sim.render()
    b = cams["base_rgb"].get_data()
    d = cams["depth"].get_data()
    w = cams["wrist_rgb"].get_data()
    base_rgb = np.array(b)[..., :3] if b is not None else None
    depth = np.array(d) if d is not None else None
    wrist_rgb = np.array(w)[..., :3] if w is not None else None
    return base_rgb, depth, wrist_rgb


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")

    print(f"\n{'='*60}")
    print(f"  Only-VLA (end-to-end Pi0.5, no VLM)")
    print(f"  VLA server: {args.vla_server}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  n_use (chunk consumption): {args.n_use} / 50")
    print(f"  safety_dist: {args.safety_dist} m")
    print(f"{'='*60}")

    # VLA 서버 health check
    vla = VLAClient(args.vla_server, args.jpeg_quality)
    h = vla.health_check()
    if h is None:
        print("  [ERROR] VLA 서버 응답 없음. 서버 기동 확인 필요:")
        print(f"     bash launch_servers.sh vla --checkpoint <ckpt_path>")
        sys.exit(1)
    print(f"  [OK] VLA server: model={h.get('model')}, device={h.get('device')}, "
          f"gpu_mem={h.get('gpu_memory_mb', 0):.0f}MB")

    # Env 세팅
    env, cams, scene_path = setup_env(args)
    env.reset()
    # Difficulty prop 전달 (Skill2Env 평가 모드)
    if hasattr(env, "set_difficulty"):
        env.set_difficulty(args.difficulty)

    # frame save
    frame_save_dir = None
    if args.frame_save_dir:
        frame_save_dir = os.path.expanduser(args.frame_save_dir)
        os.makedirs(frame_save_dir, exist_ok=True)
        print(f"  [FrameSave] {frame_save_dir} (interval={args.frame_save_interval})")

    # Action log
    action_log_f = None
    if args.action_log:
        action_log_f = open(os.path.expanduser(args.action_log), "w")
        action_log_f.write("# step action[9] depth_min safety_triggered\n")

    # ───────────── Trial loop ─────────────
    trial_stats = []
    for trial in range(args.num_trials):
        print(f"\n{'─'*60}")
        print(f"  Trial {trial+1} / {args.num_trials} ({args.difficulty})")
        print(f"{'─'*60}")

        env.reset()
        vla.reset_buffer()

        # ── Difficulty-aware spawn (scene_path 있을 때만; run_full_task.py와 동일 로직) ──
        if scene_path is not None:
            from procthor_scene import (
                apply_scene_task_layout, SceneTaskLayout,
                _load_floor_regions, _load_support_floor_z,
                _load_scene_obstacles, _find_robot_region,
                _load_floor_triangles, sample_on_floor_mesh,
                SCENE_PRESETS,
            )
            import math as _m
            import random as _rng_mod

            # run_full_task.py / eval_vlm_rl.py와 동일 (3개 파이프라인 공정 비교 — 같은 씬)
            _DIFFICULTY_MAP = {
                "easy":   ("room_9",  1.086, 33.694, 214.4, "room_9"),
                "middle": ("room_6",  7.657,  0.638, 354.9, "room_6"),
                "hard":   ("room_4",  3.25,  14.05,  -76.2, "room_3"),
            }

            _scene_str = str(scene_path.resolve())
            preset = SCENE_PRESETS.get(args.scene_idx)
            sfz = _load_support_floor_z(_scene_str, preset.support_floor_prim_path) if preset else 0.0
            regions = _load_floor_regions(_scene_str, support_floor_z=sfz)
            obstacles = _load_scene_obstacles(_scene_str)
            floor_tris = _load_floor_triangles(_scene_str)
            ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0

            _diff_entry = _DIFFICULTY_MAP[args.difficulty]
            _target_room, _rx, _ry, _ryaw_deg, _obj_room_id = _diff_entry
            _obj_tris = floor_tris.get(_obj_room_id, [])
            _fz = sfz * ss

            layout = None
            for _spawn_try in range(200):
                _rng = _rng_mod.Random()
                try:
                    _sxy = sample_on_floor_mesh(_obj_tris, obstacles, 0.3, _rng)
                    _dxy = sample_on_floor_mesh(_obj_tris, obstacles, 0.3, _rng)
                except RuntimeError:
                    continue
                if _m.dist(_sxy, _dxy) < 1.5:
                    continue
                if _obj_room_id == _target_room:
                    if _m.dist((_rx, _ry), _sxy) < 1.5 or _m.dist((_rx, _ry), _dxy) < 1.5:
                        continue
                layout = SceneTaskLayout(
                    robot_xy=(_rx * ss, _ry * ss),
                    robot_yaw_rad=_m.radians(_ryaw_deg),
                    source_xy=(_sxy[0] * ss, _sxy[1] * ss),
                    source_yaw_rad=_rng_mod.uniform(-_m.pi, _m.pi),
                    dest_xy=(_dxy[0] * ss, _dxy[1] * ss),
                    dest_yaw_rad=_rng_mod.uniform(-_m.pi, _m.pi),
                    floor_z=_fz,
                    source_rest_z=0.033,
                )
                break
            if layout is None:
                print(f"  [ERROR] {args.difficulty} spawn 실패 → trial skip")
                continue
            apply_scene_task_layout(env, layout)

            # arm → navigate tucked pose
            _env_id = torch.tensor([0], device=env.device)
            _zero_v = torch.zeros(1, env.robot.num_joints, device=env.device)
            _NAV_TUCKED = [-0.02966, -0.213839, 0.09066, -0.4, 0.058418, -0.201554]
            _jp = env.robot.data.joint_pos[0:1].clone()
            _jp[0, env.arm_idx[:5]] = torch.tensor(_NAV_TUCKED[:5], dtype=torch.float32, device=env.device)
            _jp[0, env.arm_idx[5]] = _NAV_TUCKED[5]
            _jv = torch.zeros_like(_jp)
            env.robot.write_joint_state_to_sim(_jp, _jv, env_ids=_env_id)
            env.robot.set_joint_position_target(_jp, env_ids=_env_id)
            env.robot.set_joint_velocity_target(_zero_v)
            for _ in range(60):
                env.sim.step()
                env.sim.render()
            # re-teleport (중력/충돌로 약간 밀린 것 보정)
            apply_scene_task_layout(env, layout)
            env.robot.write_joint_state_to_sim(_jp, _jv, env_ids=_env_id)
            env.robot.set_joint_position_target(_jp, env_ids=_env_id)
            env.robot.set_joint_velocity_target(_zero_v)
            env.sim.step()
            env.robot.update(env.sim.cfg.dt)

            _r_room = _find_robot_region((_rx, _ry), regions)

            def _room_id(fp):
                name = fp.path.split("/")[-1]
                idx = name.find("_visual_")
                return name[:idx] if idx >= 0 else name

            print(f"  [Spawn] {args.difficulty} | "
                  f"robot={_room_id(_r_room) if _r_room else '?'}({_rx:.1f},{_ry:.1f}) | "
                  f"src=({_sxy[0]:.1f},{_sxy[1]:.1f}) dst=({_dxy[0]:.1f},{_dxy[1]:.1f})")

        total_steps = 0
        safety_stops = 0
        t_start = time.time()
        success = False

        try:
            while total_steps < args.max_total_steps:
                # (a) capture
                base_rgb, depth, wrist_rgb = capture(env, cams)
                depth_min = get_depth_min(depth)

                # (b) base_rgb를 VLA 해상도로 (render_wide=true 경우에만 downscale)
                base_rgb_vla = downscale_for_vla(base_rgb, args.vla_width, args.vla_height)

                # (c) state
                state = get_state_9d(env)

                # (d) VLA action
                _t_vla0 = time.perf_counter()
                action = vla.get_action_9d(
                    base_rgb_vla, wrist_rgb, state, args.instruction, n_use=args.n_use
                )
                _t_vla_ms = (time.perf_counter() - _t_vla0) * 1000

                # (e) safety — depth 기반 단순 차단 (VLM 없음)
                stopped = False
                if args.safety_dist > 0 and depth_min is not None \
                        and depth_min < args.safety_dist and float(action[7]) > 0:
                    action = action.copy()
                    action[7] = 0.0  # y.vel (전진) 차단
                    stopped = True
                    safety_stops += 1

                # (f) env.step
                action_t = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
                obs, rew, term, trunc, info = env.step(action_t)
                total_steps += 1

                # (g) 로그 (10 step마다)
                if total_steps % 10 == 0:
                    elapsed = time.time() - t_start
                    hz = total_steps / elapsed if elapsed > 0 else 0
                    a = action
                    s = state
                    vla_base = f"base: {a[6]:+.2f}, {a[7]:+.2f}, {a[8]:+.2f}"
                    vla_arm = f"arm: {a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}, {a[3]:+.2f}, {a[4]:+.2f}"
                    vla_grip = f"grip: {a[5]:+.2f}"
                    real_base = f"base: {s[6]:+.2f}, {s[7]:+.2f}, {s[8]:+.2f}"
                    real_arm = f"arm: {s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f}, {s[3]:+.2f}, {s[4]:+.2f}"
                    real_grip = f"grip: {s[5]:+.2f}"
                    if depth_min is not None:
                        safe_flag = "X" if depth_min < args.safety_dist else "O"
                        depth_str = f"depth={depth_min:.2f}m (safe {safe_flag})"
                    else:
                        depth_str = "depth=None (safe O)"
                    print(f"    step={total_steps:4d} "
                          f"vla={vla.avg_latency*1000:.0f}ms({vla.call_count}) {depth_str}"
                          f"{' [SAFETY]' if stopped else ''}")
                    print(f"    VLA  = [{vla_base} | {vla_arm} | {vla_grip}] @{hz:.1f}Hz")
                    print(f"    REAL = [{real_base} | {real_arm} | {real_grip}]")

                # frame save
                if frame_save_dir and total_steps % args.frame_save_interval == 0:
                    Image.fromarray(base_rgb.astype(np.uint8)).save(
                        os.path.join(frame_save_dir, f"trial{trial:02d}_step{total_steps:05d}_base.png"))
                    if wrist_rgb is not None:
                        Image.fromarray(wrist_rgb.astype(np.uint8)).save(
                            os.path.join(frame_save_dir, f"trial{trial:02d}_step{total_steps:05d}_wrist.png"))

                # action log
                if action_log_f:
                    action_log_f.write(
                        f"{total_steps} " + " ".join(f"{v:+.4f}" for v in action.tolist())
                        + f" {depth_min if depth_min is not None else -1.0:.3f} {int(stopped)}\n"
                    )

                # env 종료
                if term.any() or trunc.any():
                    success = info.get("task_success", torch.zeros(1)).any().item()
                    print(f"\n  Trial {trial+1} → {'SUCCESS' if success else 'TIMEOUT'} "
                          f"| steps={total_steps} safety_stops={safety_stops}")
                    break
            else:
                print(f"\n  Trial {trial+1} → MAX_STEPS ({args.max_total_steps}) "
                      f"| safety_stops={safety_stops}")

        except KeyboardInterrupt:
            print(f"\n  [INTERRUPTED] at step {total_steps}")
            break

        elapsed = time.time() - t_start
        trial_stats.append({
            "trial": trial + 1,
            "steps": total_steps,
            "elapsed": elapsed,
            "safety_stops": safety_stops,
            "success": success,
            "avg_hz": total_steps / elapsed if elapsed > 0 else 0.0,
        })

    # ───────────── Summary ─────────────
    print(f"\n{'='*60}")
    print(f"  Summary ({args.num_trials} trials)")
    print(f"{'='*60}")
    n_success = sum(1 for t in trial_stats if t["success"])
    print(f"  Success: {n_success} / {len(trial_stats)} ({100*n_success/max(len(trial_stats),1):.0f}%)")
    for ts in trial_stats:
        mark = "✓" if ts["success"] else "✗"
        print(f"  {mark} Trial {ts['trial']}: "
              f"{ts['steps']:4d} steps, {ts['elapsed']:.0f}s, "
              f"{ts['avg_hz']:.1f}Hz, safety={ts['safety_stops']}")
    print(f"  VLA avg latency: {vla.avg_latency*1000:.0f}ms, {vla.call_count} calls total")

    if action_log_f:
        action_log_f.close()
        print(f"  [ActionLog] Saved: {args.action_log}")

    simulation_app.close()


if __name__ == "__main__":
    main()