#!/usr/bin/env python3
"""
VIVA 파이프라인 검증 스크립트.

S1: 키보드로 base 이동 + VLM 50스텝 호출 확인
S2: RL expert (BC+ResiP) 자동 실행 + VLM obstacle 호출 확인
S3/S4: 키보드 / 자동 (S2→S3 전환 시)

VLA는 base 모델이라 action=0이지만, 호출 자체의 입출력 포맷을 검증.
모든 VLM/VLA 호출을 로그 파일에 기록.

Usage:
    python vllm/eval_viva_pipeline.py \
        --object_usd /path/to/5_HTP/model_clean.usd \
        --dest_object_usd /path/to/ACE_Coffee_Mug/model_clean.usd \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --resip_checkpoint backup/appoachandlift/resip64%.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description="VIVA Pipeline Eval")
parser.add_argument("--vlm_server", type=str, default="http://localhost:8000")
parser.add_argument("--vla_server", type=str, default="http://localhost:8002")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument("--user_command", type=str,
    default="find the medicine bottle and place it next to the red cup")
parser.add_argument("--object_usd", type=str, required=True)
parser.add_argument("--dest_object_usd", type=str, required=True)
parser.add_argument("--dp_checkpoint", type=str, default="", help="S2 approach&lift BC")
parser.add_argument("--resip_checkpoint", type=str, default="", help="S2 approach&lift ResiP")
parser.add_argument("--nav_dp_checkpoint", type=str, default="", help="S1 navigate BC")
parser.add_argument("--nav_resip_checkpoint", type=str, default="", help="S1 navigate ResiP")
parser.add_argument("--camera_width", type=int, default=1280)
parser.add_argument("--camera_height", type=int, default=800)
parser.add_argument("--vlm_interval", type=int, default=30)
parser.add_argument("--max_total_steps", type=int, default=6000)
parser.add_argument("--safety_dist", type=float, default=0.0)
parser.add_argument("--log_file", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--scene_idx", type=int, default=0, help="ProcTHOR scene index (0=no scene)")
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_scale", type=float, default=1.0)
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import base64
import io
import json
import math
import logging
import threading

import numpy as np
import requests
import torch
from PIL import Image

import omni.replicator.core as rep

# ── VLLM 모듈 import (sys.path 설정) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from vlm_orchestrator import (
    classify_user_request, VIVAOrchestrator, SkillState,
    LIFTED_POSE_RANGE, NAVIGATE_COMMANDS, CARRY_COMMANDS,
)


# ═══════════════════════════════════════════════════════════════════════
#  run_full_task.py에서 가져온 유틸 (import 시 모듈 레벨 코드 실행 방지)
# ═══════════════════════════════════════════════════════════════════════

class VLAClient:
    """Pi0-FAST VLA 서버 클라이언트. Action chunk 버퍼링 지원."""
    def __init__(self, server_url: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        self._action_buffer: list[list[float]] = []
        self._buffer_idx = 0
        self._last_latency = 0.0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_action(self, base_rgb, wrist_rgb, state_9d, instruction):
        t0 = time.perf_counter()
        payload = {
            "base_image_b64": self.encode_image(base_rgb),
            "wrist_image_b64": self.encode_image(wrist_rgb),
            "state": state_9d,
            "instruction": instruction,
        }
        try:
            resp = self._session.post(f"{self.server_url}/act", json=payload, timeout=10.0)
            resp.raise_for_status()
            self._last_latency = time.perf_counter() - t0
            return resp.json()["actions"]
        except Exception as e:
            print(f"  [VLA] error: {e}")
            self._last_latency = time.perf_counter() - t0
            return []

    def get_action_9d(self, base_rgb, wrist_rgb, state_9d, instruction):
        if self._buffer_idx >= len(self._action_buffer):
            self._action_buffer = self.query_action(base_rgb, wrist_rgb, state_9d, instruction)
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

    def health_check(self):
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=3.0)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    @property
    def latency(self):
        return self._last_latency


def get_depth_min(depth_image):
    if depth_image is None:
        return None
    H, W = depth_image.shape[:2]
    center = depth_image[H // 3 : 2 * H // 3, W // 3 : 2 * W // 3]
    valid = (center > 0.10) & (center < 10.0)
    if valid.sum() < 10:
        return None
    return float(center[valid].min())


def get_contact_detected(env):
    try:
        jaw_forces = env.contact_sensor.data.net_forces_w
        jaw_mag = jaw_forces[0].norm(dim=-1).max().item()
        wrist_mag = 0.0
        if hasattr(env, 'wrist_contact_sensor') and env.wrist_contact_sensor is not None:
            wrist_forces = env.wrist_contact_sensor.data.net_forces_w
            wrist_mag = wrist_forces[0].norm(dim=-1).max().item()
        return (jaw_mag > 1.0) or (wrist_mag > 1.0)
    except Exception:
        return False


def check_lifted_pose(arm_joints, grip_pos, contact):
    if not contact:
        return False
    joints_with_grip = arm_joints + [grip_pos]
    for val, (low, high) in zip(joints_with_grip, LIFTED_POSE_RANGE.values()):
        if not (low <= val <= high):
            return False
    return True


def build_robot_status(env, contact, depth_min):
    jp = env.robot.data.joint_pos[0]
    arm_joints = jp[env.arm_idx[:5]].tolist()
    grip_pos = jp[env.gripper_idx].item()
    if contact:
        gripper_str = f"closed, contact detected, position={grip_pos:.3f}"
    elif grip_pos < 0.3:
        gripper_str = f"closed, no contact, position={grip_pos:.3f}"
    else:
        gripper_str = f"open, position={grip_pos:.3f}"
    lifted = check_lifted_pose(arm_joints, grip_pos, contact)
    arm_str = "LIFTED" if lifted else "NOT_LIFTED"
    if depth_min is not None and depth_min < 0.3:
        depth_str = f"CLOSE_OBJECT_DETECTED (min depth: {depth_min:.2f}m)"
    else:
        depth_str = "NONE"
    return (f"Robot status:\n- Gripper: {gripper_str}\n"
            f"- Arm pose: {arm_str}, joints={[round(j, 3) for j in arm_joints]}\n"
            f"- Depth warning: {depth_str}")


def get_state_9d(env):
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].tolist()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].tolist()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return arm + [grip] + bv[:2] + [wz]


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
#  Logging
# ═══════════════════════════════════════════════════════════════════════

def setup_logger(log_file: str):
    logger = logging.getLogger("viva_eval")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ═══════════════════════════════════════════════════════════════════════
#  Keyboard (pynput — hold 감지 지원)
# ═══════════════════════════════════════════════════════════════════════

from pynput import keyboard as pynput_kb

_pressed: dict[str, bool] = {}
_commands: list[str] = []  # one-shot 명령 (q, t, 1-4)

def _key_to_str(k) -> str | None:
    try:
        return k.char
    except AttributeError:
        mapping = {
            pynput_kb.Key.up: 'w', pynput_kb.Key.down: 's',
            pynput_kb.Key.left: 'a', pynput_kb.Key.right: 'd',
            pynput_kb.Key.space: 'space',
        }
        return mapping.get(k)

def _on_press(k):
    s = _key_to_str(k)
    if s:
        _pressed[s] = True
        if s in ('q', 't', 'r', '1', '2', '3', '4'):
            _commands.append(s)

def _on_release(k):
    s = _key_to_str(k)
    if s:
        _pressed.pop(s, None)

_kb_listener = None

def kb_setup():
    global _kb_listener
    _kb_listener = pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    _kb_listener.start()

def kb_restore():
    if _kb_listener:
        _kb_listener.stop()


# ═══════════════════════════════════════════════════════════════════════
#  S2 RL Expert (BC + ResiP)
# ═══════════════════════════════════════════════════════════════════════

def load_s2_expert(dp_path: str, resip_path: str, device):
    """BC + ResiP 로드. resip_path 없으면 BC only."""
    from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy

    if not dp_path or not os.path.isfile(dp_path):
        return None, None, None

    ckpt = torch.load(dp_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"],
        action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=4,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)

    sd = ckpt["model_state_dict"]
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    norm_sd = {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")}
    agent.model.load_state_dict(model_sd)
    agent.normalizer.load_state_dict(norm_sd, device=device)
    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()

    residual = None
    if resip_path and os.path.isfile(resip_path):
        rckpt = torch.load(resip_path, map_location=device, weights_only=False)
        saved_args = rckpt.get("args", {})
        residual = ResidualPolicy(
            obs_dim=cfg["obs_dim"], action_dim=cfg["act_dim"],
            actor_hidden_size=saved_args.get("actor_hidden_size", 256),
            actor_num_layers=saved_args.get("actor_num_layers", 2),
            critic_hidden_size=saved_args.get("critic_hidden_size", 256),
            critic_num_layers=saved_args.get("critic_num_layers", 2),
            action_scale=saved_args.get("action_scale", 0.1),
            init_logstd=saved_args.get("init_logstd", -1.0),
            learn_std=False,
        ).to(device)
        residual.load_state_dict(rckpt["residual_policy_state_dict"])
        residual.eval()

    return agent, residual, cfg


def resip_action(dp_agent, residual, obs_t, device, per_dim_scale):
    """BC + residual → 9D action. per_dim_scale은 스킬별로 다름."""
    base_nact = dp_agent.base_action_normalized(obs_t)
    if residual is not None:
        nobs = dp_agent.normalizer(obs_t, "obs", forward=True)
        nobs = torch.clamp(nobs, -3, 3)
        nobs = torch.nan_to_num(nobs, nan=0.0)
        res_input = torch.cat([nobs, base_nact], dim=-1)
        res_nact = residual.actor_mean(res_input)
        res_nact = torch.clamp(res_nact, -1.0, 1.0)
        nact = base_nact + res_nact * per_dim_scale
    else:
        nact = base_nact
    action = dp_agent.normalizer(nact, "action", forward=False)
    return action.squeeze(0).detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
#  Environment Setup
# ═══════════════════════════════════════════════════════════════════════

def setup_env(args):
    from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
    from procthor_scene import (
        resolve_scene_usd, sample_scene_task_layout, apply_scene_task_layout,
        SceneSpawnCfg,
    )

    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = 600.0
    cfg.object_usd = os.path.expanduser(args.object_usd)
    cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

    # Scene (ProcTHOR)
    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    use_scene = scene_path is not None
    if use_scene:
        cfg.scene_reference_usd = str(scene_path)
        cfg.scene_scale = args.scene_scale
        cfg.use_builtin_ground = True
        from procthor_scene import _load_support_floor_z, SCENE_PRESETS
        preset = SCENE_PRESETS.get(args.scene_idx)
        if preset:
            floor_z = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path)
        else:
            floor_z = 0.0
        cfg.builtin_ground_z = floor_z * args.scene_scale
        cfg.sim.device = "cpu"
        print(f"  [Scene] {scene_path}, floor_z={floor_z:.4f}, scale={args.scene_scale}, device=cpu")
    else:
        cfg.sim.device = "cuda:0"

    # VIVA: env terminate 비활성화 (스킬 전환은 코드로 판별)
    cfg.grasp_success_height = 100.0  # lift success terminate 방지
    cfg.object_dist_min = 1.2
    cfg.object_dist_max = 1.4
    cfg.episode_length_s = 3600.0     # 1시간
    cfg.max_dist_from_origin = 50.0   # scene 내 이동 허용
    cfg.dr_action_delay_steps = 0     # action delay 제거 (navigate DP 호환)

    env = Skill2Env(cfg=cfg)

    # topple/success terminate 비활성화 (monkey-patch)
    _original_get_dones = env._get_dones
    def _no_terminate_get_dones():
        terminated, truncated = _original_get_dones()
        terminated[:] = False
        truncated[:] = False
        return terminated, truncated
    env._get_dones = _no_terminate_get_dones

    # _reset_idx 패치: 매 리셋마다 scene 안에 스폰 (같은 방 검증)
    if use_scene:
        from procthor_scene import (
            _load_floor_regions, _load_support_floor_z, _find_robot_region,
            SCENE_PRESETS,
        )
        _preset = SCENE_PRESETS.get(args.scene_idx)
        _sfz = _load_support_floor_z(str(scene_path.resolve()), _preset.support_floor_prim_path)
        _regions = _load_floor_regions(str(scene_path.resolve()), support_floor_z=_sfz)

        _original_reset_idx = env._reset_idx
        def _scene_reset_idx(env_ids):
            _original_reset_idx(env_ids)
            if 0 in env_ids:
                ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
                _src_ov = SceneSpawnCfg(
                    min_robot_dist=float(getattr(env.cfg, "object_dist_min", 0.8)) / ss,
                    max_robot_dist=float(getattr(env.cfg, "object_dist_max", 1.2)) / ss,
                    clearance_radius=0.14,
                )
                for _r in range(50):
                    try:
                        _layout = sample_scene_task_layout(
                            args.scene_idx, scene_usd=scene_path,
                            scene_scale=args.scene_scale,
                            source_spawn_override=_src_ov,
                            robot_faces_source=True,
                            randomize_robot_xy=True,
                        )
                        # 같은 방 검증: 로봇과 source가 같은 room에 있는지
                        _robot_unscaled = (_layout.robot_xy[0] / ss, _layout.robot_xy[1] / ss)
                        _source_unscaled = (_layout.source_xy[0] / ss, _layout.source_xy[1] / ss)
                        _robot_reg = _find_robot_region(_robot_unscaled, _regions)
                        _source_reg = _find_robot_region(_source_unscaled, _regions)
                        if _robot_reg and _source_reg and _robot_reg.path == _source_reg.path:
                            break  # 같은 방 OK
                        # 다른 방이면 재시도
                        if _r < 49:
                            continue
                    except RuntimeError:
                        if _r < 49: continue
                        raise
                apply_scene_task_layout(env, _layout)
                print(f"  [Scene] Robot room: {_robot_reg.path if _robot_reg else '?'}, "
                      f"Source room: {_source_reg.path if _source_reg else '?'}")
        env._reset_idx = _scene_reset_idx

    # Ground cuboid를 검정색으로 변경 (visual material만, physics는 건드리지 않음)
    import omni.usd
    from pxr import UsdShade, Sdf, Gf
    stage = omni.usd.get_context().get_stage()
    # Cuboid의 visual mesh에 material 적용
    _cube_vis = stage.GetPrimAtPath("/World/ground/geometry/mesh")
    if not _cube_vis.IsValid():
        _cube_vis = stage.GetPrimAtPath("/World/ground")
    if _cube_vis.IsValid():
        mtl_path = "/World/Looks/BlackMatte"
        UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl = UsdShade.Material.Get(stage, mtl_path)
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(_cube_vis).Bind(mtl)
        print("  [Ground] Black matte material applied to cube")

    base_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (args.camera_width, args.camera_height),
    )
    wrist_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera",
        (args.camera_width, args.camera_height),
    )

    base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    base_rgb_annot.attach([base_rp])
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([base_rp])
    wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_rgb_annot.attach([wrist_rp])

    cams = {"base_rgb": base_rgb_annot, "depth": depth_annot, "wrist_rgb": wrist_rgb_annot}
    return env, cams, scene_path


def reset_with_scene(env, args, scene_path, log):
    """env.reset() 후 scene 내부에 로봇/물체 배치 (같은 방 보장)."""
    from procthor_scene import (
        sample_scene_task_layout, apply_scene_task_layout, SceneSpawnCfg,
        _load_floor_regions, _load_support_floor_z, _find_robot_region,
        SCENE_PRESETS,
    )
    obs, info = env.reset()
    if scene_path is not None:
        ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
        source_override = SceneSpawnCfg(
            min_robot_dist=float(getattr(env.cfg, "object_dist_min", 0.8)) / ss,
            max_robot_dist=float(getattr(env.cfg, "object_dist_max", 1.2)) / ss,
            clearance_radius=0.14,
        )
        _preset = SCENE_PRESETS.get(args.scene_idx)
        _sfz = _load_support_floor_z(str(scene_path.resolve()), _preset.support_floor_prim_path)
        _regions = _load_floor_regions(str(scene_path.resolve()), support_floor_z=_sfz)

        for _retry in range(50):
            try:
                layout = sample_scene_task_layout(
                    args.scene_idx, scene_usd=scene_path,
                    scene_scale=args.scene_scale,
                    source_spawn_override=source_override,
                    robot_faces_source=True,
                    randomize_robot_xy=True,
                )
                # 같은 방 검증
                _robot_unscaled = (layout.robot_xy[0] / ss, layout.robot_xy[1] / ss)
                _source_unscaled = (layout.source_xy[0] / ss, layout.source_xy[1] / ss)
                _robot_reg = _find_robot_region(_robot_unscaled, _regions)
                _source_reg = _find_robot_region(_source_unscaled, _regions)
                if _robot_reg and _source_reg and _robot_reg.path == _source_reg.path:
                    break
                if _retry < 49:
                    continue
            except RuntimeError:
                if _retry < 49:
                    continue
                raise
        apply_scene_task_layout(env, layout)
        for _ in range(10):
            env.sim.step()
        env.robot.update(env.sim.cfg.dt)
        log.info(f"[Scene] Robot: ({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}), "
                 f"Source: ({layout.source_xy[0]:.2f}, {layout.source_xy[1]:.2f}), "
                 f"Dest: ({layout.dest_xy[0]:.2f}, {layout.dest_xy[1]:.2f}), "
                 f"Room: {_robot_reg.path.split('/')[-1] if _robot_reg else '?'}")
    return obs


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu") if args.scene_idx > 0 else torch.device("cuda:0")

    # ── 로그 ──
    log_file = args.log_file or f"eval_viva_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log = setup_logger(log_file)
    log.info(f"Log file: {log_file}")

    # ── 서버 헬스 체크 ──
    log.info("=" * 60)
    log.info("VIVA Pipeline Eval (manual S1 + RL expert S2)")
    log.info(f"VLM: {args.vlm_server}")
    log.info(f"VLA: {args.vla_server}")

    try:
        r = requests.get(f"{args.vlm_server}/v1/models", timeout=3)
        model_id = r.json()['data'][0]['id']
        log.info(f"[CHECK] VLM: OK ({model_id})")
    except Exception as e:
        log.error(f"[CHECK] VLM: FAIL — {e}")
        log.error("→ ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 ...")
        simulation_app.close()
        return

    vla = VLAClient(args.vla_server, jpeg_quality=80)
    vla_health = vla.health_check()
    if vla_health:
        log.info(f"[CHECK] VLA: OK ({vla_health['model']}, {vla_health['gpu_memory_mb']:.0f}MB)")
    else:
        log.error("[CHECK] VLA: FAIL")
        simulation_app.close()
        return

    # ── Classify ──
    log.info(f"[Classify] User command: \"{args.user_command}\"")
    task_info = classify_user_request(args.vlm_server, args.vlm_model, args.user_command)
    source = task_info["source_object"]
    dest = task_info.get("dest_object", "")
    log.info(f"[Classify] Result: mode={task_info['mode']}, source=\"{source}\", dest=\"{dest}\"")

    # ── Per-dim action scales (학습과 동일) ──
    _nav_per_dim = torch.zeros(9, device=device)
    _nav_per_dim[6:9] = 0.25  # navigate: base only

    _s2_per_dim = torch.zeros(9, device=device)
    _s2_per_dim[0:5] = 0.20   # arm
    _s2_per_dim[5] = 0.25     # gripper
    _s2_per_dim[6:9] = 0.35   # base

    # ── S1 Navigate Expert ──
    nav_dp_agent, nav_residual, nav_dp_cfg = None, None, None
    if args.nav_dp_checkpoint:
        log.info(f"[S1 Expert] Loading BC: {args.nav_dp_checkpoint}")
        nav_dp_agent, nav_residual, nav_dp_cfg = load_s2_expert(
            args.nav_dp_checkpoint, args.nav_resip_checkpoint, device
        )
        if nav_dp_agent:
            log.info(f"[S1 Expert] BC loaded (obs={nav_dp_cfg['obs_dim']}D, act={nav_dp_cfg['act_dim']}D)")
            if nav_residual:
                log.info(f"[S1 Expert] ResiP loaded: {args.nav_resip_checkpoint}")
        else:
            log.warning("[S1 Expert] Failed to load — S1 keyboard only")

    # ── S2 Expert ──
    dp_agent, residual, dp_cfg = None, None, None
    if args.dp_checkpoint:
        log.info(f"[S2 Expert] Loading BC: {args.dp_checkpoint}")
        dp_agent, residual, dp_cfg = load_s2_expert(
            args.dp_checkpoint, args.resip_checkpoint, device
        )
        if dp_agent:
            log.info(f"[S2 Expert] BC loaded (obs={dp_cfg['obs_dim']}D, act={dp_cfg['act_dim']}D)")
            if residual:
                log.info(f"[S2 Expert] ResiP loaded: {args.resip_checkpoint}")
            else:
                log.info("[S2 Expert] BC only (no residual)")
        else:
            log.warning("[S2 Expert] Failed to load — S2 will use VLA only")

    # ── 환경 ──
    log.info("Setting up environment...")
    env, cams, scene_path = setup_env(args)
    obs = reset_with_scene(env, args, scene_path, log)

    # DP action buffer reset
    if nav_dp_agent:
        nav_dp_agent.reset()
    if dp_agent:
        dp_agent.reset()

    # 물리적 arm pose: TUCKED_POSE + j3=-0.4 (카메라 가림 방지)
    _TUCKED_ARM = [-0.02966, -0.213839, 0.09066, -0.4, 0.058418]
    _TUCKED_GRIP = -0.201554
    # Policy obs용 arm: TUCKED_POSE (BC normalizer와 일치)
    _TUCKED_ARM_OBS = torch.tensor([[-0.02966, -0.213839, 0.09066, 0.120177, 0.058418]], device="cpu")
    _TUCKED_GRIP_OBS = torch.tensor([[-0.201554]], device="cpu")
    env_id = torch.tensor([0], device=env.device)
    jp = env.robot.data.joint_pos[0:1].clone()
    jp[0, env.arm_idx[:5]] = torch.tensor(_TUCKED_ARM, device=env.device)
    jp[0, env.gripper_idx] = _TUCKED_GRIP
    jp[0, env.wheel_idx] = 0.0
    jv = torch.zeros_like(jp)
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)
    env.robot.set_joint_position_target(jp, env_ids=env_id)
    for _ in range(30):
        env.robot.write_data_to_sim()
        env.sim.step()
        env.sim.render()

    # ── Arm tucked pose 강제 (Skill1Env 방식 monkey-patch) ──
    _TUCKED_ARM_T = torch.tensor(
        [-0.02966, -0.213839, 0.09066, -0.4, 0.058418], device=env.device)
    _TUCKED_GRIP_V = -0.201554
    _original_apply_action = env._apply_action
    _force_tucked = [True]  # mutable for closure

    def _patched_apply_action():
        if _force_tucked[0]:
            # S1: Skill1Env._apply_action과 동일한 로직으로 완전 대체
            # Base: body frame → IK → wheel
            body_vx = env.actions[:, 6] * env.cfg.max_lin_vel
            body_vy = env.actions[:, 7] * env.cfg.max_lin_vel
            body_wz = env.actions[:, 8] * env.cfg.max_ang_vel
            ik_vx = body_vy
            ik_vy = -body_vx
            ik_wz = body_wz
            ik_cmd = torch.stack([ik_vx, ik_vy, ik_wz], dim=-1)
            wheel_radps = ik_cmd @ env.kiwi_M.T / env.wheel_radius
            vel_target = torch.zeros(1, env.robot.num_joints, device=env.device)
            vel_target[:, env.wheel_idx] = wheel_radps
            env.robot.set_joint_velocity_target(vel_target)
            # Arm: tucked pose 직접 설정 (Skill1Env force_tucked_pose 방식)
            pos_target = torch.zeros(1, env.robot.num_joints, device=env.device)
            pos_target[0, env.arm_idx[:5]] = _TUCKED_ARM_T
            pos_target[0, env.arm_idx[5]] = _TUCKED_GRIP_V
            env.robot.set_joint_position_target(pos_target)
        else:
            # S2/S4: 원래 Skill2Env._apply_action 사용
            _original_apply_action()

    env._apply_action = _patched_apply_action

    # ── Orchestrator ──
    orch = VIVAOrchestrator(
        vlm_server=args.vlm_server, vlm_model=args.vlm_model,
        source_object=source, dest_object=dest,
        user_request=args.user_command, jpeg_quality=80,
    )

    # ── 키보드 ──
    kb_setup()
    BASE_LIN = 0.15  # m/s
    BASE_ANG = 1.0   # rad/s
    MAX_LIN = 0.5
    MAX_ANG = 3.0

    log.info("=" * 60)
    log.info("Controls (hold for continuous movement):")
    log.info("  W/↑=forward  S/↓=backward  A/←=strafe left  D/→=strafe right")
    log.info("  Z=turn left  X=turn right  SPACE=stop  Q=quit  T=force TARGET_FOUND")
    log.info("  1=force S1  2=force S2  3=force S3  4=force S4")
    log.info("=" * 60)

    prev_skill = orch.current_skill
    total_steps = 0
    safety_stops = 0
    _s2_lift_counter = 0
    _s2_step_counter = 0
    vlm_calls_log = []
    vla_calls_log = []
    t_start = time.time()

    try:
        while total_steps < args.max_total_steps and simulation_app.is_running():
            # ── 키보드 (hold 감지) ──
            # one-shot 명령 처리
            while _commands:
                cmd = _commands.pop(0)
                if cmd == 'q':
                    log.info("[USER] Quit")
                    raise KeyboardInterrupt
                elif cmd == 'r':
                    log.info("[USER] Manual reset")
                    obs = reset_with_scene(env, args, scene_path, log)
                    vla.reset_buffer()
                    orch.reset_for_new_trial()
                    _force_tucked[0] = True
                    prev_skill = SkillState.NAVIGATE
                    continue
                elif cmd == 't':
                    log.info("[USER] Force TARGET_FOUND")
                    orch._handle_target_found()
                elif cmd == '1':
                    log.info("[USER] Force → S1 Navigate")
                    orch._transition_to(SkillState.NAVIGATE)
                elif cmd == '2':
                    log.info("[USER] Force → S2 Approach & Lift")
                    orch._transition_to(SkillState.APPROACH_AND_LIFT)
                elif cmd == '3':
                    log.info("[USER] Force → S3 Carry")
                    orch._transition_to(SkillState.CARRY)
                elif cmd == '4':
                    log.info("[USER] Force → S4 Approach & Place")
                    orch._transition_to(SkillState.APPROACH_AND_PLACE)

            # 연속 입력 (hold)
            kb_vx = kb_vy = kb_wz = 0.0
            if _pressed.get('space'):
                pass  # 정지
            else:
                if _pressed.get('w'): kb_vy += BASE_LIN
                if _pressed.get('s'): kb_vy -= BASE_LIN
                if _pressed.get('a'): kb_vx -= BASE_LIN
                if _pressed.get('d'): kb_vx += BASE_LIN
                if _pressed.get('z'): kb_wz -= BASE_ANG
                if _pressed.get('x'): kb_wz += BASE_ANG

            # ── 카메라 ──
            base_rgb, depth, wrist_rgb = capture(env, cams)
            if base_rgb is None or wrist_rgb is None:
                env.sim.step()
                total_steps += 1
                continue

            # VLM용 crop+resize (0.7 scale → 1.0 시뮬레이션)
            h, w = base_rgb.shape[:2]
            crop_ratio = 0.7
            mh = int(h * (1 - crop_ratio) / 2)
            mw = int(w * (1 - crop_ratio) / 2)
            vlm_rgb = np.array(Image.fromarray(
                base_rgb[mh:h-mh, mw:w-mw]).resize((w, h)))

            depth_min = get_depth_min(depth)

            # ── 로봇 상태 ──
            contact = get_contact_detected(env)
            robot_status = build_robot_status(env, contact, depth_min)
            orch.update_robot_status(robot_status)
            orch.update_contact(contact)
            orch.update_depth_status(depth_min)  # CONTINUE 후 새 장애물 감지
            orch.tick()

            jp = env.robot.data.joint_pos[0]
            arm_joints = jp[env.arm_idx[:5]].tolist()
            grip_pos = jp[env.gripper_idx].item()

            # S2: lift 판정 + 실패 시 리셋 (train_resip combined 기준)
            if orch.current_skill == SkillState.APPROACH_AND_LIFT:
                obj_z = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                _s2_step_counter += 1

                # lift 성공: objZ>0.05 + contact + 200step 유지 → S3
                if contact and obj_z > 0.05:
                    _s2_lift_counter += 1
                else:
                    _s2_lift_counter = 0
                if _s2_lift_counter >= 200:
                    log.info(f"[LIFTED] objZ={obj_z:.3f}, hold={_s2_lift_counter} → S3")
                    orch._transition_to(SkillState.CARRY)
                    _s2_lift_counter = 0
                    _s2_step_counter = 0

                # 실패: 700step 내 lift 못하면 (objZ<0.04) 리셋
                if _s2_step_counter >= 700 and obj_z < 0.04:
                    log.info(f"[S2 FAIL] objZ={obj_z:.3f} at {_s2_step_counter} steps → reset")
                    obs = reset_with_scene(env, args, scene_path, log)
                    vla.reset_buffer()
                    orch.reset_for_new_trial()
                    _force_tucked[0] = True
                    prev_skill = SkillState.NAVIGATE
                    _s2_lift_counter = 0
                    _s2_step_counter = 0
                    continue

                # topple: objZ<0.026 → 리셋
                if obj_z < 0.026 and _s2_step_counter > 10:
                    log.info(f"[S2 TOPPLE] objZ={obj_z:.3f} → reset")
                    obs = reset_with_scene(env, args, scene_path, log)
                    vla.reset_buffer()
                    orch.reset_for_new_trial()
                    _force_tucked[0] = True
                    prev_skill = SkillState.NAVIGATE
                    _s2_lift_counter = 0
                    _s2_step_counter = 0
                    continue

            # arm/grip 캐시 갱신 (OBSTACLE 시 _is_actually_lifted()에 필요)
            orch.check_lifted_complete(arm_joints, grip_pos, contact)
            orch.check_place_complete(grip_pos, contact)

            # ── VLM 호출 ──
            skill = orch.current_skill
            if skill in (SkillState.NAVIGATE, SkillState.CARRY):
                # depth warning 시에는 즉시 호출 (forward 차단 상태에서 빠르게 새 방향 획득)
                depth_urgent = (depth_min is not None and depth_min < args.safety_dist)
                if total_steps % args.vlm_interval == 0 or depth_urgent:
                    orch.query_async(vlm_rgb)
                    log.debug(f"[VLM-CALL] step={total_steps} skill={skill.value} "
                              f"call_count={orch.call_count}")
            elif skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
                if depth_min is not None and depth_min < args.safety_dist and not orch.obstacle_cleared:
                    orch.query_obstacle_check_async(base_rgb)
                    log.debug(f"[VLM-OBSTACLE] step={total_steps} skill={skill.value} "
                              f"depth_min={depth_min:.3f}")

            # ── 종료 체크 ──
            if orch.is_done:
                log.info(f"[DONE] Task complete at step {total_steps}")
                break
            # 전체 timeout 비활성화 (S2 실패는 위에서 개별 처리)

            # ── 스킬 전환 감지 ──
            if prev_skill != orch.current_skill:
                log.info(f"[SKILL] {prev_skill.value} → {orch.current_skill.value}")
                vla.reset_buffer()
                prev_skill = orch.current_skill
                # S1/S3: tucked 강제, S2/S4: 해제 (RL expert가 arm 제어)
                _force_tucked[0] = orch.current_skill in (SkillState.NAVIGATE, SkillState.CARRY)

            # ── Action 결정 ──
            instruction = orch.instruction
            state = get_state_9d(env)
            skill = orch.current_skill

            # VLM instruction → direction_cmd 매핑 (navigate expert용)
            _NAV_INST_TO_DIR = {
                "navigate forward":      [0.0, 1.0, 0.0],
                "navigate backward":     [0.0, -1.0, 0.0],
                "navigate strafe left":  [-1.0, 0.0, 0.0],
                "navigate strafe right": [1.0, 0.0, 0.0],
                "navigate turn left":    [0.0, 0.0, 1.0],
                "navigate turn right":   [0.0, 0.0, -1.0],
            }

            # Lookup table: VLM instruction → 고정 base velocity (normalized action)
            _NAV_INST_TO_ACTION = {
                "navigate forward":      [0.0, 0.5, 0.0],
                "navigate backward":     [0.0, -0.5, 0.0],
                "navigate strafe left":  [-0.5, 0.0, 0.0],
                "navigate strafe right": [0.5, 0.0, 0.0],
                "navigate turn left":    [0.0, 0.0, -0.33],
                "navigate turn right":   [0.0, 0.0, 0.33],
            }

            if skill in (SkillState.NAVIGATE, SkillState.CARRY):
                # Lookup table base velocity (friction 0.5 호환)
                _base_cmd = _NAV_INST_TO_ACTION.get(instruction, [0.0, 0.5, 0.0])
                action = np.zeros(9, dtype=np.float32)
                action[6] = _base_cmd[0]
                action[7] = _base_cmd[1]
                action[8] = _base_cmd[2]
                # 키보드 override
                if abs(kb_vx) > 0 or abs(kb_vy) > 0 or abs(kb_wz) > 0:
                    action[6] = np.clip(kb_vx / MAX_LIN, -1, 1)
                    action[7] = np.clip(kb_vy / MAX_LIN, -1, 1)
                    action[8] = np.clip(kb_wz / MAX_ANG, -1, 1)

                # VLA 포맷 검증: 100스텝마다 1회 비동기 호출 (별도 스레드)
                if total_steps % 100 == 0 and total_steps > 0:
                    def _vla_check():
                        va = vla.query_action(base_rgb, wrist_rgb, state, instruction)
                        dim = len(va[0]) if va and len(va) > 0 else 0
                        log.info(f"[VLA-CHECK] step={total_steps} inst=\"{instruction[:40]}\" "
                                 f"chunk={len(va)} dim={dim} latency={vla.latency*1000:.0f}ms")
                    threading.Thread(target=_vla_check, daemon=True).start()

            elif skill == SkillState.APPROACH_AND_LIFT and dp_agent is not None:
                # S2: RL expert action
                obs_t = obs["policy"].to(device)
                action = resip_action(dp_agent, residual, obs_t, device, _s2_per_dim)
                action = np.clip(action, -1, 1)

                # VLA 포맷 검증: 100스텝마다 비동기
                if total_steps % 100 == 0:
                    def _vla_check_s2():
                        va = vla.query_action(base_rgb, wrist_rgb, state, instruction)
                        dim = len(va[0]) if va and len(va) > 0 else 0
                        log.info(f"[VLA-S2] step={total_steps} inst=\"{instruction[:40]}\" "
                                 f"chunk={len(va)} dim={dim}")
                    threading.Thread(target=_vla_check_s2, daemon=True).start()

            else:
                # S2 expert 없거나 S3/S4: VLA action 그대로
                action = vla.get_action_9d(base_rgb, wrist_rgb, state, instruction)
                if total_steps % 10 == 0:
                    log.debug(f"[VLA] step={total_steps} skill={skill.value} "
                              f"inst=\"{instruction[:40]}\" "
                              f"action={[round(float(a),4) for a in action[:9]]}")

            # ── Safety layer ──
            stopped = False
            if depth_min is not None and orch.safety_layer_active and depth_min < args.safety_dist:
                action = action.copy()
                action[6:8] = 0.0
                stopped = True
                safety_stops += 1

            # ── env step ──
            action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            action_t = action_t.clamp(-1, 1)
            obs, rew, term, trunc, info = env.step(action_t)
            total_steps += 1


            # ── 로그 (vlm_interval마다) ──
            if total_steps % args.vlm_interval == 0 and total_steps > 0:
                elapsed = time.time() - t_start
                hz = total_steps / elapsed if elapsed > 0 else 0
                stop_str = " [SAFETY]" if stopped else ""
                depth_str = f" depth={depth_min:.2f}" if depth_min else ""
                contact_str = " CONTACT" if contact else ""
                base_act = f" act=[{action[6]:.3f},{action[7]:.3f},{action[8]:.3f}]"
                # 로봇 위치/yaw
                _rpos = env.robot.data.root_pos_w[0]
                _rquat = env.robot.data.root_quat_w[0]
                _ryaw = 2.0 * torch.atan2(_rquat[3], _rquat[0]).item()
                _ryaw_deg = math.degrees(_ryaw)
                _pos_str = f" pos=({_rpos[0]:.2f},{_rpos[1]:.2f}) yaw={_ryaw_deg:.1f}"
                log.info(
                    f"\n[t={total_steps:4d} {elapsed:.0f}s {hz:.1f}Hz] "
                    f"skill={orch.current_skill.value} "
                    f"inst=\"{instruction[:40]}\" "
                    f"vlm={orch.avg_latency*1000:.0f}ms({orch.call_count})"
                    f"{base_act}{_pos_str}{depth_str}{contact_str}{stop_str}"
                )

            if term.any() or trunc.any():
                log.info(f"[EPISODE END] step={total_steps}")
                obs = reset_with_scene(env, args, scene_path, log)
                vla.reset_buffer()

    except KeyboardInterrupt:
        log.info("[Ctrl+C] Stopping")

    finally:
        kb_restore()
        elapsed = time.time() - t_start
        log.info("=" * 60)
        log.info(f"Summary: {total_steps} steps, {elapsed:.0f}s, "
                 f"VLM {orch.call_count} calls, safety {safety_stops}")
        log.info(f"Final skill: {orch.current_skill.value}")
        log.info(f"Log saved: {log_file}")
        log.info("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()
