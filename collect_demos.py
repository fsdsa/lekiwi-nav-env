#!/usr/bin/env python3
"""
LeKiwi Navigation — RL Expert 데모 수집 (v8: SpawnManager/Physics + 메타데이터 + 품질 필터).

학습된 PPO 정책을 실행하여 성공 에피소드만 HDF5로 저장.
base_rgb (1280x720) + wrist_rgb (640x480) 이미지를 함께 저장.
SpawnManager로 object_pos_w 좌표에 실제 USD 물체를 스폰하여 카메라에 물체가 찍히게 한다.

HDF5 구조 (v8):
    attrs:
      has_camera, has_subtask_annotation, has_spawn_metadata
      subtask_id_to_text_json (global fallback)
      objects_index_path

    episode_N/
        obs              (T, obs_dim)       float32
        actions          (T, 9)             float32
        robot_state      (T, 9)             float32
        subtask_ids      (T,)               int64    [optional]
        images/
          base_rgb       (T, 720, 1280, 3)  uint8  gzip-4
          wrist_rgb      (T, 480, 640, 3)   uint8  gzip-4
        attrs:
          num_steps, final_object_dist, final_dist(legacy alias), success
          object_name, object_usd, object_scale
          instruction
          spawn_meta_json
          subtask_transitions
          quality_check
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="LeKiwi Nav - RL Expert 데모 수집 (v8)")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_demos", type=int, default=20)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="카메라 사용 시 1~8 권장 (VRAM), --no_camera 시 64+",
)
parser.add_argument("--max_attempts", type=int, default=500)
parser.add_argument("--output", type=str, default=None)
parser.add_argument(
    "--dynamics_json",
    type=str,
    default=None,
    help="tune_sim_dynamics.py 출력 JSON (best_params) 경로",
)
parser.add_argument(
    "--calibration_json",
    type=str,
    default=None,
    help="calibration JSON 경로 (wheel/base geometry override)",
)
parser.add_argument("--arm_limit_json", type=str, default=None, help="optional arm joint limit JSON (real2sim calibration)")
parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0, help="margin added to arm limits from --arm_limit_json")
parser.add_argument("--object_usd", type=str, default="", help="physics grasp object USD path (empty = legacy proximity grasp)")
parser.add_argument("--multi_object_json", type=str, default="", help="multi-object catalog JSON path for privileged 37D teacher obs")
parser.add_argument("--object_mass", type=float, default=0.3, help="physics grasp object mass (kg)")
parser.add_argument("--object_scale_phys", type=float, default=1.0, help="physics grasp object uniform scale")
parser.add_argument(
    "--gripper_contact_prim_path",
    type=str,
    default="",
    help="contact sensor prim path for gripper body (required in physics grasp mode)",
)
parser.add_argument("--grasp_gripper_threshold", type=float, default=-0.3, help="gripper joint position threshold for closed state")
parser.add_argument("--grasp_contact_threshold", type=float, default=0.5, help="minimum contact force magnitude for grasp success")
parser.add_argument("--grasp_max_object_dist", type=float, default=0.25, help="max object distance for contact-based grasp success")
parser.add_argument("--grasp_attach_height", type=float, default=0.15, help="attached object z-height after grasp success")
parser.add_argument(
    "--annotate_subtasks",
    action="store_true",
    help="v6 annotation (robot_state 9D + subtask_ids + transitions)",
)
parser.add_argument(
    "--allow_missing_state_preprocessor",
    action="store_true",
    help="checkpoint에 state preprocessor가 없을 때 raw obs로 진행 (기본: 에러로 중단)",
)

# Camera
parser.add_argument("--no_camera", action="store_true", help="카메라 없이 수집 (state-only)")
parser.add_argument("--base_cam_width", type=int, default=1280)
parser.add_argument("--base_cam_height", type=int, default=720)
parser.add_argument("--wrist_cam_width", type=int, default=640)
parser.add_argument("--wrist_cam_height", type=int, default=480)

# SpawnManager
parser.add_argument(
    "--objects_index",
    type=str,
    default=None,
    help="mujoco_obj_usd_index_all.jsonl 경로 (없으면 스폰 비활성)",
)
parser.add_argument("--object_scale", type=float, default=0.7)
parser.add_argument("--object_cap", type=int, default=0, help="물체별 최대 에피소드 수 (0=무제한)")
parser.add_argument("--min_steps", type=int, default=20, help="최소 에피소드 길이 (품질 필터)")

dr_group = parser.add_mutually_exclusive_group()
dr_group.add_argument("--dr_lighting", dest="dr_lighting", action="store_true", help="조명 Domain Randomization ON")
dr_group.add_argument("--no_dr_lighting", dest="dr_lighting", action="store_false", help="조명 Domain Randomization OFF")
parser.set_defaults(dr_lighting=True)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import h5py
import numpy as np
import torch

from isaaclab.sensors import Camera, CameraCfg

from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
from models import PolicyNet, ValueNet
from spawn_manager import SpawnManager, build_subtask_transitions

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler


ENV_PRIM = "/World/envs/env_.*/Robot"

BASE_RGB_CAM_PRIM = (
    f"{ENV_PRIM}/base_plate_layer1_v5/Realsense/RSD455"
    f"/Camera_OmniVision_OV9782_Color"
)
WRIST_CAM_PRIM = (
    f"{ENV_PRIM}/Wrist_Roll_08c_v1/visuals/mesh_002_3"
    f"/wrist_camera"
)

SUBTASK_ID_TO_TEXT_STATIC = {
    0: "look around to find the target object",
    1: "approach the target object",
    2: "pick up the target object",
    3: "return to the starting position",
}
FULL_TASK_ID = 10
FULL_TASK_TEXT_STATIC = "find the target and bring it back"


def _sanitize_object_name(raw: object) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return "target object"
    txt = txt.replace("_", " ").replace("-", " ")
    txt = " ".join(txt.split())
    return txt if txt else "target object"


def _full_task_text_for_object(object_name: object) -> str:
    name = _sanitize_object_name(object_name)
    lowered = name.lower()
    if lowered in {"target object", "object"}:
        return FULL_TASK_TEXT_STATIC
    return f"find the {name} and bring it back"


def _get_multi_object_episode_meta(env: LeKiwiNavEnv, env_id: int) -> dict[str, object]:
    meta = {
        "object_name": "target object",
        "object_usd": "",
        "object_scale": 1.0,
        "object_category_name": "",
        "instruction": FULL_TASK_TEXT_STATIC,
    }
    if not hasattr(env, "active_object_idx"):
        return meta
    active_idx = int(env.active_object_idx[env_id].item())
    catalog = list(getattr(env, "_object_catalog", []))
    if active_idx < 0 or active_idx >= len(catalog):
        meta["object_name"] = f"object {active_idx}"
        meta["instruction"] = _full_task_text_for_object(meta["object_name"])
        return meta

    entry = catalog[active_idx]
    if not isinstance(entry, dict):
        meta["object_name"] = f"object {active_idx}"
        meta["instruction"] = _full_task_text_for_object(meta["object_name"])
        return meta

    name_candidate = ""
    for k in ("instruction_name", "display_name", "name", "category_name", "label"):
        raw = str(entry.get(k, "")).strip()
        if raw:
            name_candidate = raw
            break
    if not name_candidate:
        name_candidate = f"object {active_idx}"

    meta["object_name"] = _sanitize_object_name(name_candidate)
    meta["instruction"] = _full_task_text_for_object(meta["object_name"])
    meta["object_usd"] = str(entry.get("usd", "") or "")
    try:
        meta["object_scale"] = float(entry.get("scale", 1.0))
    except (TypeError, ValueError):
        meta["object_scale"] = 1.0
    category_name = str(entry.get("category_name", "") or "").strip()
    if category_name:
        meta["object_category_name"] = _sanitize_object_name(category_name)
    return meta


class LeKiwiNavEnvWithCam(LeKiwiNavEnv):
    """LeKiwiNavEnv + base_rgb/wrist RGB 카메라."""

    def __init__(
        self,
        cfg,
        base_cam_w: int = 1280,
        base_cam_h: int = 720,
        wrist_cam_w: int = 640,
        wrist_cam_h: int = 480,
        render_mode=None,
        **kwargs,
    ):
        self._base_cam_w = base_cam_w
        self._base_cam_h = base_cam_h
        self._wrist_cam_w = wrist_cam_w
        self._wrist_cam_h = wrist_cam_h
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        # Camera prims already exist inside the cloned robot USD (spawn=None),
        # so camera wrappers are attached after super()._setup_scene() clone/registration.

        base_cam_cfg = CameraCfg(
            prim_path=BASE_RGB_CAM_PRIM,
            spawn=None,
            update_period=0.0,
            height=self._base_cam_h,
            width=self._base_cam_w,
            data_types=["rgb"],
        )
        self.base_cam = Camera(base_cam_cfg)
        self.scene.sensors["base_cam"] = self.base_cam

        wrist_cam_cfg = CameraCfg(
            prim_path=WRIST_CAM_PRIM,
            spawn=None,
            update_period=0.0,
            height=self._wrist_cam_h,
            width=self._wrist_cam_w,
            data_types=["rgb"],
        )
        self.wrist_cam = Camera(wrist_cam_cfg)
        self.scene.sensors["wrist_cam"] = self.wrist_cam

        print(f"  [Camera] base_rgb : {self._base_cam_w}x{self._base_cam_h}")
        print(f"  [Camera] wrist_rgb: {self._wrist_cam_w}x{self._wrist_cam_h}")

    def _extract_rgb(self, camera: Camera) -> torch.Tensor | None:
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        return rgb[:, :, :, :3]

    def get_base_rgb(self) -> torch.Tensor | None:
        return self._extract_rgb(self.base_cam)

    def get_wrist_rgb(self) -> torch.Tensor | None:
        return self._extract_rgb(self.wrist_cam)


def _extract_first_tensor(payload):
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, (tuple, list)):
        for item in payload:
            found = _extract_first_tensor(item)
            if found is not None:
                return found
    if isinstance(payload, dict):
        for key in ("actions", "action", "mean_actions"):
            if key in payload and torch.is_tensor(payload[key]):
                return payload[key]
        for item in payload.values():
            found = _extract_first_tensor(item)
            if found is not None:
                return found
    return None


def compute_deterministic_action(
    obs_policy: torch.Tensor,
    policy_model: PolicyNet,
    state_preprocessor,
    agent=None,
) -> torch.Tensor:
    if state_preprocessor is not None:
        proc_obs = state_preprocessor(obs_policy, train=False)
    else:
        proc_obs = obs_policy

    if agent is not None and callable(getattr(agent, "act", None)):
        state_dict = {"states": proc_obs}
        act_calls = [
            lambda: agent.act(state_dict, timestep=0, timesteps=1, deterministic=True),
            lambda: agent.act(state_dict, timestep=0, timesteps=1),
            lambda: agent.act(state_dict, deterministic=True),
            lambda: agent.act(state_dict),
            lambda: agent.act(proc_obs),
        ]
        for call in act_calls:
            try:
                out = call()
            except TypeError:
                continue
            except Exception:
                continue
            action_tensor = _extract_first_tensor(out)
            if torch.is_tensor(action_tensor) and action_tensor.ndim >= 2 and action_tensor.shape[0] == proc_obs.shape[0]:
                return action_tensor.clamp(-1.0, 1.0)

    # Prefer skrl model API (stable), fallback to legacy architecture path.
    try:
        out = policy_model.compute({"states": proc_obs}, role="policy")
        if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0].clamp(-1.0, 1.0)
    except Exception:
        pass

    feat = policy_model.net(proc_obs)
    return policy_model.mean_layer(feat).clamp(-1.0, 1.0)


def resolve_state_preprocessor(agent, allow_missing: bool):
    """skrl 버전 차이에 따른 state preprocessor 속성명을 안전하게 탐색."""
    candidates = [
        getattr(agent, "_state_preprocessor", None),
        getattr(agent, "state_preprocessor", None),
    ]
    pre = next((c for c in candidates if callable(c)), None)
    if pre is not None:
        return pre

    if allow_missing:
        print("  [WARN] state preprocessor를 찾지 못해 raw obs를 사용합니다.")
        return None

    raise RuntimeError(
        "state preprocessor를 찾지 못했습니다. "
        "skrl 버전/체크포인트 호환을 확인하거나 "
        "--allow_missing_state_preprocessor 로 명시적으로 raw obs를 허용하세요."
    )


def extract_robot_state_9d(env: LeKiwiNavEnv) -> torch.Tensor:
    """VLA 입력용 robot_state (N, 9): arm_pos(6) + wheel_vel(3)."""
    arm_pos = env.robot.data.joint_pos[:, env.arm_idx]
    wheel_vel = env.robot.data.joint_vel[:, env.wheel_idx]
    return torch.cat([arm_pos, wheel_vel], dim=-1)


def resolve_checkpoint_path(path: str) -> str:
    """best_agent.pt 경로를 관용적으로 해석."""
    if os.path.isfile(path):
        return path

    candidates = [
        os.path.join(path, "best_agent.pt"),
        os.path.join(path, "checkpoints", "best_agent.pt"),
    ]
    parent = os.path.dirname(path)
    candidates.extend(
        [
            os.path.join(parent, "best_agent.pt"),
            os.path.join(parent, "checkpoints", "best_agent.pt"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    found = []
    for root in [path, parent, os.getcwd()]:
        if not root:
            continue
        found.extend(glob.glob(os.path.join(root, "**", "best_agent.pt"), recursive=True))
    found = [p for p in found if os.path.isfile(p)]
    if found:
        found.sort(key=os.path.getmtime, reverse=True)
        return found[0]

    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state, ep_subtask_ids):
    ep_obs[idx].clear()
    ep_act[idx].clear()
    ep_base_img[idx].clear()
    ep_wrist_img[idx].clear()
    ep_robot_state[idx].clear()
    ep_subtask_ids[idx].clear()


def main():
    use_camera = not args.no_camera
    physics_grasp_mode = bool(str(args.object_usd).strip()) or bool(str(args.multi_object_json).strip())
    multi_object_mode = bool(str(args.multi_object_json).strip())
    use_spawn = args.objects_index is not None and use_camera and (not physics_grasp_mode)
    if args.objects_index and not use_camera:
        print("  [WARN] --objects_index가 지정됐지만 --no_camera라 SpawnManager를 비활성화합니다.")
    if args.objects_index and physics_grasp_mode:
        print("  [WARN] physics grasp 모드에서는 SpawnManager를 비활성화합니다 (--objects_index 무시).")
    use_dr_lighting = bool(args.dr_lighting and use_spawn)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if args.output:
        output_path = args.output
    else:
        os.makedirs("outputs/rl_demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_cam" if use_camera else ""
        if use_spawn:
            suffix += "_spawn"
        output_path = f"outputs/rl_demos/rl_expert{suffix}_{timestamp}.hdf5"

    spawn_mgr: SpawnManager | None = None
    if use_spawn:
        index_path = os.path.expanduser(args.objects_index)
        spawn_mgr = SpawnManager(
            index_path=index_path,
            num_envs=args.num_envs,
            object_scale=args.object_scale,
            object_cap=args.object_cap,
        )

    print("\n" + "=" * 60)
    print("  LeKiwi Nav - RL Expert 데모 수집 (v6)")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  목표       : {args.num_demos} 에피소드")
    print(f"  병렬 환경  : {args.num_envs}")
    if use_camera:
        print(
            f"  카메라     : base ({args.base_cam_width}x{args.base_cam_height})"
            f" + wrist ({args.wrist_cam_width}x{args.wrist_cam_height})"
        )
    else:
        print("  카메라     : 없음")
    if use_spawn and spawn_mgr is not None:
        print(f"  SpawnMgr   : {len(spawn_mgr.objects_list)}개 물체")
        print(f"  Object cap : {args.object_cap if args.object_cap > 0 else '무제한'}")
        print(f"  Min steps  : {args.min_steps}")
        print(f"  Lighting DR: {'ON' if use_dr_lighting else 'OFF'}")
    else:
        print("  SpawnMgr   : 비활성")
    if args.dynamics_json:
        print(f"  Dynamics   : {os.path.expanduser(args.dynamics_json)}")
    if args.calibration_json is not None:
        cal_path = str(args.calibration_json).strip()
        if cal_path:
            print(f"  Calibration: {os.path.expanduser(cal_path)}")
        else:
            print("  Calibration: (disabled)")
    if args.arm_limit_json:
        print(
            f"  Arm limits : {os.path.expanduser(args.arm_limit_json)} "
            f"(margin={args.arm_limit_margin_rad:.4f} rad)"
        )
    if multi_object_mode:
        print(f"  Multi-object catalog : {os.path.expanduser(args.multi_object_json)}")
    if physics_grasp_mode and args.object_usd:
        print(f"  Physics obj : {os.path.expanduser(args.object_usd)}")
    if physics_grasp_mode:
        print(f"  Contact prim: {args.gripper_contact_prim_path}")
        print(
            f"  Grasp cond  : gripper<{args.grasp_gripper_threshold:.4f}, "
            f"contact>{args.grasp_contact_threshold:.4f}, "
            f"dist<{args.grasp_max_object_dist:.4f}"
        )
    print(f"  Annotation : {'ON' if args.annotate_subtasks else 'OFF'}")
    print(f"  출력       : {output_path}")
    print("=" * 60 + "\n")

    env_cfg = LeKiwiNavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.calibration_json is not None:
        raw = str(args.calibration_json).strip()
        env_cfg.calibration_json = os.path.expanduser(raw) if raw else ""
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
        env_cfg.arm_limit_margin_rad = float(args.arm_limit_margin_rad)
    if multi_object_mode:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if physics_grasp_mode:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
        env_cfg.object_mass = float(args.object_mass)
        env_cfg.object_scale = float(args.object_scale_phys)
        env_cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
        env_cfg.grasp_gripper_threshold = float(args.grasp_gripper_threshold)
        env_cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)
        env_cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env_cfg.grasp_attach_height = float(args.grasp_attach_height)

    if use_camera:
        env = LeKiwiNavEnvWithCam(
            cfg=env_cfg,
            base_cam_w=args.base_cam_width,
            base_cam_h=args.base_cam_height,
            wrist_cam_w=args.wrist_cam_width,
            wrist_cam_h=args.wrist_cam_height,
        )
    else:
        env = LeKiwiNavEnv(cfg=env_cfg)

    print(
        f"  Geometry in env: wheel={env.wheel_radius:.6f}, "
        f"base={env.base_radius:.6f}"
    )
    if args.dynamics_json:
        print(
            f"  Dynamics limits: lin={env.cfg.max_lin_vel:.4f}, "
            f"ang={env.cfg.max_ang_vel:.4f}"
        )

    wrapped = wrap_env(env, wrapper="isaaclab")
    device = wrapped.device

    models = {
        "policy": PolicyNet(wrapped.observation_space, wrapped.action_space, device),
        "value": ValueNet(wrapped.observation_space, wrapped.action_space, device),
    }
    memory = RandomMemory(memory_size=24, num_envs=args.num_envs, device=device)

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": wrapped.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )
    try:
        agent.load(checkpoint_path)
    except RuntimeError as exc:
        raise RuntimeError(
            "체크포인트와 현재 환경 observation 차원이 맞지 않습니다. "
            "현재 환경에서 다시 학습한 PPO 체크포인트를 사용하세요."
        ) from exc

    agent.set_running_mode("eval")
    state_preprocessor = resolve_state_preprocessor(
        agent, allow_missing=bool(args.allow_missing_state_preprocessor)
    )
    if state_preprocessor is not None:
        print("  상태 전처리기: RunningStandardScaler")
    print("  정책 로드 완료\n")

    obs, info = env.reset()

    if use_spawn and spawn_mgr is not None:
        if use_dr_lighting:
            spawn_mgr.randomize_lighting()
        spawn_mgr.spawn_all(env.object_pos_w)
        print(f"  초기 물체 스폰 완료 ({args.num_envs} envs)\n")

    if args.annotate_subtasks and not hasattr(env, "phase"):
        raise RuntimeError("annotate_subtasks=True 이지만 env.phase가 없습니다.")

    ep_obs = [[] for _ in range(args.num_envs)]
    ep_act = [[] for _ in range(args.num_envs)]
    ep_base_img = [[] for _ in range(args.num_envs)]
    ep_wrist_img = [[] for _ in range(args.num_envs)]
    ep_robot_state = [[] for _ in range(args.num_envs)]
    ep_subtask_ids = [[] for _ in range(args.num_envs)]

    ep_object_name = ["" for _ in range(args.num_envs)]
    ep_spawn_meta = [{} for _ in range(args.num_envs)]
    if use_spawn and spawn_mgr is not None:
        for i in range(args.num_envs):
            ep_object_name[i] = spawn_mgr.get_object_name(i)
            ep_spawn_meta[i] = spawn_mgr.get_spawn_metadata(i)

    saved = 0
    skipped = 0
    attempts = 0
    dists = []

    hdf5_file = h5py.File(output_path, "w")
    hdf5_file.attrs["has_camera"] = use_camera
    hdf5_file.attrs["has_subtask_annotation"] = bool(args.annotate_subtasks)
    hdf5_file.attrs["has_robot_state"] = True
    hdf5_file.attrs["obs_dim"] = int(env.observation_space.shape[0])
    hdf5_file.attrs["action_dim"] = int(env.action_space.shape[0])
    hdf5_file.attrs["has_spawn_metadata"] = bool(use_spawn)
    hdf5_file.attrs["subtask_id_to_text_json"] = json.dumps(SUBTASK_ID_TO_TEXT_STATIC, ensure_ascii=False)
    hdf5_file.attrs["full_task_id"] = int(FULL_TASK_ID)
    hdf5_file.attrs["full_task_text"] = FULL_TASK_TEXT_STATIC
    if use_camera:
        hdf5_file.attrs["base_rgb_shape"] = [args.base_cam_height, args.base_cam_width, 3]
        hdf5_file.attrs["wrist_rgb_shape"] = [args.wrist_cam_height, args.wrist_cam_width, 3]
    if use_spawn and spawn_mgr is not None:
        hdf5_file.attrs["objects_index_path"] = str(os.path.expanduser(args.objects_index))
        hdf5_file.attrs["object_scale"] = float(args.object_scale)
        hdf5_file.attrs["object_cap"] = int(args.object_cap)
        hdf5_file.attrs["min_steps_filter"] = int(args.min_steps)
        hdf5_file.attrs["num_objects_in_library"] = int(len(spawn_mgr.objects_list))
    if physics_grasp_mode:
        hdf5_file.attrs["physics_grasp"] = True
        if args.object_usd:
            hdf5_file.attrs["physics_object_usd"] = str(os.path.expanduser(args.object_usd))
        if multi_object_mode:
            hdf5_file.attrs["multi_object_json"] = str(os.path.expanduser(args.multi_object_json))
            hdf5_file.attrs["num_object_types"] = int(getattr(env, "_num_object_types", 0))
        hdf5_file.attrs["physics_object_mass"] = float(args.object_mass)
        hdf5_file.attrs["physics_object_scale"] = float(args.object_scale_phys)
        hdf5_file.attrs["gripper_contact_prim_path"] = str(args.gripper_contact_prim_path)
        hdf5_file.attrs["grasp_gripper_threshold"] = float(args.grasp_gripper_threshold)
        hdf5_file.attrs["grasp_contact_threshold"] = float(args.grasp_contact_threshold)
        hdf5_file.attrs["grasp_max_object_dist"] = float(args.grasp_max_object_dist)
        hdf5_file.attrs["grasp_attach_height"] = float(args.grasp_attach_height)
    if args.dynamics_json:
        hdf5_file.attrs["dynamics_json"] = str(os.path.expanduser(args.dynamics_json))
        hdf5_file.attrs["dynamics_scaled_max_lin_vel"] = float(env.cfg.max_lin_vel)
        hdf5_file.attrs["dynamics_scaled_max_ang_vel"] = float(env.cfg.max_ang_vel)
    if args.arm_limit_json:
        hdf5_file.attrs["arm_limit_json"] = str(os.path.expanduser(args.arm_limit_json))
        hdf5_file.attrs["arm_limit_margin_rad"] = float(args.arm_limit_margin_rad)

    try:
        while saved < args.num_demos and attempts < args.max_attempts:
            step_robot_state = extract_robot_state_9d(env)
            step_subtask_ids = None
            if args.annotate_subtasks:
                step_subtask_ids = env.phase.clone()

            # obs/action과 동일 시점(t)의 이미지를 먼저 캡처한다.
            base_rgb = None
            wrist_rgb = None
            if use_camera:
                base_rgb = env.get_base_rgb()
                wrist_rgb = env.get_wrist_rgb()

            with torch.no_grad():
                action = compute_deterministic_action(
                    obs_policy=obs["policy"],
                    policy_model=models["policy"],
                    state_preprocessor=state_preprocessor,
                    agent=agent,
                )

            next_obs, reward, terminated, truncated, info = env.step(action)

            if use_spawn and spawn_mgr is not None:
                spawn_mgr.update_all_positions(env, env.object_pos_w)

            for i in range(args.num_envs):
                ep_obs[i].append(obs["policy"][i].cpu().numpy())
                ep_act[i].append(action[i].cpu().numpy())
                ep_robot_state[i].append(step_robot_state[i].cpu().numpy())
                if args.annotate_subtasks and step_subtask_ids is not None:
                    ep_subtask_ids[i].append(int(step_subtask_ids[i].item()))
                if use_camera:
                    if base_rgb is not None:
                        ep_base_img[i].append(base_rgb[i].cpu().numpy())
                    if wrist_rgb is not None:
                        ep_wrist_img[i].append(wrist_rgb[i].cpu().numpy())

            done = terminated | truncated
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_ids.tolist():
                attempts += 1
                if len(ep_obs[idx]) < 5:
                    _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state, ep_subtask_ids)
                    if use_spawn and spawn_mgr is not None:
                        spawn_mgr.respawn_for_env(idx, env.object_pos_w[idx])
                        ep_object_name[idx] = spawn_mgr.get_object_name(idx)
                        ep_spawn_meta[idx] = spawn_mgr.get_spawn_metadata(idx)
                    continue

                root = env.robot.data.root_pos_w[idx, :2].cpu().numpy()
                goal = env.goal_pos_w[idx, :2].cpu().numpy()
                dist = float(np.linalg.norm(root - goal))

                if hasattr(env, "task_success"):
                    success = bool(env.task_success[idx].item())
                else:
                    success = bool(truncated[idx].item() and dist < 0.3)

                if success and saved < args.num_demos:
                    quality_ok = True
                    quality_reason = "ok"
                    physics_obj_meta = None

                    if use_spawn and spawn_mgr is not None and args.annotate_subtasks:
                        quality_ok, quality_reason = spawn_mgr.check_quality(
                            env_id=idx,
                            subtask_ids=ep_subtask_ids[idx],
                            min_steps=args.min_steps,
                        )

                    if not quality_ok:
                        skipped += 1
                        if skipped <= 10 or skipped % 50 == 0:
                            print(f"  Skip #{skipped} | env={idx} | {quality_reason}")
                        _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state, ep_subtask_ids)
                        if use_spawn and spawn_mgr is not None:
                            spawn_mgr.respawn_for_env(idx, env.object_pos_w[idx])
                            ep_object_name[idx] = spawn_mgr.get_object_name(idx)
                            ep_spawn_meta[idx] = spawn_mgr.get_spawn_metadata(idx)
                        continue

                    grp = hdf5_file.create_group(f"episode_{saved}")
                    grp.create_dataset("obs", data=np.array(ep_obs[idx]))
                    grp.create_dataset("actions", data=np.array(ep_act[idx]))
                    robot_state_np = np.array(ep_robot_state[idx], dtype=np.float32)
                    grp.create_dataset("robot_state", data=robot_state_np)

                    if args.annotate_subtasks:
                        subtask_ids_np = np.array(ep_subtask_ids[idx], dtype=np.int64)
                        grp.create_dataset("subtask_ids", data=subtask_ids_np)

                        if use_spawn:
                            obj_name_for_transition = ep_object_name[idx]
                        elif multi_object_mode:
                            physics_obj_meta = _get_multi_object_episode_meta(env, idx)
                            obj_name_for_transition = str(physics_obj_meta.get("object_name", ""))
                        else:
                            obj_name_for_transition = ""
                        transitions = build_subtask_transitions(
                            ep_subtask_ids[idx],
                            object_name=obj_name_for_transition,
                        )
                        grp.attrs["subtask_transitions"] = json.dumps(transitions, ensure_ascii=False)
                        grp.attrs["num_subtask_transitions"] = len(transitions)
                        if len(subtask_ids_np) > 0:
                            grp.attrs["initial_subtask_id"] = int(subtask_ids_np[0])
                            grp.attrs["final_subtask_id"] = int(subtask_ids_np[-1])

                    if use_camera:
                        img_grp = grp.create_group("images")
                        if len(ep_base_img[idx]) > 0:
                            img_grp.create_dataset(
                                "base_rgb",
                                data=np.array(ep_base_img[idx], dtype=np.uint8),
                                compression="gzip",
                                compression_opts=4,
                                chunks=(1, args.base_cam_height, args.base_cam_width, 3),
                            )
                        if len(ep_wrist_img[idx]) > 0:
                            img_grp.create_dataset(
                                "wrist_rgb",
                                data=np.array(ep_wrist_img[idx], dtype=np.uint8),
                                compression="gzip",
                                compression_opts=4,
                                chunks=(1, args.wrist_cam_height, args.wrist_cam_width, 3),
                            )

                    grp.attrs["num_steps"] = len(ep_obs[idx])
                    grp.attrs["final_object_dist"] = dist
                    # Legacy alias for backward compatibility with old analysis scripts.
                    grp.attrs["final_dist"] = dist
                    grp.attrs["success"] = True
                    grp.attrs["has_images"] = bool(use_camera)
                    grp.attrs["has_subtask_annotation"] = bool(args.annotate_subtasks)
                    grp.attrs["num_base_imgs"] = len(ep_base_img[idx])
                    grp.attrs["num_wrist_imgs"] = len(ep_wrist_img[idx])
                    grp.attrs["quality_check"] = quality_reason

                    if hasattr(env, "phase"):
                        grp.attrs["final_phase"] = int(env.phase[idx].item())
                    if hasattr(env, "object_visible"):
                        grp.attrs["final_object_visible"] = bool(env.object_visible[idx].item())
                    if hasattr(env, "object_grasped"):
                        grp.attrs["final_object_grasped"] = bool(env.object_grasped[idx].item())
                    if hasattr(env, "home_pos_w"):
                        home = env.home_pos_w[idx, :2].cpu().numpy()
                        grp.attrs["final_home_dist"] = float(np.linalg.norm(root - home))
                    if hasattr(env, "object_bbox"):
                        grp.attrs["object_bbox_xyz"] = env.object_bbox[idx].detach().cpu().numpy().astype(np.float32).tolist()
                    if hasattr(env, "object_category_id"):
                        grp.attrs["object_category_id"] = int(env.object_category_id[idx].item())
                    if hasattr(env, "active_object_idx"):
                        grp.attrs["active_object_type_idx"] = int(env.active_object_idx[idx].item())

                    if use_spawn and spawn_mgr is not None:
                        obj_name = ep_object_name[idx]
                        grp.attrs["object_name"] = obj_name
                        grp.attrs["instruction"] = spawn_mgr.get_full_task_instruction(idx)
                        grp.attrs["spawn_meta_json"] = json.dumps(ep_spawn_meta[idx], ensure_ascii=False)
                        grp.attrs["object_usd"] = str(ep_spawn_meta[idx].get("object_usd", ""))
                        grp.attrs["object_scale"] = float(ep_spawn_meta[idx].get("object_scale", args.object_scale))

                        subtask_map = {}
                        for sid in range(4):
                            subtask_map[str(sid)] = spawn_mgr.get_subtask_instruction(idx, sid)
                        subtask_map[str(FULL_TASK_ID)] = spawn_mgr.get_full_task_instruction(idx)
                        grp.attrs["subtask_id_to_text_json"] = json.dumps(subtask_map, ensure_ascii=False)

                        spawn_mgr.record_saved(idx)
                    elif multi_object_mode:
                        if physics_obj_meta is None:
                            physics_obj_meta = _get_multi_object_episode_meta(env, idx)

                        obj_name = str(physics_obj_meta.get("object_name", "target object"))
                        instruction = str(physics_obj_meta.get("instruction", FULL_TASK_TEXT_STATIC))
                        grp.attrs["object_name"] = obj_name
                        grp.attrs["instruction"] = instruction
                        grp.attrs["object_usd"] = str(physics_obj_meta.get("object_usd", ""))
                        grp.attrs["object_scale"] = float(physics_obj_meta.get("object_scale", 1.0))
                        cat_name = str(physics_obj_meta.get("object_category_name", "")).strip()
                        if cat_name:
                            grp.attrs["object_category_name"] = cat_name

                        subtask_map = dict(SUBTASK_ID_TO_TEXT_STATIC)
                        subtask_map[str(FULL_TASK_ID)] = instruction
                        grp.attrs["subtask_id_to_text_json"] = json.dumps(subtask_map, ensure_ascii=False)
                    else:
                        grp.attrs["instruction"] = FULL_TASK_TEXT_STATIC
                        grp.attrs["subtask_id_to_text_json"] = json.dumps(
                            SUBTASK_ID_TO_TEXT_STATIC,
                            ensure_ascii=False,
                        )

                    hdf5_file.flush()

                    saved += 1
                    dists.append(dist)

                    img_info = f" | imgs={len(ep_base_img[idx])}" if use_camera else ""
                    if use_spawn:
                        obj_info = f" | obj={ep_object_name[idx]}"
                    elif multi_object_mode:
                        if physics_obj_meta is None:
                            physics_obj_meta = _get_multi_object_episode_meta(env, idx)
                        obj_info = f" | obj={physics_obj_meta.get('object_name', 'target object')}"
                    else:
                        obj_info = ""
                    print(
                        f"  Demo {saved:>3}/{args.num_demos} | "
                        f"steps={len(ep_obs[idx]):>4} | dist={dist:.3f}m | "
                        f"att={attempts}{img_info}{obj_info}"
                    )

                _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state, ep_subtask_ids)

                if use_spawn and spawn_mgr is not None:
                    if use_dr_lighting:
                        spawn_mgr.randomize_lighting()
                    spawn_mgr.respawn_for_env(idx, env.object_pos_w[idx])
                    ep_object_name[idx] = spawn_mgr.get_object_name(idx)
                    ep_spawn_meta[idx] = spawn_mgr.get_spawn_metadata(idx)

            obs = next_obs

    except KeyboardInterrupt:
        print("\n  중단됨")
    finally:
        hdf5_file.close()

    rate = saved / max(attempts, 1) * 100
    print("\n" + "=" * 60)
    print(f"  수집 완료: {saved}/{args.num_demos}")
    print(f"  시도: {attempts} (성공률: {rate:.1f}%)")
    print(f"  품질 필터 skip: {skipped}")
    if dists:
        print(f"  평균 최종 object 거리: {np.mean(dists):.3f}m")
    if use_camera:
        print(
            f"  카메라: base ({args.base_cam_width}x{args.base_cam_height})"
            f" + wrist ({args.wrist_cam_width}x{args.wrist_cam_height})"
        )
    if use_spawn and spawn_mgr is not None:
        stats = spawn_mgr.get_object_stats()
        print(f"  물체 다양성: {stats['total_objects_used']}/{stats['total_library']}종 사용")
        if stats["top_5"]:
            print(f"  Top 5: {stats['top_5']}")
    print(f"  Annotation: {'ON' if args.annotate_subtasks else 'OFF'}")
    print(f"  파일: {output_path}")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  파일 크기: {size_mb:.1f} MB")
    print("\n  다음 단계: HDF5 -> LeRobot v3 변환 -> VLA 파인튜닝")
    print("=" * 60)

    if use_spawn and spawn_mgr is not None:
        spawn_mgr.despawn_all()
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
