#!/usr/bin/env python3
"""
VLM + VLA 전체 태스크 실행기

아키텍처:
  [A100 서버 — 최초 1회]
    유저 지시어 → VLM /classify → source_object + dest_object 추출

  [실행 루프]
    3090 로컬 (Isaac Sim + 카메라)
      ├─ base_cam + wrist_cam + depth (omni.replicator)
      └─ depth < 0.3m → 긴급정지

    A100 서버
      ├─ VLM (0.3Hz, 비동기): base_cam → 자연어 instruction 생성
      └─ VLA (5-10Hz, 동기): base_cam + wrist_cam + instruction + 9D state → 9D action

    3090 로컬
      └─ env.step(action)

Usage:
    # SSH 터널 먼저
    ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 jovyan@218.148.55.186 -p 30179

    python run_full_task.py \
        --user_command "약병을 찾아서 빨간 컵 옆에 놓아" \
        --object_usd <path> \
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Add own dir first (for vlm_orchestrator etc.), then parent (for lekiwi_skill*_env)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Full Task: VLM + VLA")

# Servers
parser.add_argument("--vlm_server", type=str, default="http://localhost:8000")
parser.add_argument("--vla_server", type=str, default="http://localhost:8002")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

# Task — 유저가 직접 입력하는 지시어
parser.add_argument("--user_command", type=str,
                    default="find the medicine bottle and place it next to the red cup",
                    help="유저 지시어 (VLM이 source/dest 자동 추출)")
# 수동 override (VLM classify 건너뛸 때)
parser.add_argument("--target_object", type=str, default="",
                    help="수동 지정 시 VLM classify 건너뜀")
parser.add_argument("--dest_object", type=str, default="",
                    help="수동 지정 시 VLM classify 건너뜀")

parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--action_log", type=str, default="",
                    help="매 step VLA action 기록 파일 경로 (빈 문자열=비활성)")

# Camera
# VLM용: 큰 해상도 (인식 정확도 ↑)
parser.add_argument("--vlm_width", type=int, default=1280)
parser.add_argument("--vlm_height", type=int, default=800)
# VLA용: 학습 데이터와 일치 (base/wrist 모두 640x400)
parser.add_argument("--vla_width", type=int, default=640)
parser.add_argument("--vla_height", type=int, default=400)
parser.add_argument("--jpeg_quality", type=int, default=80)

# Safety
parser.add_argument("--safety_dist", type=float, default=0.3)
parser.add_argument("--enable_safety", action="store_true", default=True)

# Timing
parser.add_argument("--vlm_interval", type=int, default=30,
                    help="VLM 호출 시도 간격 (steps, 비동기 _pending으로 자동 throttle)")
parser.add_argument("--max_total_steps", type=int, default=6000,
                    help="최대 스텝 (6000 = 10분 at 10Hz)")

# Mode
parser.add_argument("--mode", type=str, default="viva",
                    choices=["viva", "single_vla"],
                    help="viva: VIVA 구조 (VLM+VLA), single_vla: 비교군 ①-B")

# Skill timeouts (VIVA mode)
parser.add_argument("--navigate_timeout", type=int, default=2000)
parser.add_argument("--approach_lift_timeout", type=int, default=1000)
parser.add_argument("--carry_timeout", type=int, default=2000)
parser.add_argument("--approach_place_timeout", type=int, default=1000)

# S2/S4 재시도 제한
parser.add_argument("--s2_max_attempts", type=int, default=3,
                    help="S2 approach & lift 최대 진입 횟수 (첫 진입 포함)")
parser.add_argument("--s4_max_attempts", type=int, default=3,
                    help="S4 approach & place 최대 진입 횟수 (첫 진입 포함)")

# S4 미구현 → carry까지만 평가
parser.add_argument("--stop_at_carry", action="store_true", default=True,
                    help="S4 (approach_and_place) 건너뛰고 carry에서 종료 (success)")
parser.add_argument("--no_stop_at_carry", dest="stop_at_carry", action="store_false")

# Env
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

# Scene (ProcTHOR)
parser.add_argument("--scene_idx", type=int, default=0, help="ProcTHOR scene index (0=no scene)")
parser.add_argument("--scene_usd", type=str, default="", help="Scene USD path (overrides scene_idx)")
parser.add_argument("--scene_scale", type=float, default=1.0, help="Scene scale factor")
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")

# Difficulty
parser.add_argument("--difficulty", type=str, default="easy",
                    choices=["easy", "hard"],
                    help="easy: 같은 방 (로봇+물체), hard: 다른 방 (로봇과 물체 분리)")

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

from vlm_orchestrator import classify_user_request, RelativePlacementOrchestrator, VIVAOrchestrator, SkillState, LIFTED_POSE_RANGE


# ═══════════════════════════════════════════════════════════════════════
#  VLA Client
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

    def query_action(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                     state_9d: list[float], instruction: str) -> list[list[float]]:
        """동기 호출: images + state + instruction → action chunk."""
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
            return result["actions"]
        except Exception as e:
            print(f"  [VLA] error: {e}")
            self._last_latency = time.perf_counter() - t0
            return []

    def get_action_9d(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                      state_9d: list[float], instruction: str) -> np.ndarray:
        """Action chunk에서 1개 반환 (9D). 버퍼 소진 시 새 쿼리."""
        if self._buffer_idx >= len(self._action_buffer):
            self._action_buffer = self.query_action(
                base_rgb, wrist_rgb, state_9d, instruction
            )
            self._buffer_idx = 0
            if not self._action_buffer:
                return np.zeros(9, dtype=np.float32)

        raw = np.array(self._action_buffer[self._buffer_idx], dtype=np.float32)
        self._buffer_idx += 1
        # 32D → 9D truncate (fine-tuned 모델은 9D 직접 반환)
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


# ═══════════════════════════════════════════════════════════════════════
#  Robot Status (VIVA mode)
# ═══════════════════════════════════════════════════════════════════════

def check_lifted_pose(arm_joints: list, grip_pos: float, contact: bool) -> bool:
    """joint가 lifted pose range 내 + contact 감지 시 True.
    S2→S3 전환 판정에 사용. LIFTED_POSE_RANGE는 vlm_orchestrator에서 import."""
    if not contact:
        return False
    joints_with_grip = arm_joints + [grip_pos]
    for val, (low, high) in zip(joints_with_grip, LIFTED_POSE_RANGE.values()):
        if not (low <= val <= high):
            return False
    return True


def get_depth_min(depth_image: np.ndarray) -> float | None:
    """전방 중앙 1/3 영역의 min depth 반환."""
    if depth_image is None:
        return None
    H, W = depth_image.shape[:2]
    center = depth_image[H // 3 : 2 * H // 3, W // 3 : 2 * W // 3]
    valid = (center > 0.10) & (center < 10.0)
    if valid.sum() < 10:
        return None
    return float(center[valid].min())


def get_contact_detected(env) -> bool:
    """gripper contact sensor 확인. jaw OR wrist force > 1.0N."""
    try:
        # jaw contact
        jaw_forces = env.contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
        jaw_mag = jaw_forces[0].norm(dim=-1).max().item()
        # wrist contact
        wrist_mag = 0.0
        if hasattr(env, 'wrist_contact_sensor') and env.wrist_contact_sensor is not None:
            wrist_forces = env.wrist_contact_sensor.data.net_forces_w
            wrist_mag = wrist_forces[0].norm(dim=-1).max().item()
        return (jaw_mag > 1.0) or (wrist_mag > 1.0)
    except Exception:
        return False


def build_robot_status(env, contact: bool, depth_min: float | None) -> str:
    """로봇 상태를 VLM 프롬프트용 텍스트로 변환.
    매 VLM 호출마다 시스템 프롬프트에 포함된다."""
    jp = env.robot.data.joint_pos[0]
    arm_joints = jp[env.arm_idx[:5]].tolist()
    grip_pos = jp[env.gripper_idx].item()

    # gripper 상태
    if contact:
        gripper_str = f"closed, contact detected (object grasped), position={grip_pos:.3f}"
    elif grip_pos < 0.3:
        gripper_str = f"closed, no contact, position={grip_pos:.3f}"
    else:
        gripper_str = f"open, position={grip_pos:.3f}"

    # arm pose 판정
    lifted = check_lifted_pose(arm_joints, grip_pos, contact)
    arm_str = "LIFTED (object held in lifted pose)" if lifted else "NOT_LIFTED"

    # depth warning (S2/S4에서 VLM에게 장애물/목표물 판단을 위임하기 위한 정보)
    if depth_min is not None and depth_min < 0.3:
        depth_str = f"CLOSE_OBJECT_DETECTED (min depth: {depth_min:.2f}m)"
    else:
        depth_str = "NONE"

    return (
        f"Robot status:\n"
        f"- Gripper: {gripper_str}\n"
        f"- Arm pose: {arm_str}, joints={[round(j, 3) for j in arm_joints]}\n"
        f"- Depth warning: {depth_str}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Isaac Sim 환경 + 카메라
# ═══════════════════════════════════════════════════════════════════════

def setup_env(args):
    """Skill2 환경 + omni.replicator 카메라."""
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
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

    # Scene (ProcTHOR)
    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    if scene_path is not None:
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

    env = Skill2Env(cfg=cfg)

    # omni.replicator 카메라 — VLM / VLA / Depth 분리
    # 1) base 큰 RP (VLM용, 1280x800)
    base_vlm_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (args.vlm_width, args.vlm_height),
    )
    # 2) wrist (VLA용, 640x400, 학습 데이터 일치)
    wrist_rgb_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera",
        (args.vla_width, args.vla_height),
    )
    # 3) Depth 전용 RP — safety check용, 매 step 렌더
    DEPTH_W, DEPTH_H = 320, 200
    depth_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (DEPTH_W, DEPTH_H),
    )

    base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    base_rgb_annot.attach([base_vlm_rp])
    wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_rgb_annot.attach([wrist_rgb_rp])
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([depth_rp])

    # RP enable/disable 토글 시도
    def _set_rp_enabled(rp, enabled: bool) -> bool:
        """render_product의 update를 켜고/끄기. 가능한 API들을 시도."""
        try:
            rp.hydra_texture.set_updates_enabled(enabled)
            return True
        except Exception:
            pass
        try:
            rp.set_paused(not enabled)
            return True
        except Exception:
            pass
        # 마지막 fallback: schedule_update 토글
        try:
            if enabled:
                rp.schedule_update()
            return True
        except Exception:
            pass
        return False

    # 시작 시 큰 RP 비활성화 시도
    # GUI 모드에서는 toggle이 viewport texture에 영향 → 화면 일그러짐 발생
    # 따라서 headless일 때만 toggle 사용
    is_headless = bool(getattr(args, "headless", False))
    if is_headless:
        big_toggle_works = _set_rp_enabled(base_vlm_rp, False) and _set_rp_enabled(wrist_rgb_rp, False)
        # 다시 enable (capture 시 즉시 사용 가능하도록)
        _set_rp_enabled(base_vlm_rp, True)
        _set_rp_enabled(wrist_rgb_rp, True)
    else:
        big_toggle_works = False  # GUI 모드: toggle 사용 안 함 (viewport 보호)
    print(f"  [Render] Big RP toggle: {'OK' if big_toggle_works else 'DISABLED (GUI mode or unsupported)'}")
    print(f"  [Render] VLM={args.vlm_width}x{args.vlm_height} VLA={args.vla_width}x{args.vla_height} Depth={DEPTH_W}x{DEPTH_H}")

    cams = {
        "base_rgb": base_rgb_annot,        # 1280x800 VLM용
        "depth": depth_annot,
        "wrist_rgb": wrist_rgb_annot,      # 640x400 VLA용
        "_base_vlm_rp": base_vlm_rp,
        "_wrist_rp": wrist_rgb_rp,
        "_depth_rp": depth_rp,
        "_big_toggle_works": big_toggle_works,
        "_set_rp_enabled": _set_rp_enabled,
    }
    return env, cams, scene_path


def capture_depth_only(env, cams: dict):
    """매 step 호출. 작은 depth만 렌더 → 빠름."""
    set_en = cams["_set_rp_enabled"]
    if cams["_big_toggle_works"]:
        set_en(cams["_base_vlm_rp"], False)
        set_en(cams["_wrist_rp"], False)
    env.sim.render()
    d = cams["depth"].get_data()
    depth = np.array(d) if d is not None else None
    return depth


def capture_full(env, cams: dict):
    """VLM/VLA 호출 시점에만. RGB + depth 모두 렌더.
    base_rgb는 VLM 해상도 (1280x800), wrist_rgb는 VLA 해상도 (640x400)."""
    set_en = cams["_set_rp_enabled"]
    if cams["_big_toggle_works"]:
        set_en(cams["_base_vlm_rp"], True)
        set_en(cams["_wrist_rp"], True)
    env.sim.render()
    b = cams["base_rgb"].get_data()
    d = cams["depth"].get_data()
    w = cams["wrist_rgb"].get_data()
    base_rgb = np.array(b)[..., :3] if b is not None else None
    depth = np.array(d) if d is not None else None
    wrist_rgb = np.array(w)[..., :3] if w is not None else None
    return base_rgb, depth, wrist_rgb


def downscale_for_vla(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """VLA 입력용 RGB downscale (PIL Lanczos). VLM용 큰 RGB → VLA용 작은 RGB."""
    if rgb is None:
        return None
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def capture(env, cams: dict):
    """레거시 호환 — capture_full 호출."""
    return capture_full(env, cams)


def get_state_9d(env) -> list[float]:
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].tolist()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].tolist()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return arm + [grip] + bv[:2] + [wz]


# ═══════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")

    # ── 0. 서버 헬스 체크 ──
    print(f"\n{'='*60}")
    print(f"  Full Task: VLM + VLA")
    print(f"  VLM: {args.vlm_server} ({args.vlm_model})")
    print(f"  VLA: {args.vla_server}")
    print(f"{'='*60}")

    print(f"\n  [Health Check]")
    try:
        r = requests.get(f"{args.vlm_server}/v1/models", timeout=3)
        print(f"  VLM: OK ({r.json()['data'][0]['id']})")
    except Exception as e:
        print(f"  VLM: FAIL — {e}")
        print(f"  → SSH 터널: ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 ...")
        simulation_app.close()
        return

    vla = VLAClient(args.vla_server, args.jpeg_quality)
    vla_health = vla.health_check()
    if vla_health:
        print(f"  VLA: OK ({vla_health['model']}, {vla_health['gpu_memory_mb']:.0f}MB)")
    else:
        print(f"  VLA: FAIL")
        simulation_app.close()
        return

    # ── 1. VLM /classify: 유저 지시어 → source/dest 추출 (1회) ──
    if args.target_object and args.dest_object:
        source = args.target_object
        dest = args.dest_object
        print(f"\n  [Classify] Manual override: {source} → {dest}")
    else:
        print(f"\n  [Classify] User command: \"{args.user_command}\"")
        task_info = classify_user_request(
            args.vlm_server, args.vlm_model, args.user_command
        )
        source = task_info["source_object"]
        dest = task_info.get("dest_object", "")
        print(f"  [Classify] Result: mode={task_info['mode']}, "
              f"source=\"{source}\", dest=\"{dest}\"")

    user_request = args.user_command if args.user_command else f"find the {source} and place it next to the {dest}"

    # ── 2. 환경 + 카메라 ──
    print(f"\n  Setting up environment...")
    env, cams, scene_path = setup_env(args)
    obs, _ = env.reset()

    # warm up
    for _ in range(5):
        env.sim.step()
        env.sim.render()

    # ── 3. Orchestrator 생성 ──
    if args.mode == "viva":
        orch = VIVAOrchestrator(
            vlm_server=args.vlm_server,
            vlm_model=args.vlm_model,
            source_object=source,
            dest_object=dest,
            user_request=user_request,
            jpeg_quality=args.jpeg_quality,
            navigate_timeout=args.navigate_timeout,
            approach_lift_timeout=args.approach_lift_timeout,
            carry_timeout=args.carry_timeout,
            approach_place_timeout=args.approach_place_timeout,
            stop_at_carry=args.stop_at_carry,
            s2_max_attempts=args.s2_max_attempts,
            s4_max_attempts=args.s4_max_attempts,
        )
    elif args.mode == "single_vla":
        orch = RelativePlacementOrchestrator(
            vlm_server=args.vlm_server,
            vlm_model=args.vlm_model,
            source_object=source,
            dest_object=dest,
            user_request=user_request,
            jpeg_quality=args.jpeg_quality,
        )

    prev_skill = orch.current_skill if args.mode == "viva" else None

    # Ground-truth 결과 집계
    trial_results = []  # list of dict: {s1, s2, s3, full, steps, time}

    # Action log file
    action_log_f = None
    if args.action_log:
        action_log_f = open(args.action_log, "w")
        action_log_f.write("trial\tstep\tskill\tinstruction\tarm0\tarm1\tarm2\tarm3\tarm4\tgrip\tvx\tvy\twz\tact_vx\tact_vy\tact_wz\n")
        action_log_f.flush()
        print(f"  [ActionLog] Writing to {args.action_log}")

    print(f"\n{'='*60}")
    print(f"  Task: \"{user_request}\"")
    print(f"  Mode: {args.mode}")
    print(f"  Source: {source}, Dest: {dest}")
    print(f"  Camera VLM={args.vlm_width}x{args.vlm_height} VLA={args.vla_width}x{args.vla_height}")
    print(f"  Safety: {args.safety_dist}m, VLM interval: {args.vlm_interval} steps")
    print(f"  Max steps: {args.max_total_steps}")
    if args.mode == "viva":
        print(f"  Timeouts: nav={args.navigate_timeout}, lift={args.approach_lift_timeout}, "
              f"carry={args.carry_timeout}, place={args.approach_place_timeout}")
        print(f"  Max attempts: S2={args.s2_max_attempts}, S4={args.s4_max_attempts}")
    if scene_path is not None:
        print(f"  Difficulty: {args.difficulty} ({'같은 방' if args.difficulty == 'easy' else '다른 방'})")
    print(f"{'='*60}\n")

    # ── 4. 메인 루프 ──
    for trial in range(args.num_trials):
        print(f"  === Trial {trial + 1}/{args.num_trials} ({args.difficulty}) ===")
        obs, _ = env.reset()

        # Trial 시작 시 orchestrator 상태 리셋 (multi-trial에서 이전 trial 상태 잔류 방지)
        if args.mode == "viva":
            orch.reset_for_new_trial()
        prev_skill = orch.current_skill if args.mode == "viva" else None

        # ── Difficulty-aware 스폰 ──
        if scene_path is not None:
            from procthor_scene import (
                sample_scene_task_layout, apply_scene_task_layout,
                SceneSpawnCfg, _load_floor_regions, _load_support_floor_z,
                _find_robot_region, SCENE_PRESETS,
            )
            import math as _m
            preset = SCENE_PRESETS.get(args.scene_idx)
            sfz = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path) if preset else 0.0
            regions = _load_floor_regions(str(scene_path.resolve()), support_floor_z=sfz)
            ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
            src_ov = SceneSpawnCfg(
                min_robot_dist=float(getattr(env.cfg, "object_dist_min", 0.8)) / ss,
                max_robot_dist=float(getattr(env.cfg, "object_dist_max", 1.2)) / ss,
                clearance_radius=0.14,
            )
            layout = None
            r_room = s_room = d_room = None
            r_eq_s = r_eq_d = False
            for _spawn_try in range(200):
                try:
                    layout = sample_scene_task_layout(
                        args.scene_idx, scene_usd=scene_path,
                        scene_scale=args.scene_scale,
                        source_spawn_override=src_ov,
                        robot_faces_source=True,
                        randomize_robot_xy=True,
                    )
                except RuntimeError:
                    continue
                # Room 체크 (robot / source / dest)
                r_xy = (layout.robot_xy[0] / ss, layout.robot_xy[1] / ss)
                s_xy = (layout.source_xy[0] / ss, layout.source_xy[1] / ss)
                d_xy = (layout.dest_xy[0] / ss, layout.dest_xy[1] / ss)
                r_room = _find_robot_region(r_xy, regions)
                s_room = _find_robot_region(s_xy, regions)
                d_room = _find_robot_region(d_xy, regions)
                if r_room is None or s_room is None or d_room is None:
                    continue
                r_eq_s = (r_room.path == s_room.path)
                r_eq_d = (r_room.path == d_room.path)
                if args.difficulty == "easy" and r_eq_s and r_eq_d:
                    break  # 로봇 = source 방 = dest 방
                elif args.difficulty == "hard" and (not r_eq_s) and (not r_eq_d):
                    break  # source ≠ 로봇 방, dest ≠ 로봇 방
            else:
                print(f"  [WARN] {args.difficulty} 조건 만족 실패 200회 → 마지막 layout 사용")
            if layout is None:
                print(f"  [ERROR] Spawn completely failed (200 RuntimeErrors) → skip trial")
                continue
            apply_scene_task_layout(env, layout)
            # settle
            _zero_v = torch.zeros(1, env.robot.num_joints, device=env.device)
            env.robot.set_joint_velocity_target(_zero_v)
            for _ in range(60):
                env.sim.step()
                env.sim.render()
            r_room_str = r_room.path if r_room else "?"
            s_room_str = s_room.path if s_room else "?"
            d_room_str = d_room.path if d_room else "?"
            print(f"  [Spawn] robot={r_room_str} | source={s_room_str} | dest={d_room_str} | "
                  f"r==s:{r_eq_s} r==d:{r_eq_d} | "
                  f"src_dist={_m.dist(layout.robot_xy, layout.source_xy):.2f}m")

        vla.reset_buffer()
        total_steps = 0
        safety_stops = 0
        t_start = time.time()

        # ── Ground-truth 추적 변수 ──
        gt_s1_success = False  # NAVIGATE → APPROACH_AND_LIFT 전환 발생
        gt_s2_success = False  # 언젠가 lift_counter ≥ s2_lift_hold 도달
        gt_s3_success = False  # trial 종료 시점에 물체 들고있음
        gt_s2_lift_counter = 0
        gt_s2_lift_hold = 200  # eval_s3.py 동일
        s3_drop_detected = False
        seen_skill_s2 = False
        seen_skill_s3 = False
        # Stuck 감지 (navigate / carry)
        stuck_counter = 0
        STUCK_GRACE = 40  # 40 step 연속 stuck → fail
        s1_stuck_fail = False
        s3_stuck_fail = False

        # 캐시: 큰 RP는 가끔 렌더, 그 사이엔 캐시 사용
        cached_base_rgb = None
        cached_wrist_rgb = None

        # Timing 누적
        _t_capture_full_sum = 0.0; _t_capture_full_n = 0
        _t_capture_depth_sum = 0.0; _t_capture_depth_n = 0
        _t_envstep_sum = 0.0
        _t_vla_sum = 0.0; _t_vla_n = 0
        _t_misc_sum = 0.0

        try:
            while total_steps < args.max_total_steps and simulation_app.is_running():
                _t_step_start = time.perf_counter()

                # (a) 카메라 캡처 — depth는 매 step (작은 RP), RGB는 필요 시만 (큰 RP)
                #     큰 RGB 필요 조건: VLA buffer 비었거나, VLM 호출 예정
                vla_buffer_empty = (vla._buffer_idx >= len(vla._action_buffer))
                vlm_will_call = False
                if args.mode == "viva":
                    if orch.current_skill in (SkillState.NAVIGATE, SkillState.CARRY):
                        vlm_will_call = (total_steps % args.vlm_interval == 0)
                    # S2/S4의 obstacle check VLM 호출은 depth 본 후 결정 → 일단 우선순위 낮음
                else:
                    vlm_will_call = (total_steps % args.vlm_interval == 0)

                need_full_render = vla_buffer_empty or vlm_will_call or (cached_base_rgb is None)

                _t_cap0 = time.perf_counter()
                if need_full_render:
                    base_rgb, depth, wrist_rgb = capture_full(env, cams)
                    cached_base_rgb = base_rgb
                    cached_wrist_rgb = wrist_rgb
                    _t_capture_full_sum += time.perf_counter() - _t_cap0
                    _t_capture_full_n += 1
                else:
                    depth = capture_depth_only(env, cams)
                    base_rgb = cached_base_rgb
                    wrist_rgb = cached_wrist_rgb
                    _t_capture_depth_sum += time.perf_counter() - _t_cap0
                    _t_capture_depth_n += 1

                if base_rgb is None or wrist_rgb is None:
                    env.sim.step()
                    total_steps += 1
                    continue

                # (b) depth min 계산 (safety layer + VLM depth_warning 공용)
                depth_min = get_depth_min(depth)

                # (c) 로봇 상태 수집 + orchestrator에 전달 (VIVA 모드만)
                if args.mode == "viva":
                    contact = get_contact_detected(env)
                    robot_status = build_robot_status(env, contact, depth_min)
                    orch.update_robot_status(robot_status)
                    orch.update_contact(contact)
                    orch.update_depth_status(depth_min)  # CONTINUE 후 새 장애물 감지
                    orch.tick()  # timeout 체크

                    # ── 코드 기반 전환 판별 (S2/S4) ──
                    jp = env.robot.data.joint_pos[0]
                    arm_joints = jp[env.arm_idx[:5]].tolist()
                    grip_pos = jp[env.gripper_idx].item()

                    orch.check_lifted_complete(arm_joints, grip_pos, contact)
                    orch.check_place_complete(grip_pos, contact)

                    # ── Ground-truth tracking ──
                    # 약병이 없으면 (object_usd 미지정) objZ 계산 스킵
                    if hasattr(env, "object_pos_w") and env.object_pos_w is not None:
                        try:
                            objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                        except Exception:
                            objZ = 0.0
                    else:
                        objZ = 0.0
                    grip_closed = grip_pos < float(env.cfg.grasp_gripper_threshold)
                    held = grip_closed and contact and (objZ > 0.05)

                    # S1 success: navigate → S2 전환 (한 번이라도)
                    if orch.current_skill == SkillState.APPROACH_AND_LIFT and not seen_skill_s2:
                        gt_s1_success = True
                        seen_skill_s2 = True
                    if orch.current_skill == SkillState.CARRY and not seen_skill_s3:
                        seen_skill_s3 = True
                        # S2 → S3 전환했으면 S1도 성공 (S1 건너뛰는 경우 대비)
                        gt_s1_success = True

                    # S2 success: 200 step 동안 lift 유지 (eval_s3.py 동일)
                    if orch.current_skill == SkillState.APPROACH_AND_LIFT:
                        if held:
                            gt_s2_lift_counter += 1
                            if gt_s2_lift_counter >= gt_s2_lift_hold:
                                gt_s2_success = True
                        else:
                            gt_s2_lift_counter = 0

                    # S3 drop 감지 (carry 중 objZ < 0.04)
                    if orch.current_skill == SkillState.CARRY:
                        # S2 통과한 상태에서 carry로 왔으면 S2 성공으로 인정
                        if not gt_s2_success:
                            gt_s2_success = True
                        if objZ < 0.04:
                            s3_drop_detected = True
                            print(f"  [GT] S3 drop detected at step {total_steps} (objZ={objZ:.3f})")
                            break

                # (d) VLM 호출 — 스킬별 분기
                #     VLM 호출 시점에 cached RGB가 stale이면 fresh 캡처
                def _ensure_fresh_rgb():
                    nonlocal base_rgb, wrist_rgb, cached_base_rgb, cached_wrist_rgb
                    if not need_full_render:
                        b, _, w = capture_full(env, cams)
                        if b is not None and w is not None:
                            base_rgb = b
                            wrist_rgb = w
                            cached_base_rgb = b
                            cached_wrist_rgb = w

                if args.mode == "viva":
                    if orch.current_skill in (SkillState.NAVIGATE, SkillState.CARRY):
                        # S1/S3: 매 vlm_interval 스텝마다 VLM 호출 (방향 지시)
                        # depth warning 시에는 즉시 호출 (forward 차단 상태에서 빠르게 새 방향 획득)
                        depth_urgent = (depth_min is not None and depth_min < args.safety_dist)
                        if total_steps % args.vlm_interval == 0 or depth_urgent:
                            _ensure_fresh_rgb()
                            orch.query_async(base_rgb)
                    elif orch.current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
                        # S2/S4: depth warning 시에만 VLM 호출 (장애물 판별)
                        # obstacle_cleared=True이면 재호출 억제 (CONTINUE 이미 받음)
                        if depth_min is not None and depth_min < args.safety_dist and not orch.obstacle_cleared:
                            _ensure_fresh_rgb()
                            orch.query_obstacle_check_async(base_rgb)
                        # 그 외에는 VLM 호출 안 함 → VLA가 고정 instruction으로 자율 수행
                else:
                    if total_steps % args.vlm_interval == 0:
                        _ensure_fresh_rgb()
                        orch.query_async(base_rgb)

                # (e) 종료 체크
                if orch.is_done:
                    print(f"\n  [DONE] Task complete at step {total_steps}")
                    break
                if args.mode == "viva" and orch.is_timed_out:
                    print(f"\n  [TIMEOUT] Skill timed out at step {total_steps}")
                    break

                # (f) VLA action — base_rgb는 VLM 해상도(1280x800)이므로 VLA용 640x400으로 downscale
                instruction = orch.instruction
                state = get_state_9d(env)
                base_rgb_vla = downscale_for_vla(base_rgb, args.vla_width, args.vla_height)
                _t_vla0 = time.perf_counter()
                action = vla.get_action_9d(base_rgb_vla, wrist_rgb, state, instruction)
                _t_vla_sum += time.perf_counter() - _t_vla0
                _t_vla_n += 1

                # (g) Safety layer — 스킬별 분기
                #     S1/S3: depth < safety_dist → 전진(vy > 0)만 차단
                #            (backward, strafe, rotation은 허용 → 탈출 가능)
                #     S2/S4: 비활성화 (목표물 접근 필요)
                #            BUT: VLM obstacle check pending 중이면 전진 감속 (충돌 방지)
                stopped = False
                if args.enable_safety and depth_min is not None and depth_min < args.safety_dist:
                    if args.mode == "viva":
                        if orch.safety_layer_active and float(action[7]) > 0:
                            action = action.copy()
                            action[7] = 0.0  # 전진(vy > 0)만 차단
                            stopped = True
                            safety_stops += 1
                        # S2/S4: VLM 응답 대기 중이고 obstacle 의심 시 전진 감속
                        elif orch.current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
                            if orch.is_pending and not orch.obstacle_cleared and float(action[7]) > 0:
                                action = action.copy()
                                if depth_min < 0.2:
                                    action[7] = 0.0   # 매우 가까움 → 완전 정지
                                else:
                                    action[7] *= 0.1   # 0.2~0.3m → 강하게 감속
                                stopped = True
                                safety_stops += 1
                    elif args.mode == "single_vla":
                        if float(action[7]) > 0:
                            action = action.copy()
                            action[7] = 0.0
                            stopped = True
                            safety_stops += 1

                # (h) 스킬 전환 감지 → VLA buffer 리셋
                #     중요: 스킬이 바뀌면 이전 스킬의 action chunk가 남아있을 수 있다.
                #     특히 S2→S3 전환 시 S2의 arm action이 남아있으면 물체를 놓아버린다.
                if args.mode == "viva" and prev_skill != orch.current_skill:
                    vla.reset_buffer()
                    prev_skill = orch.current_skill

                # (i) env step
                action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                _t_env0 = time.perf_counter()
                obs, rew, term, trunc, info = env.step(action_t)
                _t_envstep_sum += time.perf_counter() - _t_env0
                total_steps += 1
                _t_misc_sum += time.perf_counter() - _t_step_start

                # Action 로그 파일 기록 (매 step)
                if action_log_f is not None:
                    a = action
                    act_vx = env.robot.data.root_lin_vel_b[0, 0].item()
                    act_vy = env.robot.data.root_lin_vel_b[0, 1].item()
                    act_wz_real = env.robot.data.root_ang_vel_b[0, 2].item()
                    sk = orch.current_skill.value if args.mode == "viva" else "single"
                    inst_escaped = instruction.replace("\t", " ").replace("\n", " ")
                    action_log_f.write(
                        f"{trial+1}\t{total_steps}\t{sk}\t{inst_escaped}\t"
                        f"{a[0]:+.4f}\t{a[1]:+.4f}\t{a[2]:+.4f}\t{a[3]:+.4f}\t{a[4]:+.4f}\t"
                        f"{a[5]:+.4f}\t{a[6]:+.4f}\t{a[7]:+.4f}\t{a[8]:+.4f}\t"
                        f"{act_vx:+.4f}\t{act_vy:+.4f}\t{act_wz_real:+.4f}\n"
                    )
                    if total_steps % 20 == 0:
                        action_log_f.flush()

                # (i') Stuck 감지 — navigate / carry 중에만, warmup 30 step 후
                if args.mode == "viva" and total_steps > 30 and \
                   orch.current_skill in (SkillState.NAVIGATE, SkillState.CARRY):
                    cmd_speed = abs(float(action[6])) + abs(float(action[7])) + abs(float(action[8]))
                    actual_lin = env.robot.data.root_lin_vel_b[0, :2]
                    actual_wz = env.robot.data.root_ang_vel_b[0, 2].item()
                    actual_speed = float(actual_lin.norm().item()) + abs(actual_wz)
                    if cmd_speed > 0.05 and actual_speed < 0.02:
                        stuck_counter += 1
                        if stuck_counter >= STUCK_GRACE:
                            if orch.current_skill == SkillState.NAVIGATE:
                                if not seen_skill_s2:
                                    s1_stuck_fail = True
                                    print(f"  [STUCK] Navigate stuck {STUCK_GRACE} steps → S1 fail at {total_steps}")
                                else:
                                    # S2 도달 후 obstacle 회피 NAVIGATE에서 stuck → S1 성공은 유지
                                    print(f"  [STUCK] Recovery navigate stuck {STUCK_GRACE} steps at {total_steps}")
                            else:
                                s3_stuck_fail = True
                                print(f"  [STUCK] Carry stuck {STUCK_GRACE} steps → S3 fail at {total_steps}")
                            break
                    else:
                        stuck_counter = 0
                else:
                    stuck_counter = 0

                # Timing breakdown (50 step마다)
                if total_steps % 50 == 0:
                    avg_full = (_t_capture_full_sum / _t_capture_full_n * 1000) if _t_capture_full_n else 0
                    avg_depth = (_t_capture_depth_sum / _t_capture_depth_n * 1000) if _t_capture_depth_n else 0
                    avg_envstep = (_t_envstep_sum / 50 * 1000)
                    avg_vla = (_t_vla_sum / _t_vla_n * 1000) if _t_vla_n else 0
                    avg_total = (_t_misc_sum / 50 * 1000)
                    print(f"  [TIMING] step_total={avg_total:.0f}ms | "
                          f"capture_full={avg_full:.0f}ms({_t_capture_full_n}x) "
                          f"capture_depth={avg_depth:.0f}ms({_t_capture_depth_n}x) "
                          f"env.step={avg_envstep:.0f}ms "
                          f"vla_call={avg_vla:.0f}ms({_t_vla_n}x)")
                    _t_capture_full_sum = 0; _t_capture_full_n = 0
                    _t_capture_depth_sum = 0; _t_capture_depth_n = 0
                    _t_envstep_sum = 0
                    _t_vla_sum = 0; _t_vla_n = 0
                    _t_misc_sum = 0

                # (j) 로그 (10 step마다)
                if total_steps % 10 == 0:
                    elapsed = time.time() - t_start
                    hz = total_steps / elapsed if elapsed > 0 else 0
                    skill_str = orch.current_skill.value if args.mode == "viva" else "single"
                    stop_str = " [SAFETY]" if stopped else ""
                    # 9D action: [arm5, grip1, base_vx, base_vy, base_wz]
                    a = action
                    arm_str = f"[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f},{a[4]:+.2f}]"
                    base_str = f"[vx={a[6]:+.3f},vy={a[7]:+.3f},wz={a[8]:+.3f}]"
                    # 실제 base velocity
                    act_vx = env.robot.data.root_lin_vel_b[0, 0].item()
                    act_vy = env.robot.data.root_lin_vel_b[0, 1].item()
                    act_wz = env.robot.data.root_ang_vel_b[0, 2].item()
                    actual_str = f"act=[vx={act_vx:+.3f},vy={act_vy:+.3f},wz={act_wz:+.3f}]"
                    print(f"    [t={total_steps:4d} {elapsed:.0f}s {hz:.1f}Hz] "
                          f"skill={skill_str} "
                          f"inst=\"{instruction[:40]}\" "
                          f"vlm={orch.avg_latency*1000:.0f}ms({orch.call_count})"
                          f"{stop_str}")
                    print(f"         arm={arm_str} grip={a[5]:+.3f} base={base_str} {actual_str}")

                # (k) 에피소드 종료 (환경에서 terminate/truncate)
                if term.any() or trunc.any():
                    success = info.get("task_success", torch.zeros(1)).any().item()
                    print(f"\n  Trial {trial+1} → {'SUCCESS' if success else 'TIMEOUT'} "
                          f"| steps={total_steps} safety={safety_stops}")
                    break

        except KeyboardInterrupt:
            print("\n  중단 (Ctrl+C)")
            break

        # Trial 요약 + ground-truth 결과
        elapsed = time.time() - t_start
        skill_str = orch.current_skill.value if args.mode == "viva" else "single"

        # Stuck fail 반영
        if s1_stuck_fail:
            gt_s1_success = False
        if s3_stuck_fail:
            gt_s3_success = False

        # S3 success: drop 없이 carry 단계 끝까지 진행 (또는 carry 도달 후 done)
        if args.mode == "viva":
            # carry까지 진입했고 drop/stuck 안 났으면 S3 성공
            if seen_skill_s3 and not s3_drop_detected and not s3_stuck_fail:
                # 추가 조건: trial 종료 시점에 여전히 물체 들고있어야 함
                try:
                    final_objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                    final_jp = env.robot.data.joint_pos[0]
                    final_grip = final_jp[env.gripper_idx].item()
                    final_contact = get_contact_detected(env)
                    final_held = (final_grip < float(env.cfg.grasp_gripper_threshold)) and final_contact and (final_objZ > 0.04)
                    gt_s3_success = final_held
                except Exception:
                    gt_s3_success = False
            gt_full = gt_s1_success and gt_s2_success and gt_s3_success
        else:
            gt_full = False

        trial_results.append({
            "trial": trial + 1,
            "s1_attempted": True,
            "s1": gt_s1_success,
            "s2_attempted": seen_skill_s2,
            "s2": gt_s2_success,
            "s3_attempted": seen_skill_s3,
            "s3": gt_s3_success,
            "full": gt_full,
            "steps": total_steps,
            "time": elapsed,
            "drop": s3_drop_detected,
            "s1_stuck": s1_stuck_fail,
            "s3_stuck": s3_stuck_fail,
            "final_skill": skill_str,
            "timed_out": orch.is_timed_out if args.mode == "viva" else False,
            "s2_attempts": orch.s2_attempt_count if args.mode == "viva" else 0,
        })

        print(f"  Trial {trial+1} summary: {total_steps} steps, {elapsed:.0f}s, "
              f"VLM {orch.call_count} calls (avg {orch.avg_latency*1000:.0f}ms), "
              f"safety {safety_stops}, final_skill={skill_str}")
        print(f"  Trial {trial+1} GT: S1={gt_s1_success} S2={gt_s2_success} S3={gt_s3_success} "
              f"FULL={gt_full}{' (drop)' if s3_drop_detected else ''}")

    # ═════════ N trials 집계 ═════════
    if trial_results:
        n = len(trial_results)
        # Attempted (해당 skill까지 도달한 trial 수)
        a_s1 = sum(r["s1_attempted"] for r in trial_results)
        a_s2 = sum(r["s2_attempted"] for r in trial_results)
        a_s3 = sum(r["s3_attempted"] for r in trial_results)
        # Success
        n_s1 = sum(r["s1"] for r in trial_results)
        n_s2 = sum(r["s2"] for r in trial_results)
        n_s3 = sum(r["s3"] for r in trial_results)
        n_full = sum(r["full"] for r in trial_results)
        avg_steps = sum(r["steps"] for r in trial_results) / n
        avg_time = sum(r["time"] for r in trial_results) / n

        def pct(x, y):
            return f"{x/y*100:.0f}%" if y > 0 else "n/a"

        print(f"\n{'='*70}")
        print(f"  Eval Summary: {args.difficulty} mode, {n} trials, {args.mode}")
        print(f"{'='*70}")
        print(f"  Funnel:")
        print(f"    Total trials:        {n}")
        print(f"    ↓ S1 attempted:      {a_s1}")
        print(f"    ↓ S2 reached:        {a_s2}  (S1 success rate: {n_s1}/{a_s1} = {pct(n_s1, a_s1)})")
        print(f"    ↓ S3 reached:        {a_s3}  (S2 success rate: {n_s2}/{a_s2} = {pct(n_s2, a_s2)} cond)")
        print(f"    Full task:           {n_full}  (S3 success rate: {n_s3}/{a_s3} = {pct(n_s3, a_s3)} cond)")
        print(f"  ---")
        print(f"  Skill success (conditional / absolute):")
        print(f"    S1 Navigate:         {n_s1}/{a_s1} ({pct(n_s1, a_s1)}) cond  |  {n_s1}/{n} ({pct(n_s1, n)}) abs")
        print(f"    S2 Approach&Lift:    {n_s2}/{a_s2} ({pct(n_s2, a_s2)}) cond  |  {n_s2}/{n} ({pct(n_s2, n)}) abs")
        print(f"    S3 Carry:            {n_s3}/{a_s3} ({pct(n_s3, a_s3)}) cond  |  {n_s3}/{n} ({pct(n_s3, n)}) abs")
        print(f"    Full task:           {n_full}/{n} ({pct(n_full, n)})")
        print(f"  ---")
        print(f"  Avg steps per trial:   {avg_steps:.0f}")
        print(f"  Avg time per trial:    {avg_time:.0f}s")
        print(f"  ---")
        print(f"  Per trial:")
        for r in trial_results:
            tag = "✓" if r["full"] else "✗"
            extras = []
            if r["drop"]: extras.append("drop")
            if r.get("s1_stuck"): extras.append("S1_stuck")
            if r.get("s3_stuck"): extras.append("S3_stuck")
            if r.get("timed_out"): extras.append("timeout")
            if r.get("s2_attempts", 0) > 1: extras.append(f"S2×{r['s2_attempts']}")
            extras_str = " " + ",".join(extras) if extras else ""
            s1 = "✓" if r["s1"] else ("·" if not r["s1_attempted"] else "✗")
            s2 = "✓" if r["s2"] else ("·" if not r["s2_attempted"] else "✗")
            s3 = "✓" if r["s3"] else ("·" if not r["s3_attempted"] else "✗")
            print(f"    {tag} ep{r['trial']:2d}: S1={s1} S2={s2} S3={s3} "
                  f"steps={r['steps']:4d} {r['time']:.0f}s "
                  f"end={r.get('final_skill', '?')}{extras_str}")
        print(f"{'='*70}")

    if action_log_f is not None:
        action_log_f.close()
        print(f"  [ActionLog] Saved: {args.action_log}")

    simulation_app.close()


if __name__ == "__main__":
    main()
