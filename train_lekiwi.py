#!/usr/bin/env python3
"""
LeKiwi Navigation — PPO Training (skrl 1.4.3) + BC warm-start.

BC → RL 파이프라인:
    1. record_teleop.py로 텔레옵 데모 수집 (ROS2: 리더암 + 키보드)
    2. train_bc.py로 BC 학습 → checkpoints/bc_nav.pt
    3. 이 스크립트로 BC 가중치를 PPO Actor에 로드 → RL Fine-tune

Usage:
    cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env
    conda activate env_isaaclab && source ~/isaacsim/setup_conda_env.sh

    # ── BC → RL Fine-tune (37D 메인 실험) ──
    python train_lekiwi.py --num_envs 2048 \
        --bc_checkpoint checkpoints/bc_nav.pt \
        --multi_object_json object_catalog.json \
        --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
        --dynamics_json calibration/tuned_dynamics.json --headless

    # ── RL from scratch (37D baseline) ──
    python train_lekiwi.py --num_envs 2048 \
        --multi_object_json object_catalog.json \
        --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
        --dynamics_json calibration/tuned_dynamics.json --headless

    # ── 소규모 테스트 ──
    python train_lekiwi.py --num_envs 64 --max_iterations 100 \
        --bc_checkpoint checkpoints/bc_nav.pt \
        --multi_object_json object_catalog.json \
        --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>"

    # ── 학습 재개 ──
    python train_lekiwi.py --num_envs 2048 --resume --checkpoint logs/ppo_lekiwi/best_agent.pt --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# —— AppLauncher (반드시 다른 import 전에) ——————————————————————
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeKiwi Nav PPO Training")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--resume", action="store_true",
                    help="이전 PPO 체크포인트에서 학습 재개")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="PPO 체크포인트 경로 (--resume와 함께 사용)")
parser.add_argument("--bc_checkpoint", type=str, default=None,
                    help="BC 가중치 경로 (train_bc.py 출력물)")
parser.add_argument("--bc_lr_scale", type=float, default=0.3,
                    help="BC warm-start 시 learning rate 스케일 (기본 0.3 → 3e-4 × 0.3 = 9e-5)")
parser.add_argument("--allow_bc_norm_mismatch", action="store_true",
                    help="BC obs 정규화 파일이 있어도 강제 진행 (권장하지 않음)")
parser.add_argument("--dynamics_json", type=str, default=None,
                    help="tune_sim_dynamics.py 출력 JSON (best_params) 경로")
parser.add_argument("--calibration_json", type=str, default=None,
                    help="calibration JSON 경로 (wheel/base geometry override)")
parser.add_argument("--arm_limit_json", type=str, default=None,
                    help="optional arm joint limit JSON (real2sim calibration)")
parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0,
                    help="margin added to arm limits from --arm_limit_json")
parser.add_argument("--object_usd", type=str, default="",
                    help="physics grasp object USD path (empty = legacy proximity grasp)")
parser.add_argument("--multi_object_json", type=str, default="",
                    help="multi-object catalog JSON path (enables privileged 37D teacher obs)")
parser.add_argument("--object_mass", type=float, default=0.3,
                    help="physics grasp object mass (kg)")
parser.add_argument("--object_scale", type=float, default=1.0,
                    help="physics grasp object uniform scale")
parser.add_argument("--gripper_contact_prim_path", type=str, default="",
                    help="contact sensor prim path for gripper body (required in physics grasp mode)")
parser.add_argument("--grasp_gripper_threshold", type=float, default=0.7,
                    help="gripper joint position threshold for closed state")
parser.add_argument("--grasp_contact_threshold", type=float, default=0.5,
                    help="minimum contact force magnitude for grasp success")
parser.add_argument("--grasp_max_object_dist", type=float, default=0.25,
                    help="max object distance allowed for contact-based grasp success")
parser.add_argument("--grasp_attach_height", type=float, default=0.15,
                    help="attached object z-height after grasp success")
parser.add_argument("--skill", type=str, default="approach_and_grasp",
                    choices=["approach_and_grasp", "carry_and_place", "legacy"],
                    help="학습할 skill (legacy = v8 monolithic env)")
parser.add_argument("--handoff_buffer", type=str, default=None,
                    help="Skill-3용 handoff buffer pickle 경로")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

# —— 나머지 import ——————————————————————————————————————————————
import torch

from models import PolicyNet, ValueNet, CriticNet

# skrl imports (1.4.3 — 클래스 직접 참조)
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# AAC (Asymmetric Actor-Critic) components
from aac_wrapper import wrap_env_aac
from aac_ppo import AAC_PPO
from aac_trainer import AACSequentialTrainer


# ═══════════════════════════════════════════════════════════════════════
#  BC Weight Loading
# ═══════════════════════════════════════════════════════════════════════

def load_bc_into_policy(policy_model: PolicyNet, bc_checkpoint: str) -> bool:
    """
    BC 가중치를 skrl PolicyNet에 로드.

    train_bc.py의 BCPolicy와 PolicyNet은 동일한 구조:
        net.0.weight / net.0.bias   (Linear obs_dim→256)
        net.2.weight / net.2.bias   (Linear 256→128)
        net.4.weight / net.4.bias   (Linear 128→64)
        mean_layer.weight / mean_layer.bias  (Linear 64→9)

    BC에는 log_std_parameter가 없으므로 net + mean_layer만 로드.
    """
    print(f"\n  ── BC 가중치 로드 ──")
    print(f"  경로: {bc_checkpoint}")

    if not os.path.exists(bc_checkpoint):
        print(f"  ❌ 파일 없음: {bc_checkpoint}")
        return False

    bc_state = torch.load(bc_checkpoint, map_location="cpu", weights_only=True)
    policy_state = policy_model.state_dict()

    loaded = 0
    skipped = 0
    loaded_keys: set[str] = set()
    required_keys = (
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
        "mean_layer.weight",
        "mean_layer.bias",
    )

    for bc_key, bc_val in bc_state.items():
        if bc_key in policy_state:
            if (
                bc_key == "net.0.weight"
                and bc_val.ndim == 2
                and policy_state[bc_key].ndim == 2
                and bc_val.shape[0] == policy_state[bc_key].shape[0]
            ):
                # obs_dim mismatch(예: BC 33D -> RL 37D) 시 입력 차원 공통 부분만 복사.
                dst = policy_state[bc_key]
                src = bc_val.to(device=dst.device, dtype=dst.dtype)
                shared_in = min(src.shape[1], dst.shape[1])
                merged = dst.clone()
                merged[:, :shared_in] = src[:, :shared_in]
                policy_state[bc_key] = merged
                loaded += 1
                loaded_keys.add(bc_key)
                if src.shape[1] != dst.shape[1]:
                    print(
                        "  ⚠ net.0.weight 입력 차원 불일치 어댑트: "
                        f"BC in={src.shape[1]}, RL in={dst.shape[1]}, "
                        f"shared={shared_in}"
                    )
                continue
            if bc_val.shape == policy_state[bc_key].shape:
                policy_state[bc_key] = bc_val
                loaded += 1
                loaded_keys.add(bc_key)
            else:
                print(f"  ⚠ Shape 불일치: {bc_key} "
                      f"BC={bc_val.shape} vs Policy={policy_state[bc_key].shape}")
                skipped += 1
        else:
            print(f"  ⚠ Key 없음 (skip): {bc_key}")
            skipped += 1

    missing_required = [k for k in required_keys if k not in loaded_keys]
    if missing_required:
        print("  ❌ BC 핵심 레이어를 모두 로드하지 못했습니다. warm-start를 중단합니다.")
        for k in missing_required:
            print(f"    - missing: {k}")
        return False

    policy_model.load_state_dict(policy_state)

    print(f"  ✅ {loaded}개 텐서 로드 (net + mean_layer)")
    if skipped:
        print(f"  ⚠ {skipped}개 스킵")

    # BC는 deterministic → log_std를 올려서 RL 탐색 유도
    with torch.no_grad():
        policy_model.log_std_parameter.fill_(-0.5)  # 기본 -1.0 → -0.5
    print(f"  log_std → -0.5 (BC 기반 탐색 강화)")

    return True


def infer_bc_obs_dim(bc_checkpoint: str) -> int | None:
    """BC 체크포인트의 입력 obs 차원을 추론."""
    if not os.path.exists(bc_checkpoint):
        return None
    try:
        bc_state = torch.load(bc_checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        return None
    w = bc_state.get("net.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return None


def find_bc_norm_path(bc_checkpoint: str) -> str | None:
    """BC 정규화 파일 경로 추정."""
    ckpt_dir = os.path.dirname(os.path.abspath(bc_checkpoint))
    stem, _ = os.path.splitext(os.path.basename(bc_checkpoint))
    candidates = [
        os.path.join(ckpt_dir, "bc_nav_norm.npz"),
        os.path.join(ckpt_dir, f"{stem}_norm.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def _mode_label(mode: str) -> str:
    return {
        "resume": "PPO 학습 재개",
        "bc_finetune": "BC → PPO Fine-tune",
        "scratch": "PPO from scratch",
    }[mode]


def main():
    set_seed(42)

    if args.resume and not args.checkpoint:
        raise ValueError("--resume 옵션은 --checkpoint와 함께 사용해야 합니다.")

    # —— 모드 결정 ————————————————————————————————————————————
    if args.resume and args.checkpoint:
        mode = "resume"
    elif args.bc_checkpoint:
        mode = "bc_finetune"
    else:
        mode = "scratch"

    mode_label = _mode_label(mode)

    # —— Skill 분기 및 환경 생성 ——————————————————————————————
    if args.skill == "approach_and_grasp":
        from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
        env_cfg = Skill2EnvCfg()
    elif args.skill == "carry_and_place":
        from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
        if not args.handoff_buffer:
            raise ValueError("--handoff_buffer required for carry_and_place skill")
        env_cfg = Skill3EnvCfg()
        env_cfg.handoff_buffer_path = os.path.expanduser(args.handoff_buffer)
    else:
        # legacy: v8 monolithic env
        from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
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
    physics_grasp_mode = bool(str(args.object_usd).strip()) or bool(str(args.multi_object_json).strip())
    multi_object_mode = bool(str(args.multi_object_json).strip())
    if multi_object_mode:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if physics_grasp_mode:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
        env_cfg.object_mass = float(args.object_mass)
        env_cfg.object_scale = float(args.object_scale)
        env_cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
        env_cfg.grasp_gripper_threshold = float(args.grasp_gripper_threshold)
        env_cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)
        env_cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env_cfg.grasp_attach_height = float(args.grasp_attach_height)

    if args.skill == "approach_and_grasp":
        raw_env = Skill2Env(cfg=env_cfg)
    elif args.skill == "carry_and_place":
        raw_env = Skill3Env(cfg=env_cfg)
    else:
        raw_env = LeKiwiNavEnv(cfg=env_cfg)

    # AAC wrapper for skill-2/3 (critic obs 별도 전달), legacy는 symmetric
    use_aac = args.skill in ("approach_and_grasp", "carry_and_place")
    if use_aac:
        env = wrap_env_aac(raw_env)
    else:
        env = wrap_env(raw_env, wrapper="isaaclab")

    device = env.device

    if mode == "bc_finetune":
        bc_obs_dim = infer_bc_obs_dim(args.bc_checkpoint)
        env_obs_dim = int(env.observation_space.shape[0])
        if args.skill == "approach_and_grasp":
            expected_dim = 30
        elif args.skill == "carry_and_place":
            expected_dim = 29
        else:
            expected_dim = env_obs_dim
        if bc_obs_dim is None:
            raise RuntimeError(
                "BC checkpoint에서 obs_dim을 추론할 수 없습니다. "
                "유효한 BC 체크포인트를 지정하세요."
            )
        elif bc_obs_dim != env_obs_dim:
            print(
                f"  ⚠ BC obs_dim({bc_obs_dim}) != env obs_dim({env_obs_dim}) "
                "-> net.0.weight 입력 차원 어댑트로 BC warm-start를 계속 진행합니다."
            )
        if bc_obs_dim != expected_dim:
            print(
                f"  ⚠ BC obs_dim({bc_obs_dim}) != expected({expected_dim}) for skill={args.skill}. "
                "BC 체크포인트가 현재 skill과 다른 obs_dim으로 학습되었을 수 있습니다."
            )

    print("\n" + "=" * 60)
    print(f"  LeKiwi Nav PPO Training — {mode_label}")
    print(f"  Envs: {args.num_envs} | Device: {device}")
    print(f"  Obs: {env.observation_space.shape} | Act: {env.action_space.shape}")
    print(f"  Max iterations: {args.max_iterations}")
    if env_cfg.calibration_json:
        print(f"  Calibration JSON: {env_cfg.calibration_json}")
    else:
        print("  Calibration JSON: (disabled)")
    print(
        f"  Geometry in env: wheel={raw_env.wheel_radius:.6f}, "
        f"base={raw_env.base_radius:.6f}"
    )
    if args.dynamics_json:
        print(f"  Dynamics JSON: {os.path.expanduser(args.dynamics_json)}")
        print(f"  Scaled cmd limits: lin={raw_env.cfg.max_lin_vel:.4f}, ang={raw_env.cfg.max_ang_vel:.4f}")
    if args.arm_limit_json:
        print(
            f"  Arm limits : {os.path.expanduser(args.arm_limit_json)} "
            f"(margin={args.arm_limit_margin_rad:.4f} rad)"
        )
    if multi_object_mode:
        print(f"  Multi-object catalog : {os.path.expanduser(args.multi_object_json)}")
    if physics_grasp_mode and args.object_usd:
        print(f"  Physics grasp object : {os.path.expanduser(args.object_usd)}")
    if physics_grasp_mode:
        print(f"  Gripper contact prim : {args.gripper_contact_prim_path}")
        print(
            f"  Grasp thresholds     : gripper<{args.grasp_gripper_threshold:.4f}, "
            f"contact>{args.grasp_contact_threshold:.4f}, "
            f"dist<{args.grasp_max_object_dist:.4f}"
        )
    if mode == "bc_finetune":
        print(f"  BC checkpoint: {args.bc_checkpoint}")
        print(f"  BC LR scale: {args.bc_lr_scale}")
    print("=" * 60 + "\n")

    # —— Models ————————————————————————————————————————————————
    if use_aac and env._aac_state_space is not None:
        critic_obs_dim = env._aac_state_space.shape[0]
        models = {
            "policy": PolicyNet(env.observation_space, env.action_space, device),
            "value": CriticNet(
                env.observation_space, env.action_space, device,
                critic_obs_dim=critic_obs_dim,
            ),
        }
        print(f"  AAC enabled: actor={env.observation_space.shape[0]}D, critic={critic_obs_dim}D")
    else:
        models = {
            "policy": PolicyNet(env.observation_space, env.action_space, device),
            "value": ValueNet(env.observation_space, env.action_space, device),
        }

    # —— BC 가중치 로드 ————————————————————————————————————————
    if mode == "bc_finetune":
        bc_norm_path = find_bc_norm_path(args.bc_checkpoint)
        if bc_norm_path and not args.allow_bc_norm_mismatch:
            raise RuntimeError(
                "BC 정규화 파일이 감지되었습니다.\n"
                f"  - checkpoint: {args.bc_checkpoint}\n"
                f"  - norm: {bc_norm_path}\n"
                "현재 PPO는 RunningStandardScaler를 사용하므로 입력 정규화가 불일치할 수 있습니다.\n"
                "해결: train_bc.py를 기본 설정(정규화 OFF)으로 다시 학습 후 진행하세요.\n"
                "강제로 진행하려면 --allow_bc_norm_mismatch 옵션을 사용하세요."
            )
        if bc_norm_path and args.allow_bc_norm_mismatch:
            print(f"  ⚠ BC norm mismatch 허용: {bc_norm_path}")
        success = load_bc_into_policy(models["policy"], args.bc_checkpoint)
        if not success:
            raise RuntimeError("BC 로드 실패: BC->RL warm-start를 계속할 수 없습니다.")

    # —— Memory ————————————————————————————————————————————————
    memory = RandomMemory(
        memory_size=24,
        num_envs=args.num_envs,
        device=device,
    )

    # —— PPO Config ————————————————————————————————————————————
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()

    cfg_ppo["rollouts"] = 24
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 4
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95

    # BC fine-tune 시 LR을 낮춰서 BC 지식 보존
    base_lr = 3e-4
    if mode == "bc_finetune":
        cfg_ppo["learning_rate"] = base_lr * args.bc_lr_scale
        print(f"  LR: {base_lr} × {args.bc_lr_scale} = {cfg_ppo['learning_rate']:.1e} (BC 지식 보존)")
    else:
        cfg_ppo["learning_rate"] = base_lr

    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveLR
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg_ppo["grad_norm_clip"] = 0.5       # PPO 37 details: 0.5 권장
    cfg_ppo["ratio_clip"] = 0.15          # manipulation 태스크: 0.1-0.2 사이 보수적 설정
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["entropy_loss_scale"] = 0.01
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0.0
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # AAC: critic observation preprocessor (별도 scaler)
    if use_aac and env._aac_state_space is not None:
        cfg_ppo["critic_state_preprocessor"] = RunningStandardScaler
        cfg_ppo["critic_state_preprocessor_kwargs"] = {
            "size": env._aac_state_space,
            "device": device,
        }

    # Logging
    skill_tag = args.skill.replace("_", "")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"ppo_{skill_tag}")
    exp_name = f"ppo_{skill_tag}_{mode}"
    cfg_ppo["experiment"] = {
        "directory": log_dir,
        "experiment_name": exp_name,
        "write_interval": 100,
        "checkpoint_interval": 500,
    }

    # —— Agent ————————————————————————————————————————————————
    if use_aac and env._aac_state_space is not None:
        agent = AAC_PPO(
            models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            critic_observation_space=env._aac_state_space,
        )
    else:
        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    if mode == "resume":
        agent.load(args.checkpoint)
        print(f"  ✅ PPO 체크포인트 로드: {args.checkpoint}")

    # —— Trainer ——————————————————————————————————————————————
    trainer_cfg = {
        "timesteps": args.max_iterations * cfg_ppo["rollouts"] * args.num_envs,
        "headless": args.headless,
    }
    if use_aac and env._aac_state_space is not None:
        trainer = AACSequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    else:
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # —— Train! ——————————————————————————————————————————————
    print(f"\n  Training started ({mode_label})...")
    trainer.train()

    best_ckpt = os.path.join(log_dir, exp_name, "checkpoints", "best_agent.pt")
    print(f"\n  Training complete! Logs: {log_dir}")
    print(f"\n  다음 단계:")
    print(f"    tensorboard --logdir {log_dir}")
    print(f"    python collect_demos.py --checkpoint {best_ckpt} --headless")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
