# Skill 2 & 3 BC+RL 구현 가이드

## 배경 및 목표

Skill 1은 순수 RL로 거의 완성됨. Skill 2, 3는 아래 이유로 순수 RL만으로는 위험함.

- **Skill 2**: `rew_grasp_success_bonus(10.0)`, `rew_lift_bonus(5.0)`가 희박 보상(sparse reward). 물체를 잡기 전까지 RL이 gripper 동작을 학습할 signal이 없음.
- **Skill 3**: `handoff_buffer`가 없으면 `_reset_fallback()`(teleport carry)으로 빠짐 → sim2real gap 커서 실기 배포 불가.

### 채택 전략: BC 초기화 + BC Auxiliary Loss (λ annealing)

가장 검증된 방법. 논문 및 실무에서 manipulation task에 반복적으로 효과 확인됨.

```
텔레옵 데모 수집
    ↓
BC 학습 (train_bc.py) — 기존 코드 그대로
    ↓
PPO 학습 시 BC loss를 auxiliary로 유지 (λ 점진 감소)
    ↓ (Skill 2 완료 후)
BC policy 롤아웃 → Handoff Buffer 자동 생성 (신규 스크립트)
    ↓
Skill 3 BC 학습 + PPO (동일 방식)
```

**핵심**: BC 초기화만으로는 RL이 학습되면서 좋은 초기값을 망가뜨림. BC loss를 auxiliary로 유지해야 sparse reward 구간에서 policy가 무너지지 않음.

---

## 수정이 필요한 파일 목록

| 파일 | 변경 유형 | 우선순위 |
|---|---|---|
| `train_lekiwi.py` | BC auxiliary loss + λ annealing 추가 | **필수** |
| `generate_handoff_buffer.py` | 신규 스크립트 작성 | **필수** (Skill 3용) |
| `lekiwi_skill2_env.py` | reward dense shaping 보완 | 권장 |

`train_bc.py`, `lekiwi_skill3_env.py`는 수정 불필요.

---

## 1. `train_lekiwi.py` 수정

### 1-1. BCPolicy 로드 및 freeze

`train_lekiwi.py`에 이미 `--bc_checkpoint` 인자와 `load_bc_into_policy()`가 있음.  
여기에 추가로 **BC policy를 auxiliary loss용으로 별도 유지**해야 함.

```python
# 기존 import에 추가
import torch.nn.functional as F
```

```python
# main() 또는 Runner 초기화 부분에 추가
# bc_checkpoint가 있을 때 BC policy를 frozen 상태로 별도 보관
bc_policy_frozen = None
if args.bc_checkpoint and os.path.isfile(args.bc_checkpoint):
    from train_bc import BCPolicy
    obs_dim = env.cfg.observation_space   # Skill-2: 30, Skill-3: 29
    bc_policy_frozen = BCPolicy(obs_dim=obs_dim, act_dim=9).to(device)
    bc_policy_frozen.load_state_dict(
        torch.load(args.bc_checkpoint, weights_only=True)
    )
    bc_policy_frozen.eval()
    for p in bc_policy_frozen.parameters():
        p.requires_grad_(False)
    print(f"  [BC] Frozen BC policy loaded from {args.bc_checkpoint}")
```

### 1-2. λ annealing 스케줄러

```python
# Runner 또는 학습 루프 시작 전에 추가
lambda_bc_init = 0.5        # BC loss 초기 가중치
lambda_bc_final = 0.0       # 최종 가중치 (순수 RL로 수렴)
bc_anneal_steps = int(total_train_steps * 0.6)  # 전체 학습의 60%까지 annealing

def get_lambda_bc(current_step: int) -> float:
    """λ_bc를 current_step에 따라 선형 감소."""
    if bc_policy_frozen is None:
        return 0.0
    ratio = min(current_step / max(bc_anneal_steps, 1), 1.0)
    return lambda_bc_init * (1.0 - ratio) + lambda_bc_final * ratio
```

### 1-3. PPO update loop에 BC loss 추가

PPO의 actor loss를 계산하는 부분을 찾아서 아래를 추가.  
(rsl_rl 기반이면 `on_policy_runner.py` 또는 `ppo.py`의 `update()` 내부)

```python
# 기존 PPO actor loss 계산 (surrogate loss)
# surrogate_loss = ... (기존 코드)

# BC auxiliary loss 추가
if bc_policy_frozen is not None:
    lambda_bc = get_lambda_bc(current_train_step)
    if lambda_bc > 1e-6:
        with torch.no_grad():
            bc_target = bc_policy_frozen(obs_batch)  # (N, 9) — frozen BC action
        actor_output = actor_net(obs_batch)           # (N, 9) — 현재 policy mean
        bc_loss = F.mse_loss(actor_output, bc_target)
        actor_loss = surrogate_loss + lambda_bc * bc_loss
    else:
        actor_loss = surrogate_loss
        bc_loss = torch.tensor(0.0)
else:
    actor_loss = surrogate_loss
    bc_loss = torch.tensor(0.0)

# 로깅 (tensorboard 또는 wandb)
# writer.add_scalar("train/bc_loss", bc_loss.item(), current_train_step)
# writer.add_scalar("train/lambda_bc", get_lambda_bc(current_train_step), current_train_step)
```

> **주의**: `obs_batch`는 actor가 보는 관측값 (30D for Skill-2, 29D for Skill-3).  
> Critic의 37D/36D obs와 구분해야 함. BC policy는 actor obs만 받음.

### 1-4. CLI 인자 추가 (없으면)

```python
parser.add_argument("--lambda_bc_init", type=float, default=0.5)
parser.add_argument("--bc_anneal_ratio", type=float, default=0.6,
                    help="전체 학습 스텝의 몇 % 동안 BC loss annealing할지")
```

---

## 2. `generate_handoff_buffer.py` 신규 작성

Skill 2 BC/RL policy를 시뮬레이터에서 롤아웃하여 **grasp 성공 시점의 상태를 자동 수집**. Skill 3의 `handoff_buffer`로 사용.

```python
#!/usr/bin/env python3
"""
Skill-2 policy 롤아웃으로 Skill-3 Handoff Buffer 자동 생성.

Usage:
    python generate_handoff_buffer.py \
        --checkpoint checkpoints/skill2_ppo.pt \
        --num_envs 512 \
        --target_size 5000 \
        --output handoff_buffer_skill3.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle

import torch

# Isaac Lab 환경 import (실제 경로에 맞게 조정)
# from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg


def generate_handoff_buffer(
    checkpoint_path: str,
    env,                     # Skill2Env 인스턴스
    target_size: int = 5000,
    output_path: str = "handoff_buffer_skill3.pkl",
    device: str = "cpu",
):
    """
    Skill-2 policy를 env에서 롤아웃하며 grasp 성공 상태를 수집.

    수집하는 항목 (Skill-3 _reset_from_handoff가 기대하는 형식):
        base_pos      : (3,) float — env origin 기준 상대 좌표
        base_ori      : (4,) float — quaternion [w, x, y, z]
        object_pos    : (3,) float — env origin 기준 상대 좌표
        object_ori    : (4,) float — quaternion [w, x, y, z]
        arm_joints    : (5,) float — arm joint positions (gripper 제외)
        gripper_state : float      — gripper joint position
        object_type_idx: int       — active_object_idx
    """
    # Policy 로드 (rsl_rl ActorCritic 또는 BCPolicy 구조)
    policy = torch.load(checkpoint_path, map_location=device)
    if hasattr(policy, "actor"):
        actor = policy.actor
    else:
        actor = policy
    actor.eval()

    buffer = []
    obs_dict, _ = env.reset()

    print(f"  [HandoffGen] target={target_size}, num_envs={env.num_envs}")

    with torch.no_grad():
        while len(buffer) < target_size:
            obs = obs_dict["policy"]

            # Policy 추론
            actions = actor(obs)
            if hasattr(actions, "mean"):
                actions = actions.mean  # 분포인 경우 mean 사용
            actions = actions.clamp(-1.0, 1.0)

            obs_dict, _, terminated, truncated, _ = env.step(actions)

            # Grasp 성공한 env 찾기 (just_grasped: 이번 step에 처음 잡은 env)
            just_grasped = env.just_grasped  # (num_envs,) bool

            if just_grasped.any():
                env_origins = env.scene.env_origins  # (num_envs, 3)
                success_ids = just_grasped.nonzero(as_tuple=False).squeeze(-1)

                for eid in success_ids:
                    eid_i = int(eid.item())
                    origin = env_origins[eid_i].cpu()

                    # 상대 좌표로 변환 (env origin 빼기)
                    base_pos_abs = env.robot.data.root_pos_w[eid_i].cpu()
                    obj_pos_abs = env.object_pos_w[eid_i].cpu()

                    entry = {
                        "base_pos": (base_pos_abs - origin).tolist(),
                        "base_ori": env.robot.data.root_quat_w[eid_i].cpu().tolist(),
                        "object_pos": (obj_pos_abs - origin).tolist(),
                        "object_ori": (
                            env.object_rigid.data.root_quat_w[eid_i].cpu().tolist()
                            if env.object_rigid is not None
                            else [1.0, 0.0, 0.0, 0.0]
                        ),
                        "arm_joints": env.robot.data.joint_pos[eid_i, env.arm_idx[:5]].cpu().tolist(),
                        "gripper_state": float(env.robot.data.joint_pos[eid_i, env.gripper_idx].item()),
                        "object_type_idx": int(env.active_object_idx[eid_i].item()),
                    }
                    buffer.append(entry)

                    if len(buffer) % 500 == 0:
                        print(f"  [HandoffGen] collected {len(buffer)}/{target_size}")

                    if len(buffer) >= target_size:
                        break

    with open(output_path, "wb") as f:
        pickle.dump(buffer, f)

    print(f"\n  [HandoffGen] Done. {len(buffer)} entries → {output_path}")
    return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--target_size", type=int, default=5000)
    parser.add_argument("--output", type=str, default="handoff_buffer_skill3.pkl")
    args = parser.parse_args()

    # 실제 env 생성 코드는 프로젝트의 run_env.py 또는 train_lekiwi.py 방식 따름
    # env = Skill2Env(cfg, ...)
    # generate_handoff_buffer(args.checkpoint, env, args.target_size, args.output)
    print("env 생성 코드를 프로젝트 방식에 맞게 추가할 것")
```

---

## 3. `lekiwi_skill2_env.py` Reward 보완 (권장)

BC auxiliary loss가 있어도 reward 설계가 나쁘면 학습이 느림. 아래 두 가지를 추가.

### 3-1. Gripper Dense Shaping 강화

현재 코드에 이미 gripper shaping이 있지만 조건이 `approach_thresh(0.35m)` 이내일 때만 발동. 범위를 넓힘.

```python
# 기존 (lekiwi_skill2_env.py _get_rewards 내)
near_object = metrics["object_dist"] < self.cfg.approach_thresh   # 0.35m

# 수정: 더 넓은 범위에서 gripper closing 유도
wider_near_object = metrics["object_dist"] < (self.cfg.approach_thresh * 2.0)  # 0.70m
reward += wider_near_object.float() * (0.3 * close_progress)  # 약한 버전
reward += near_object.float() * (0.5 * close_progress)        # 기존 강한 버전
```

### 3-2. Approach Progress에 높이 패널티 추가

물체가 낮은 위치에 있을 때 로봇이 지나치게 위에서 접근하는 것을 방지.

```python
# _get_rewards() 내 추가
# object_pos_b의 z 성분이 너무 크면 패널티 (팔이 물체 위로 접근하지 않도록)
vertical_offset = metrics["object_pos_b"][:, 2].abs()
height_penalty = -0.1 * torch.clamp(vertical_offset - 0.10, min=0.0)
reward += height_penalty
```

---

## 실행 순서

```bash
# Step 1: 텔레옵 데모 수집 (Skill 2)
python record_teleop.py --skill 2 --output demos_skill2/

# Step 2: BC 학습 (train_bc.py 그대로)
python train_bc.py \
    --demo_dir demos_skill2/ \
    --epochs 200 \
    --expected_obs_dim 30 \
    --save_dir checkpoints/

# Step 3: Skill 2 PPO (BC init + BC auxiliary loss)
python train_lekiwi.py \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_nav.pt \
    --lambda_bc_init 0.5 \
    --bc_anneal_ratio 0.6 \
    --headless

# Step 4: Handoff Buffer 자동 생성 (BC/RL 체크포인트 사용)
python generate_handoff_buffer.py \
    --checkpoint checkpoints/skill2_ppo.pt \
    --num_envs 512 \
    --target_size 5000 \
    --output handoff_buffer_skill3.pkl

# Step 5: 텔레옵 데모 수집 (Skill 3)
python record_teleop.py --skill 3 --output demos_skill3/

# Step 6: BC 학습 (Skill 3)
python train_bc.py \
    --demo_dir demos_skill3/ \
    --epochs 200 \
    --expected_obs_dim 29 \
    --save_dir checkpoints/

# Step 7: Skill 3 PPO (BC init + BC auxiliary loss + Handoff Buffer)
python train_lekiwi.py \
    --skill 3 \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_nav.pt \
    --handoff_buffer handoff_buffer_skill3.pkl \
    --lambda_bc_init 0.5 \
    --bc_anneal_ratio 0.6 \
    --headless
```

---

## 구현 시 주의사항

### train_lekiwi.py 수정 시

1. **obs_batch 차원 확인 필수**: BC policy는 actor obs (30D/29D)만 받음. critic obs (37D/36D)를 넘기면 shape mismatch.
2. **bc_policy_frozen.eval() 확인**: BC policy가 학습 중 BatchNorm이나 Dropout을 사용하면 eval mode 강제 필요.
3. **gradient 차단 확인**: `bc_target`은 반드시 `torch.no_grad()` 안에서 계산. BC policy의 gradient가 PPO update에 섞이면 안 됨.

### generate_handoff_buffer.py 작성 시

1. **상대 좌표 변환 필수**: `Skill3Env._reset_from_handoff()`가 `base_pos + env_origins`로 절대 좌표를 복원함. env origin을 빼서 저장해야 함.
2. **just_grasped 타이밍**: `env.step()` 직후에만 유효. step을 더 진행하면 False로 초기화됨.
3. **buffer 다양성**: 같은 에피소드에서 연속 수집하면 유사한 state만 쌓임. 에피소드가 reset될 때마다 다양한 entry가 들어오도록 충분한 `target_size` 설정 (최소 2000).

### λ annealing 값 가이드

| 상황 | `lambda_bc_init` | `bc_anneal_ratio` |
|---|---|---|
| 데모 품질 좋음 (50+ 에피소드) | 0.5 | 0.6 |
| 데모 적음 (20 이하) | 0.3 | 0.4 |
| grasp 거의 안 됨 (초기 학습 불안정) | 1.0 | 0.7 |

---

## 요약

Claude Code에게 요청할 작업 순서:

1. **`train_lekiwi.py`**: `BCPolicy` frozen 로드 → `get_lambda_bc()` 함수 추가 → PPO actor loss에 `lambda_bc * F.mse_loss(actor_output, bc_target)` 추가
2. **`generate_handoff_buffer.py`**: 위 코드를 기반으로 프로젝트의 env 생성 방식에 맞게 작성
3. **`lekiwi_skill2_env.py`**: `_get_rewards()`에 wider gripper shaping과 height penalty 추가
