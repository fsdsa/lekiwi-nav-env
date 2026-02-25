#!/usr/bin/env python3
"""
Navigate (Skill-1) 학습 결과 GUI 시각화 검증 스크립트.

학습된 best_agent.pt를 로드하여 Isaac Sim GUI에서 로봇 동작을 시각적으로 확인.
카메라/HDF5 저장 없이 순수 시각화만 수행.

사용법:
    python eval_navigate.py
    python eval_navigate.py --num_envs 8
    python eval_navigate.py --checkpoint <path_to_checkpoint>
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys

# Isaac Sim 엔진 초기화 전 OS 레벨에서 로그 강제 차단
os.environ["CARB_LOG_LEVEL"] = "fatal"
os.environ["OMNI_LOG_LEVEL"] = "fatal"
os.environ["PHYSX_LOG_LEVEL"] = "fatal"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Navigate Skill-1 GUI 평가")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/ppo_navigate/ppo_navigate_scratch/checkpoints/best_agent.pt",
    help="학습된 체크포인트 경로",
)
parser.add_argument("--num_envs", type=int, default=1, help="병렬 환경 수 (검증: 1 권장)")
parser.add_argument("--dynamics_json", type=str, default=None, help="dynamics calibration JSON")
parser.add_argument("--arm_limit_json", type=str, default=None, help="arm joint limit JSON")
parser.add_argument("--max_steps", type=int, default=0, help="최대 step 수 (0=무한)")
parser.add_argument("--no_masking", action="store_true", help="inference-time masking 비활성화")
parser.add_argument("--print_interval", type=int, default=200, help="메트릭 출력 간격 (steps)")
parser.add_argument("--sequential", action="store_true", default=True,
                    help="6방향을 순서대로 평가 (fwd→bwd→left→right→turnL→turnR, 기본값)")
parser.add_argument("--random", action="store_true",
                    help="방향을 랜덤으로 평가 (기본 sequential 해제)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import torch

# AppLauncher 로드 이후 Omniverse 내부 로거 채널 무효화
try:
    import omni.log
    log_iface = omni.log.get_log()
    log_iface.set_channel_enabled("omni.physx.tensors.plugin", False)
    log_iface.set_channel_enabled("omni.physx.plugin", False)
    log_iface.set_channel_enabled("usdrt.hydra.fabric_scene_delegate.plugin", False)
    log_iface.set_channel_enabled("omni.fabric.plugin", False)
except Exception:
    pass

from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
from models import PolicyNet, CriticNet

# AAC (Asymmetric Actor-Critic)
from aac_wrapper import wrap_env_aac
from aac_ppo import AAC_PPO

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Direction command labels
CMD_LABELS = ["forward", "backward", "strafe_left", "strafe_right", "turn_left", "turn_right"]
CMD_VECTORS = [
    (0.0, 1.0, 0.0),     # forward
    (0.0, -1.0, 0.0),    # backward
    (-1.0, 0.0, 0.0),    # strafe left
    (1.0, 0.0, 0.0),     # strafe right
    (0.0, 0.0, 0.33),    # turn left CCW
    (0.0, 0.0, -0.33),   # turn right CW
]

def cmd_to_label(cmd_vec: torch.Tensor) -> str:
    for i, (vx, vy, wz) in enumerate(CMD_VECTORS):
        ref = torch.tensor([vx, vy, wz], device=cmd_vec.device)
        if torch.allclose(cmd_vec, ref, atol=0.1):
            return CMD_LABELS[i]
    return f"({cmd_vec[0]:.2f},{cmd_vec[1]:.2f},{cmd_vec[2]:.2f})"

class SuppressCLogs:
    """C/C++ 엔진 레벨에서 발생하는 모든 콘솔 출력을 OS 단에서 물리적으로 차단하는 컨텍스트 매니저"""
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        os.close(self.null_fd)
        os.close(self.save_stdout)
        os.close(self.save_stderr)

def main():
    ckpt = os.path.expanduser(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt}")

    env_cfg = Skill1EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.arm_limit_write_to_sim = False
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
    
    env_cfg.eval_cardinal_yaw = True
    

    # 환경 초기화 중 발생하는 C++ 에러도 차단
    with SuppressCLogs():
        raw_env = Skill1Env(cfg=env_cfg)

    env = wrap_env_aac(raw_env)
    device = env.device

    critic_obs_dim = env._aac_state_space.shape[0]
    models = {
        "policy": PolicyNet(env.observation_space, env.action_space, device),
        "value": CriticNet(
            env.observation_space, env.action_space, device,
            critic_obs_dim=critic_obs_dim,
        ),
    }
    memory = RandomMemory(memory_size=24, num_envs=args.num_envs, device=device)

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg_ppo["critic_state_preprocessor"] = RunningStandardScaler
    cfg_ppo["critic_state_preprocessor_kwargs"] = {"size": env._aac_state_space, "device": device}

    agent = AAC_PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        critic_observation_space=env._aac_state_space,
    )
    agent.load(ckpt)
    agent.set_running_mode("eval")

    pre = getattr(agent, "_state_preprocessor", None) or getattr(agent, "state_preprocessor", None)
    policy_model = models["policy"]

    pre_mean = pre.running_mean if hasattr(pre, "running_mean") else None
    pre_ok = pre_mean is not None and pre_mean.abs().sum().item() > 0
    print("\n" + "=" * 60)
    print("  Navigate (Skill-1) GUI 평가 — Pure Velocity Tracker (v6e)")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Num envs   : {args.num_envs}")
    print(f"  Obs dim    : {env.observation_space.shape[0]}D (actor) / {critic_obs_dim}D (critic)")
    print(f"  Preprocessor: {'LOADED (count={})'.format(int(pre.current_count.item())) if pre_ok else 'NOT LOADED!'}")
    if pre_ok:
        m = pre.running_mean.float()
        print(f"  Pre mean[6:12]: vx={m[6]:.4f} vy={m[7]:.4f} wz={m[8]:.4f} cx={m[9]:.4f} cy={m[10]:.4f} cz={m[11]:.4f}")
    print(f"  Max steps  : {'무한' if args.max_steps == 0 else args.max_steps}")
    print("  종료: Ctrl+C 또는 창 닫기")
    print("=" * 60 + "\n")

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_navigate_log.txt")
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # Sequential mode: 6방향 순서대로 평가
    seq_idx = 0  # 현재 방향 인덱스
    per_cmd_metrics = {label: {"compliance": 0.0, "speed": 0.0, "lin_err": 0.0, "ang_err": 0.0, "steps": 0}
                       for label in CMD_LABELS}

    # 로봇 yaw를 +Y world 방향으로 고정하는 헬퍼
    _FIXED_YAW = math.pi / 2  # yaw=π/2 → robot forward(+Y body) = +Y world
    def _fix_yaw():
        """Reset 직후 모든 env의 yaw를 +Y world로 덮어쓴다."""
        root = raw_env.robot.data.default_root_state[:raw_env.num_envs].clone()
        cur = raw_env.robot.data.root_state_w[:raw_env.num_envs].clone()
        # xy/z 위치는 현재값 유지, orientation만 고정
        half = _FIXED_YAW * 0.5
        cur[:, 3] = math.cos(half)   # qw
        cur[:, 4] = 0.0              # qx
        cur[:, 5] = 0.0              # qy
        cur[:, 6] = math.sin(half)   # qz
        cur[:, 7:] = 0.0             # 속도 초기화
        env_ids = torch.arange(raw_env.num_envs, device=device)
        raw_env.robot.write_root_state_to_sim(cur, env_ids)

    with SuppressCLogs():
        obs, _ = raw_env.reset()
    _fix_yaw()
    # reset 후 obs 갱신 (고정 yaw 반영)
    with SuppressCLogs():
        raw_env.scene.update(dt=raw_env.physics_dt)
        obs = raw_env._get_observations()

    if args.random:
        args.sequential = False

    if args.sequential:
        # 첫 번째 방향으로 강제 설정
        cmd_vec = torch.tensor(CMD_VECTORS[seq_idx], device=device)
        raw_env._direction_cmd[:] = cmd_vec
        log(f"[Sequential] 방향 {seq_idx+1}/6: {CMD_LABELS[seq_idx]}")

    step = 0
    episode_count = 0
    cumulative_compliance = 0.0
    cumulative_speed = 0.0
    cumulative_lin_err = 0.0
    cumulative_ang_err = 0.0
    metric_steps = 0

    log("=== Direction commands ===")
    for i in range(min(args.num_envs, 8)):
        label = cmd_to_label(raw_env._direction_cmd[i])
        log(f"  env {i}: {label}")

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            with torch.no_grad():
                proc = pre(obs["policy"], train=False) if callable(pre) else obs["policy"]
                feat = policy_model.net(proc)
                action = policy_model.mean_layer(feat).clamp(-1.0, 1.0).contiguous()

                # Inference-time masking: zero out cross-axis residuals
                if not args.no_masking:
                    _cmd = raw_env._direction_cmd
                    fwd_bwd = (_cmd[:, 1].abs() > 0.5)
                    strafe  = (_cmd[:, 0].abs() > 0.5)
                    action[fwd_bwd, 6] = 0.0; action[fwd_bwd, 8] = 0.0
                    action[strafe, 7] = 0.0;  action[strafe, 8] = 0.0

            # 엔진 스텝이 실행되는 동안 발생하는 모든 에러 출력을 하드웨어 수준에서 버림
            with SuppressCLogs():
                obs, reward, terminated, truncated, _ = raw_env.step(action)
                
            step += 1

            if step % 10 == 1:
                a = action[0]
                bv = raw_env._read_base_body_vel()[0]
                cmd = raw_env._direction_cmd[0]
                cmd_label = cmd_to_label(cmd)
                log(
                    f"[DBG] step={step:4d}  cmd={cmd_label:13s}  "
                    f"act=[{a[6]:+.3f},{a[7]:+.3f},{a[8]:+.3f}]  "
                    f"body_vel=[{bv[0]:+.3f},{bv[1]:+.3f},{bv[2]:+.3f}]  "
                    f"reward={reward[0]:.3f}"
                )

            extras_log = raw_env.extras.get("log", {})
            if extras_log:
                cumulative_compliance += float(extras_log.get("direction_compliance", 0))
                cumulative_speed += float(extras_log.get("avg_speed", 0))
                cumulative_lin_err += float(extras_log.get("lin_vel_error", 0))
                cumulative_ang_err += float(extras_log.get("ang_vel_error", 0))
                metric_steps += 1

            # 방향별 메트릭 집계
            if extras_log:
                cur_label = cmd_to_label(raw_env._direction_cmd[0])
                if cur_label in per_cmd_metrics:
                    m = per_cmd_metrics[cur_label]
                    m["compliance"] += float(extras_log.get("direction_compliance", 0))
                    m["speed"] += float(extras_log.get("avg_speed", 0))
                    m["lin_err"] += float(extras_log.get("lin_vel_error", 0))
                    m["ang_err"] += float(extras_log.get("ang_vel_error", 0))
                    m["steps"] += 1

            done = terminated | truncated
            num_done = done.sum().item()
            if num_done > 0:
                episode_count += num_done
                _fix_yaw()

                if args.sequential:
                    seq_idx = (seq_idx + 1) % len(CMD_LABELS)
                    cmd_vec = torch.tensor(CMD_VECTORS[seq_idx], device=device)
                    raw_env._direction_cmd[:] = cmd_vec
                    log(f"[Sequential] 방향 {seq_idx+1}/6: {CMD_LABELS[seq_idx]}")
                else:
                    reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    for idx in reset_ids:
                        i = idx.item()
                        if i < 8:
                            label = cmd_to_label(raw_env._direction_cmd[i])
                            log(f"[Reset] env {i} -> cmd: {label}")

            if step % args.print_interval == 0 and metric_steps > 0:
                avg_comp = cumulative_compliance / metric_steps
                avg_spd = cumulative_speed / metric_steps
                avg_lin_err = cumulative_lin_err / metric_steps
                avg_ang_err = cumulative_ang_err / metric_steps
                cur_label = cmd_to_label(raw_env._direction_cmd[0])
                log(
                    f"step={step:5d}  cmd={cur_label:13s}  "
                    f"compliance={avg_comp:.3f}  speed={avg_spd:.3f}m/s  "
                    f"lin_err={avg_lin_err:.4f}  ang_err={avg_ang_err:.4f}  "
                    f"ep={episode_count}"
                )

    except KeyboardInterrupt:
        log("사용자 중단 (Ctrl+C)")

    if metric_steps > 0:
        log(
            f"FINAL: {step} steps, {episode_count} ep | "
            f"compliance={cumulative_compliance / metric_steps:.4f}  "
            f"speed={cumulative_speed / metric_steps:.3f}m/s  "
            f"lin_err={cumulative_lin_err / metric_steps:.4f}  "
            f"ang_err={cumulative_ang_err / metric_steps:.4f}"
        )

    # 방향별 메트릭 출력
    has_per_cmd = any(m["steps"] > 0 for m in per_cmd_metrics.values())
    if has_per_cmd:
        log("")
        log(f"{'direction':>13s}  {'compliance':>10s}  {'speed':>8s}  {'lin_err':>8s}  {'ang_err':>8s}  {'steps':>6s}")
        log("-" * 65)
        for label in CMD_LABELS:
            m = per_cmd_metrics[label]
            if m["steps"] > 0:
                n = m["steps"]
                log(f"{label:>13s}  {m['compliance']/n:>10.4f}  {m['speed']/n:>7.3f}  {m['lin_err']/n:>8.4f}  {m['ang_err']/n:>8.4f}  {n:>6d}")
            else:
                log(f"{label:>13s}  {'—':>10s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'0':>6s}")

    log_f.close()
    print(f"\n  로그 저장: {log_path}")
    
    with SuppressCLogs():
        sim_app.close()

if __name__ == "__main__":
    main()
