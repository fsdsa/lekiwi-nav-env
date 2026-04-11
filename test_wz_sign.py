"""wz sign 검증 — GUI에서 양/음 wz를 순차 적용해 회전 방향 확인."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg

cfg = Skill2EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.enable_domain_randomization = False
cfg.arm_limit_write_to_sim = False
cfg.episode_length_s = 600.0
env = Skill2Env(cfg=cfg)
env.reset()

device = env.device
zero = torch.zeros(1, 9, dtype=torch.float32, device=device)

def apply(action_value, label, n_steps=120):
    print(f"\n>>> {label}: action[8]={action_value:+.3f}")
    print(f"    (max_ang_vel={cfg.max_ang_vel}, expected body_wz={action_value*cfg.max_ang_vel:+.3f} rad/s)")
    a = zero.clone()
    a[0, 8] = action_value
    for i in range(n_steps):
        env.step(a)
        if i % 20 == 0:
            wz_actual = env.robot.data.root_ang_vel_b[0, 2].item()
            quat = env.robot.data.root_quat_w[0]
            yaw = 2.0 * torch.atan2(quat[3], quat[0]).item()
            print(f"    step {i:3d}: actual_wz={wz_actual:+.3f}  yaw={yaw:+.3f} rad")

# 정지 → 양수 wz → 정지 → 음수 wz
print("\n========================================")
print("  wz sign 검증 시작")
print("========================================")

print("\n[0] 정지 (60 step)")
for _ in range(60):
    env.step(zero)

apply(+0.33, "POSITIVE wz +0.33 (CCW이면 좌회전)", 120)

print("\n[2] 정지 (60 step)")
for _ in range(60):
    env.step(zero)

apply(-0.33, "NEGATIVE wz -0.33 (CW이면 우회전)", 120)

print("\n[4] 정지 — GUI에서 30초 대기 (Ctrl+C로 종료)")
for _ in range(900):
    env.step(zero)

env.close()
simulation_app.close()
