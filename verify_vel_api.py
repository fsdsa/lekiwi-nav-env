# ~/IsaacLab/scripts/lekiwi_nav_env/verify_vel_api.py
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))

# LeKiwi USD 직접 로드
from lekiwi_robot_cfg import LEKIWI_CFG
robot_cfg = LEKIWI_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg=robot_cfg)

sim.reset()
robot.reset()

# API 확인
d = robot.data
print("=== VELOCITY API ===")
for attr in ["root_lin_vel_b", "root_ang_vel_b", "root_lin_vel_w", "root_ang_vel_w"]:
    exists = hasattr(d, attr)
    shape = getattr(d, attr).shape if exists else "N/A"
    print(f"  {attr}: exists={exists}, shape={shape}")

print("\n=== BODY API ===")
print(f"  body_pos_w shape: {d.body_pos_w.shape}")
print(f"  body_names: {robot.body_names}")

try:
    ids, names = robot.find_bodies(["Moving_Jaw_08d_v1"])
    print(f"  find_bodies('Moving_Jaw_08d_v1'): ids={ids}, names={names}")
except Exception as e:
    print(f"  find_bodies error: {e}")

sim_app.close()
