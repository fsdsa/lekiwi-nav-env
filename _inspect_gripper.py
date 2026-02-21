#!/usr/bin/env python3
"""Quick script to inspect gripper body prims in the robot USD."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
launcher = AppLauncher(args)
sim_app = launcher.app

import omni.usd
from pxr import Usd, UsdPhysics

# Spawn robot using the USD directly
usd_path = os.environ.get("LEKIWI_USD_PATH", "/home/yubin11/Downloads/lekiwi_robot.usd")
import isaaclab.sim as sim_utils
spawn_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, activate_contact_sensors=True)
spawn_cfg.func("/World/Robot", spawn_cfg)

stage = omni.usd.get_context().get_stage()

print("\n=== Gripper-related prims ===", flush=True)
for prim in stage.Traverse():
    name = prim.GetName().lower()
    path = str(prim.GetPath())
    if any(k in name for k in ["jaw", "grip", "finger", "moving"]):
        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        print(f"  {path}  rigid={has_rb}  collision={has_col}  type={prim.GetTypeName()}", flush=True)

print("\n=== All rigid bodies under /World/Robot ===", flush=True)
count = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/World/Robot") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  {path}", flush=True)
        count += 1
print(f"  Total: {count} rigid bodies", flush=True)

sim_app.close()
