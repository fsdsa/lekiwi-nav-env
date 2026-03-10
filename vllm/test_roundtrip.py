#!/usr/bin/env python3
"""
로컬 Isaac Sim → 서버 VLM/VLA 라운드트립 속도 측정.

서버에서 VLM(port 8000) + VLA(port 8002) 가동 상태에서 실행:
    python test_roundtrip.py --object_usd <path> --num_steps 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roundtrip latency test")
parser.add_argument("--vlm_server", type=str, default="http://localhost:8000")
parser.add_argument("--vla_server", type=str, default="http://localhost:8002")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--num_steps", type=int, default=20)
parser.add_argument("--camera_width", type=int, default=320)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--jpeg_quality", type=int, default=70)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

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

from vlm_prompts import NAVIGATE_SYSTEM_PROMPT, NAVIGATE_USER_TEMPLATE, COMMAND_TO_DIRECTION

# ── Phase → VLA instruction 매핑 ──
PHASE_INSTRUCTION = {
    "NAVIGATE_SEARCH": "explore the room to find the {target}",
    "NAVIGATE_TO_TARGET": "move toward the {target}",
    "APPROACH_AND_LIFT": "approach and pick up the {target}",
    "NAVIGATE_TO_DEST": "carry the {target} and find the {dest}",
    "NAVIGATE_TO_DEST_CLOSE": "move toward the {dest} while holding the {target}",
    "CARRY_AND_PLACE": "place the {target} next to the {dest}",
}

# ── Env setup ──
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg

cfg = Skill2EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.enable_domain_randomization = False
cfg.arm_limit_write_to_sim = False
cfg.episode_length_s = 600.0
if args.object_usd:
    cfg.object_usd = os.path.expanduser(args.object_usd)
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    cfg.arm_limit_json = args.arm_limit_json
cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

env = Skill2Env(cfg=cfg)

# ── Cameras via omni.replicator ──
import omni.replicator.core as rep

base_cam_path = "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
wrist_cam_path = "/World/envs/env_0/Robot/LeKiwi/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera"

# Create render products
base_rp = rep.create.render_product(base_cam_path, (args.camera_width, args.camera_height))
wrist_rp = rep.create.render_product(wrist_cam_path, (args.camera_width, args.camera_height))

# Annotators for RGB
base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
base_rgb_annot.attach([base_rp])

wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
wrist_rgb_annot.attach([wrist_rp])

# Depth annotator
base_depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
base_depth_annot.attach([base_rp])


def encode_image(rgb: np.ndarray, quality: int = 70) -> str:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def get_state_9d() -> list[float]:
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].tolist()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].tolist()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return arm + [grip] + bv[:2] + [wz]


# ── Reset ──
print("\n  Resetting env...")
obs, info = env.reset()
dt = env.sim.cfg.dt

# warm up cameras (a few sim steps + render)
for _ in range(5):
    env.sim.step()
    env.sim.render()

# ── Health check ──
sess = requests.Session()

print(f"\n  VLM server: {args.vlm_server}")
try:
    r = sess.get(f"{args.vlm_server}/v1/models", timeout=3)
    print(f"    models: {r.json()['data'][0]['id']}")
except Exception as e:
    print(f"    FAIL: {e}")

print(f"  VLA server: {args.vla_server}")
try:
    r = sess.get(f"{args.vla_server}/health", timeout=3)
    print(f"    health: {r.json()}")
except Exception as e:
    print(f"    FAIL: {e}")

# ── Roundtrip test ──
print(f"\n  === Roundtrip test: {args.num_steps} steps ===\n")

target_object = "medicine bottle"
dest_object = "red cup"
current_phase = "NAVIGATE_SEARCH"
vlm_cmd = "FORWARD"
vla_instruction = PHASE_INSTRUCTION["NAVIGATE_SEARCH"].format(target=target_object, dest=dest_object)

timings = {
    "cam_capture": [],
    "encode": [],
    "vlm": [],
    "vla": [],
    "env_step": [],
    "total": [],
}

for step in range(args.num_steps):
    t_total = time.perf_counter()

    # 1. Camera capture (omni.replicator annotators)
    t0 = time.perf_counter()
    env.sim.step()
    env.sim.render()

    base_rgb_data = base_rgb_annot.get_data()
    wrist_rgb_data = wrist_rgb_annot.get_data()
    depth_data = base_depth_annot.get_data()

    if base_rgb_data is None or wrist_rgb_data is None:
        print(f"  [{step}] cam data not ready, skip")
        continue

    # replicator returns (H, W, 4) RGBA uint8
    base_rgb = np.array(base_rgb_data)[..., :3]
    wrist_rgb = np.array(wrist_rgb_data)[..., :3]
    depth = np.array(depth_data) if depth_data is not None else None
    timings["cam_capture"].append(time.perf_counter() - t0)

    # 2. Encode
    t0 = time.perf_counter()
    b64_base = encode_image(base_rgb, args.jpeg_quality)
    b64_wrist = encode_image(wrist_rgb, args.jpeg_quality)
    timings["encode"].append(time.perf_counter() - t0)

    # 3. VLM — 실제 프롬프트 사용, 매 5스텝 호출
    if step % 5 == 0:
        t0 = time.perf_counter()
        try:
            r = sess.post(f"{args.vlm_server}/v1/chat/completions", json={
                "model": args.vlm_model,
                "messages": [
                    {"role": "system", "content": NAVIGATE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_base}"}},
                        {"type": "text", "text": NAVIGATE_USER_TEMPLATE.format(target_object=target_object)},
                    ]},
                ],
                "max_tokens": 10, "temperature": 0.0,
            }, timeout=5)
            raw = r.json()["choices"][0]["message"]["content"].strip().upper()
            # 유효한 command 파싱
            vlm_cmd = "FORWARD"
            for cmd in COMMAND_TO_DIRECTION:
                if cmd in raw:
                    vlm_cmd = cmd
                    break
            # VLM command → VLA instruction 업데이트
            if vlm_cmd == "STOP":
                vla_instruction = PHASE_INSTRUCTION["APPROACH_AND_LIFT"].format(
                    target=target_object, dest=dest_object)
            else:
                vla_instruction = PHASE_INSTRUCTION[current_phase].format(
                    target=target_object, dest=dest_object)
        except Exception as e:
            vlm_cmd = f"ERROR: {e}"
        timings["vlm"].append(time.perf_counter() - t0)
        if step < 5 or step % 5 == 0:
            print(f"    [{step}] VLM: {vlm_cmd} → inst: \"{vla_instruction[:50]}\"")

    # 4. VLA — VLM instruction 연동, base+wrist 이미지 전달
    t0 = time.perf_counter()
    state = get_state_9d()
    try:
        r = sess.post(f"{args.vla_server}/act", json={
            "base_image_b64": b64_base,
            "wrist_image_b64": b64_wrist,
            "state": state,
            "instruction": vla_instruction,
        }, timeout=15)
        if r.status_code == 200:
            vla_data = r.json()
            actions = vla_data["actions"]
            vla_time = vla_data["inference_time_ms"]
        else:
            actions = []
            vla_time = -1
    except Exception as e:
        actions = []
        vla_time = -1
        print(f"    [{step}] VLA error: {e}")
    timings["vla"].append(time.perf_counter() - t0)

    # 5. env step with action (or zero)
    t0 = time.perf_counter()
    if actions:
        a = actions[0][:9] if len(actions[0]) >= 9 else actions[0] + [0.0] * (9 - len(actions[0]))
        action_tensor = torch.tensor([a], dtype=torch.float32, device="cuda:0")
    else:
        action_tensor = torch.zeros(1, 9, dtype=torch.float32, device="cuda:0")
    obs, rew, term, trunc, info = env.step(action_tensor)
    timings["env_step"].append(time.perf_counter() - t0)

    timings["total"].append(time.perf_counter() - t_total)

    if step < 3 or step % 5 == 0:
        n_act = len(actions)
        dim = len(actions[0]) if n_act > 0 else 0
        print(f"    [{step:3d}] total={timings['total'][-1]*1000:.0f}ms | "
              f"cam={timings['cam_capture'][-1]*1000:.0f}ms "
              f"enc={timings['encode'][-1]*1000:.0f}ms "
              f"vla={timings['vla'][-1]*1000:.0f}ms(srv={vla_time:.0f}ms) "
              f"step={timings['env_step'][-1]*1000:.0f}ms | "
              f"act={n_act}x{dim}D")

# ── Summary ──
print(f"\n  {'='*60}")
print(f"  Roundtrip Summary ({len(timings['total'])} steps)")
print(f"  {'='*60}")
for key in ["cam_capture", "encode", "vlm", "vla", "env_step", "total"]:
    vals = timings[key]
    if vals:
        avg = np.mean(vals) * 1000
        std = np.std(vals) * 1000
        mn = np.min(vals) * 1000
        mx = np.max(vals) * 1000
        print(f"  {key:15s}: avg={avg:7.1f}ms  std={std:5.1f}ms  min={mn:7.1f}ms  max={mx:7.1f}ms")

total_avg = np.mean(timings["total"])
hz = 1.0 / total_avg if total_avg > 0 else 0
print(f"\n  Overall: {hz:.2f} Hz ({total_avg*1000:.0f}ms per step)")
print(f"  VLM: {1.0/np.mean(timings['vlm']):.2f} Hz" if timings["vlm"] else "  VLM: not tested")
print(f"  VLA: {1.0/np.mean(timings['vla']):.2f} Hz" if timings["vla"] else "  VLA: not tested")

simulation_app.close()
