"""
LeKiwi Sim vs Real Calibration Test (Isaac Sim Script Editor)
==============================================================

사용법:
  1) Isaac Sim에서 LeKiwi USD 로드
  2) Play 상태 확인
  3) Script Editor에서 이 파일 내용을 실행

핵심 수정점:
  - pose 측정 prim을 /World/LeKiwi가 아니라 실제 articulation root로 사용
    (기본: /World/LeKiwi/base_plate_layer1_v5)
  - ratio(R/S)=real/sim 해석을 명확히 출력
"""

import asyncio
import json
import math
import os

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# =============================================================================
# 설정값
# =============================================================================
ROBOT_PRIM_PATH = "/World/LeKiwi"
# 핵심: pose 측정은 실제 물리 루트(body)에서 해야 함
POSE_PRIM_PATH_OVERRIDE = "/World/LeKiwi/base_plate_layer1_v5"

# 물리 기하값
WHEEL_RADIUS = 0.049  # m
BASE_RADIUS = 0.1085  # m

# 테스트 입력
VX_CMD = 0.15
VX_DURATION = 5.0
WZ_CMD = 60.0  # deg/s
WZ_DURATION = 5.0

# -----------------------------------------------------------------------------
# Sim<->Real 보정 (최근 측정값)
#   real -> sim:
#     [sim_vx, sim_vy]^T = LIN_SCALE * M * [real_vx, real_vy]^T
#     sim_wz =  WZ_SIGN * ANG_SCALE * real_wz
# -----------------------------------------------------------------------------
USE_REAL_TO_SIM_COMP = True
LIN_SCALE = 1.0166
ANG_SCALE = 1.2360
# real(+)=CCW, sim(+)=CW 이면 -1.0, 동일 기준이면 +1.0
WZ_SIGN = -1.0
# tuned_dynamics.json의 command_transform에서 보정값 자동 동기화 (없으면 fallback 상수 사용)
AUTO_LOAD_COMP_FROM_DYNAMICS = True
DYNAMICS_JSON_PATH = "calibration/tuned_dynamics.json"

# Linear map 후보(행렬 M): sim = LIN_SCALE * M * real
LINEAR_MAP_CANDIDATES = {
    "identity": ((1.0, 0.0), (0.0, 1.0)),
    "flip_180": ((-1.0, 0.0), (0.0, -1.0)),
    "rot_cw_90": ((0.0, 1.0), (-1.0, 0.0)),
    "rot_ccw_90": ((0.0, -1.0), (1.0, 0.0)),
}
LINEAR_MAP_NAME = "rot_cw_90"  # fallback/manual
AUTO_SELECT_LINEAR_MAP = True
LINEAR_PROBE_VX = 0.10
LINEAR_PROBE_DURATION = 1.2

# Sim body frame에서 "실제 전진"이 어떤 축인지 지정
#   "x" -> body +x가 전진
#   "y" -> body +y가 전진  (현재 LeKiwi에서 사용자 관찰값)
SIM_FORWARD_AXIS = "y"
SIM_FORWARD_SIGN = +1.0

# step
SIM_DT = 1.0 / 60.0
CONTROL_HZ = 20.0

# 실측 기준
REAL_DISTANCE_CM = 75.0
REAL_ROTATION_DEG = 298.0

# Kiwi wheel layout
WHEEL_ANGLES_DEG = {
    "axle_0_joint": 90.0,    # back
    "axle_1_joint": -150.0,  # front-right
    "axle_2_joint": -30.0,   # front-left
}

JOINT_PATHS = {
    name: f"{ROBOT_PRIM_PATH}/joints/{name}"
    for name in WHEEL_ANGLES_DEG
}
APP = omni.kit.app.get_app()
ACTIVE_LINEAR_MAP_NAME = LINEAR_MAP_NAME
ACTIVE_LINEAR_MAP = LINEAR_MAP_CANDIDATES[LINEAR_MAP_NAME]


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _rot_dir_text(deg_signed):
    if deg_signed > 1e-6:
        return "CCW"
    if deg_signed < -1e-6:
        return "CW"
    return "NONE"


def _apply_linear_map(vx_real, vy_real, map_m):
    sim_vx = map_m[0][0] * float(vx_real) + map_m[0][1] * float(vy_real)
    sim_vy = map_m[1][0] * float(vx_real) + map_m[1][1] * float(vy_real)
    return sim_vx, sim_vy


def _to_float_or(default_value, value):
    try:
        return float(value)
    except Exception:
        return float(default_value)


def _resolve_dynamics_json_path():
    raw = os.path.expanduser(DYNAMICS_JSON_PATH)
    candidates = [
        raw,
        os.path.join(os.getcwd(), raw),
        os.path.join(os.getcwd(), "scripts", "lekiwi_nav_env", raw),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def maybe_load_comp_from_dynamics_json():
    """tuned_dynamics.json command_transform을 읽어 보정 상수를 동기화."""
    global LIN_SCALE, ANG_SCALE, WZ_SIGN
    global LINEAR_MAP_NAME, ACTIVE_LINEAR_MAP_NAME, ACTIVE_LINEAR_MAP

    if not AUTO_LOAD_COMP_FROM_DYNAMICS:
        return

    path = _resolve_dynamics_json_path()
    if path is None:
        print(f"[INFO] dynamics json not found; keep manual compensation constants ({DYNAMICS_JSON_PATH})")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to load dynamics json: {path} ({exc})")
        return

    command_transform = payload.get("command_transform")
    if not isinstance(command_transform, dict):
        print(f"[INFO] command_transform missing in dynamics json: {path}")
        return

    linear_map_name = str(command_transform.get("linear_map", LINEAR_MAP_NAME))
    if linear_map_name not in LINEAR_MAP_CANDIDATES:
        print(f"[WARN] unknown linear_map in dynamics json: {linear_map_name!r}; keep {LINEAR_MAP_NAME!r}")
        linear_map_name = LINEAR_MAP_NAME

    LIN_SCALE = _to_float_or(LIN_SCALE, command_transform.get("lin_scale"))
    ANG_SCALE = _to_float_or(ANG_SCALE, command_transform.get("ang_scale"))
    WZ_SIGN = _to_float_or(WZ_SIGN, command_transform.get("wz_sign"))
    LINEAR_MAP_NAME = linear_map_name
    ACTIVE_LINEAR_MAP_NAME = linear_map_name
    ACTIVE_LINEAR_MAP = LINEAR_MAP_CANDIDATES[linear_map_name]
    print(
        "[INFO] loaded compensation from dynamics json: "
        f"path={path}, lin={LIN_SCALE:.4f}, ang={ANG_SCALE:.4f}, wz_sign={WZ_SIGN:+.1f}, map={linear_map_name}"
    )


def body_to_semantic(dx_body, dy_body):
    """body(x,y) 변위를 의미축(forward,lateral-left)으로 변환."""
    if SIM_FORWARD_AXIS == "x":
        forward = SIM_FORWARD_SIGN * dx_body
        lateral_left = SIM_FORWARD_SIGN * dy_body
    elif SIM_FORWARD_AXIS == "y":
        forward = SIM_FORWARD_SIGN * dy_body
        lateral_left = -SIM_FORWARD_SIGN * dx_body
    else:
        raise ValueError(f"Unsupported SIM_FORWARD_AXIS={SIM_FORWARD_AXIS!r}")
    return forward, lateral_left


def real_to_sim_cmd(vx_real, vy_real, wz_real_deg):
    if not USE_REAL_TO_SIM_COMP:
        return float(vx_real), float(vy_real), float(wz_real_deg)
    sim_vx, sim_vy = _apply_linear_map(vx_real, vy_real, ACTIVE_LINEAR_MAP)
    sim_vx *= LIN_SCALE
    sim_vy *= LIN_SCALE
    sim_wz_deg = WZ_SIGN * ANG_SCALE * float(wz_real_deg)
    return sim_vx, sim_vy, sim_wz_deg


def kiwi_ik(vx, vy, wz_rad):
    """
    Canonical Kiwi IK (same convention as calibrate_real_robot.py / lekiwi_nav_env.py):
      omega_i = (1/r) * [cos(theta_i)*vx + sin(theta_i)*vy + L*wz]
    """
    vels = {}
    for name, theta_deg in WHEEL_ANGLES_DEG.items():
        theta = math.radians(theta_deg)
        v_tan = math.cos(theta) * vx + math.sin(theta) * vy + BASE_RADIUS * wz_rad
        vels[name] = v_tan / WHEEL_RADIUS
    return vels


def resolve_pose_prim_path():
    """Pose 측정 prim 경로 자동 결정."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage is open.")

    p = stage.GetPrimAtPath(POSE_PRIM_PATH_OVERRIDE)
    if p.IsValid():
        return POSE_PRIM_PATH_OVERRIDE

    robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot.IsValid():
        raise RuntimeError(f"Robot prim not found: {ROBOT_PRIM_PATH}")

    if robot.HasAPI(UsdPhysics.ArticulationRootAPI):
        return ROBOT_PRIM_PATH

    for prim in Usd.PrimRange(robot):  # noqa: F821 (Usd is available in Script Editor)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())

    # fallback
    print(f"[WARN] ArticulationRoot prim not found under {ROBOT_PRIM_PATH}; fallback to robot prim.")
    return ROBOT_PRIM_PATH


def get_pose_xy_yaw(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Pose prim not found: {prim_path}")

    # NOTE:
    #   ComputeLocalToWorldTransform(0)은 authored time=0 값만 읽어
    #   physics runtime pose를 놓칠 수 있다.
    #   Script Editor에서는 omni.usd live world transform을 우선 사용한다.
    tf = None
    try:
        tf = omni.usd.get_world_transform_matrix(prim)
    except Exception:
        tf = None

    if tf is None:
        xform = UsdGeom.Xformable(prim)
        tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    pos = tf.ExtractTranslation()
    rot = tf.ExtractRotationMatrix()
    yaw = math.atan2(float(rot[1][0]), float(rot[0][0]))
    return float(pos[0]), float(pos[1]), float(yaw)


def get_wheel_joint_pos_deg():
    """Wheel joint state(angular position, deg) 읽기. 없으면 nan."""
    stage = omni.usd.get_context().get_stage()
    out = {}
    for name, path in JOINT_PATHS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            out[name] = float("nan")
            continue
        attr = prim.GetAttribute("state:angular:physics:position")
        v = attr.Get() if attr else None
        out[name] = float(v) if v is not None else float("nan")
    return out


def set_wheel_velocities(wheel_vels):
    stage = omni.usd.get_context().get_stage()
    for name, omega_rad in wheel_vels.items():
        prim = stage.GetPrimAtPath(JOINT_PATHS[name])
        if not prim.IsValid():
            print(f"[WARN] joint not found: {JOINT_PATHS[name]}")
            continue
        attr = prim.GetAttribute("drive:angular:physics:targetVelocity")
        if not attr:
            print(f"[WARN] targetVelocity attr missing: {JOINT_PATHS[name]}")
            continue
        attr.Set(float(math.degrees(omega_rad)))  # USD expects deg/s


def stop_wheels():
    stage = omni.usd.get_context().get_stage()
    for name in JOINT_PATHS:
        prim = stage.GetPrimAtPath(JOINT_PATHS[name])
        if prim.IsValid():
            attr = prim.GetAttribute("drive:angular:physics:targetVelocity")
            if attr:
                attr.Set(0.0)


async def step_sim_frames(num_frames, control_cb=None):
    """Script Editor 안전 모드: app.next_update_async()로 물리 스텝."""
    if num_frames <= 0:
        return
    for i in range(num_frames):
        if control_cb is not None:
            control_cb(i)
        await APP.next_update_async()


async def run_test(label, vx, vy, wz_deg, duration, pose_prim_path):
    print("")
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    vx_sim, vy_sim, wz_sim_deg = real_to_sim_cmd(vx, vy, wz_deg)
    wz_rad = math.radians(wz_sim_deg)
    wheel_vels = kiwi_ik(vx_sim, vy_sim, wz_rad)

    print(f"  cmd(real): vx={vx}, vy={vy}, wz={wz_deg} deg/s, duration={duration}s")
    print(f"  cmd(sim) : vx={vx_sim:.4f}, vy={vy_sim:.4f}, wz={wz_sim_deg:.4f} deg/s")
    for n, w in wheel_vels.items():
        print(f"    {n}: {w:.3f} rad/s ({math.degrees(w):.1f} deg/s)")

    x0, y0, yaw0 = get_pose_xy_yaw(pose_prim_path)
    wheel_pos0 = get_wheel_joint_pos_deg()
    print(f"  start: x={x0:.4f} y={y0:.4f} yaw={math.degrees(yaw0):.2f} deg")

    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    if not was_playing:
        timeline.play()
        await APP.next_update_async()

    num_steps = int(duration / SIM_DT)
    ctrl_every = max(int(1.0 / (CONTROL_HZ * SIM_DT)), 1)
    set_wheel_velocities(wheel_vels)

    # yaw 누적(unwrap)으로 총 회전량 계산.
    # 끝점 yaw 차이만 쓰면 |delta|>180deg 구간에서 값이 작게 잘릴 수 있다.
    yaw_prev = yaw0
    yaw_total = 0.0

    def _cb(i):
        if i % ctrl_every == 0:
            set_wheel_velocities(wheel_vels)
        nonlocal yaw_prev, yaw_total
        _, _, yaw_now = get_pose_xy_yaw(pose_prim_path)
        yaw_total += wrap_pi(yaw_now - yaw_prev)
        yaw_prev = yaw_now

    await step_sim_frames(num_steps, control_cb=_cb)

    stop_wheels()
    await step_sim_frames(10)

    if not was_playing:
        timeline.pause()
        await APP.next_update_async()

    x1, y1, yaw1 = get_pose_xy_yaw(pose_prim_path)
    wheel_pos1 = get_wheel_joint_pos_deg()
    print(f"  end:   x={x1:.4f} y={y1:.4f} yaw={math.degrees(yaw1):.2f} deg")

    print("  wheel joint delta (deg):")
    for name in JOINT_PATHS:
        a = wheel_pos0.get(name, float("nan"))
        b = wheel_pos1.get(name, float("nan"))
        if math.isfinite(a) and math.isfinite(b):
            print(f"    {name}: {b - a:+.2f}")
        else:
            print(f"    {name}: n/a")

    dx = x1 - x0
    dy = y1 - y0
    dist_cm = math.hypot(dx, dy) * 100.0

    # 시작 body frame 기준 변위 분해(직진/횡이동 확인용)
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    dx_body = c0 * dx + s0 * dy
    dy_body = -s0 * dx + c0 * dy
    d_fwd, d_lat_left = body_to_semantic(dx_body, dy_body)

    rot_deg_shortest_signed = math.degrees(wrap_pi(yaw1 - yaw0))
    rot_deg_total_signed = math.degrees(yaw_total)
    rot_deg_shortest = abs(rot_deg_shortest_signed)
    rot_deg_total = abs(rot_deg_total_signed)

    print(f"  body-frame delta: x={dx_body * 100.0:.2f} cm, y={dy_body * 100.0:.2f} cm")
    print(f"  semantic delta: forward={d_fwd * 100.0:.2f} cm, left={d_lat_left * 100.0:.2f} cm")
    cmd_norm_real = math.hypot(vx, vy)
    if cmd_norm_real > 1e-8:
        along_real = (d_fwd * vx + d_lat_left * vy) / cmd_norm_real
        cross_real = (-d_fwd * vy + d_lat_left * vx) / cmd_norm_real
        print(f"  real-cmd frame: along={along_real * 100.0:.2f} cm, cross={cross_real * 100.0:.2f} cm")
    cmd_norm_sim = math.hypot(vx_sim, vy_sim)
    if cmd_norm_sim > 1e-8:
        along_sim = (d_fwd * vx_sim + d_lat_left * vy_sim) / cmd_norm_sim
        cross_sim = (-d_fwd * vy_sim + d_lat_left * vx_sim) / cmd_norm_sim
        print(f"  sim-cmd  frame: along={along_sim * 100.0:.2f} cm, cross={cross_sim * 100.0:.2f} cm")
    print(
        "  result: distance={:.2f} cm, rotation_total={:.2f} deg "
        "(endpoint_shortest={:.2f} deg)".format(dist_cm, rot_deg_total, rot_deg_shortest)
    )
    print(
        "  rotation sign: total={:+.2f} deg [{}], endpoint={:+.2f} deg [{}]".format(
            rot_deg_total_signed,
            _rot_dir_text(rot_deg_total_signed),
            rot_deg_shortest_signed,
            _rot_dir_text(rot_deg_shortest_signed),
        )
    )
    return dist_cm, rot_deg_total_signed


async def probe_linear_map(map_name, map_m, pose_prim_path):
    """real vx+ 직진 기준으로 map 후보를 평가."""
    vx_sim, vy_sim = _apply_linear_map(LINEAR_PROBE_VX, 0.0, map_m)
    vx_sim *= LIN_SCALE
    vy_sim *= LIN_SCALE
    wheel_vels = kiwi_ik(vx_sim, vy_sim, 0.0)

    x0, y0, yaw0 = get_pose_xy_yaw(pose_prim_path)
    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    if not was_playing:
        timeline.play()
        await APP.next_update_async()

    num_steps = int(LINEAR_PROBE_DURATION / SIM_DT)
    ctrl_every = max(int(1.0 / (CONTROL_HZ * SIM_DT)), 1)
    set_wheel_velocities(wheel_vels)

    def _cb(i):
        if i % ctrl_every == 0:
            set_wheel_velocities(wheel_vels)

    await step_sim_frames(num_steps, control_cb=_cb)
    stop_wheels()
    await step_sim_frames(10)
    if not was_playing:
        timeline.pause()
        await APP.next_update_async()

    x1, y1, _ = get_pose_xy_yaw(pose_prim_path)
    dx = x1 - x0
    dy = y1 - y0
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    dx_body = c0 * dx + s0 * dy
    dy_body = -s0 * dx + c0 * dy
    d_fwd, d_lat_left = body_to_semantic(dx_body, dy_body)
    along_cm = d_fwd * 100.0
    cross_cm = d_lat_left * 100.0
    score = along_cm - 2.0 * abs(cross_cm)
    if along_cm < 0.0:
        score -= 1000.0
    print(
        f"  [probe] {map_name:<10s} -> along={along_cm:+.2f}cm, "
        f"cross={cross_cm:+.2f}cm, score={score:+.2f}"
    )
    return score


async def auto_select_linear_map(pose_prim_path):
    global ACTIVE_LINEAR_MAP_NAME, ACTIVE_LINEAR_MAP
    if not AUTO_SELECT_LINEAR_MAP:
        ACTIVE_LINEAR_MAP_NAME = LINEAR_MAP_NAME
        ACTIVE_LINEAR_MAP = LINEAR_MAP_CANDIDATES[LINEAR_MAP_NAME]
        return

    print("")
    print("  === AUTO SELECT LINEAR MAP ===")
    print(
        f"  criterion: real vx>0 should move +forward with minimal lateral drift "
        f"(forward_axis={SIM_FORWARD_AXIS}{'+' if SIM_FORWARD_SIGN > 0 else '-'}, "
        f"probe_vx={LINEAR_PROBE_VX}, T={LINEAR_PROBE_DURATION}s)"
    )
    best_name = None
    best_score = -1e18
    for name, mat in LINEAR_MAP_CANDIDATES.items():
        score = await probe_linear_map(name, mat, pose_prim_path)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        best_name = LINEAR_MAP_NAME
    ACTIVE_LINEAR_MAP_NAME = best_name
    ACTIVE_LINEAR_MAP = LINEAR_MAP_CANDIDATES[best_name]
    print(f"  selected linear map: {ACTIVE_LINEAR_MAP_NAME} (score={best_score:+.2f})")


async def _run_all_tests():
    print("=" * 60)
    print("  LeKiwi Sim-Real Calibration Test")
    print("=" * 60)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Open a stage first.")

    maybe_load_comp_from_dynamics_json()
    pose_prim_path = resolve_pose_prim_path()
    await auto_select_linear_map(pose_prim_path)
    print(f"  robot prim : {ROBOT_PRIM_PATH}")
    print(f"  pose prim  : {pose_prim_path}")
    print(f"  WHEEL_RADIUS = {WHEEL_RADIUS} m")
    print(f"  BASE_RADIUS  = {BASE_RADIUS} m")
    print(
        f"  real->sim compensation: enabled={USE_REAL_TO_SIM_COMP}, "
        f"lin={LIN_SCALE:.4f}, ang={ANG_SCALE:.4f}, wz_sign={WZ_SIGN:+.1f}, "
        f"linear_map={ACTIVE_LINEAR_MAP_NAME}, fwd_axis={SIM_FORWARD_AXIS}{'+' if SIM_FORWARD_SIGN > 0 else '-'}"
    )
    print(f"  real reference: linear={REAL_DISTANCE_CM} cm, angular={REAL_ROTATION_DEG} deg")

    # Test 1: linear
    sim_dist_cm, _ = await run_test(
        "TEST 1: Linear (vx only)",
        vx=VX_CMD,
        vy=0.0,
        wz_deg=0.0,
        duration=VX_DURATION,
        pose_prim_path=pose_prim_path,
    )

    # Test 2: angular
    _, sim_rot_deg_signed = await run_test(
        "TEST 2: Angular (wz only)",
        vx=0.0,
        vy=0.0,
        wz_deg=WZ_CMD,
        duration=WZ_DURATION,
        pose_prim_path=pose_prim_path,
    )
    sim_rot_deg = abs(sim_rot_deg_signed)

    # 결과
    lr = REAL_DISTANCE_CM / sim_dist_cm if sim_dist_cm > 1e-2 else float("inf")
    ar = REAL_ROTATION_DEG / sim_rot_deg if sim_rot_deg > 1e-2 else float("inf")

    print("")
    print("=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    print("  {:20s}  {:>10s}  {:>10s}  {:>12s}".format("", "SIM", "REAL", "ratio(R/S)"))
    print("  " + "-" * 55)
    print("  {:20s}  {:10.2f}  {:10.2f}  {:12.4f}".format("linear (cm)", sim_dist_cm, REAL_DISTANCE_CM, lr))
    print("  {:20s}  {:10.2f}  {:10.2f}  {:12.4f}".format("angular (deg)", sim_rot_deg, REAL_ROTATION_DEG, ar))

    print("")
    print("  === VERDICT ===")
    for tag, ratio in [("linear", lr), ("angular", ar)]:
        if 0.95 < ratio < 1.05:
            print(f"  {tag}: MATCH (error {abs(1.0 - ratio) * 100:.1f}%)")
        else:
            print(f"  {tag}: MISMATCH (R/S={ratio:.4f})")

    print("")
    if abs(REAL_ROTATION_DEG) > 1e-6 and abs(sim_rot_deg_signed) > 1e-6:
        same_sign = (REAL_ROTATION_DEG * sim_rot_deg_signed) > 0.0
        print(
            "  rotation direction: real={} / sim={} -> {}".format(
                _rot_dir_text(REAL_ROTATION_DEG),
                _rot_dir_text(sim_rot_deg_signed),
                "MATCH" if same_sign else "MISMATCH",
            )
        )
        print("")
    print("  === ACTION SCALING ===")
    print("  ratio(R/S) = real_output / sim_output")
    print("  sim -> real command match:")
    print(f"    real_vx = sim_vx / {lr:.4f}")
    print(f"    real_vy = sim_vy / {lr:.4f}")
    print(f"    real_wz = sim_wz / {ar:.4f}")
    print("  real -> sim command match:")
    print(f"    sim_vx  = real_vx * {lr:.4f}")
    print(f"    sim_vy  = real_vy * {lr:.4f}")
    print(f"    sim_wz  = real_wz * {ar:.4f}")


def _launch_async():
    task_key = "__LEKIWI_SIM_REAL_CAL_TASK__"
    old_task = globals().get(task_key)
    if old_task is not None and not old_task.done():
        old_task.cancel()
        print("[INFO] previous calibration task cancelled.")

    task = asyncio.ensure_future(_run_all_tests())
    globals()[task_key] = task

    def _done_cb(done_task):
        try:
            done_task.result()
        except asyncio.CancelledError:
            print("[INFO] calibration task cancelled.")
        except Exception as exc:
            print(f"[ERROR] calibration task failed: {exc}")
            import traceback
            traceback.print_exc()

    task.add_done_callback(_done_cb)
    print("[INFO] calibration task started (async).")


_launch_async()
