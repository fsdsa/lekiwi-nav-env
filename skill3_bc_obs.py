from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

S3_BC_OBS_BASE_DIM = 36
S3_BC_RELEASE_FLAG_NAME = "release_phase_flag"
S3_BC_RETRACT_FLAG_NAME = "retract_started_flag"
S3_BC_OBS_RELEASE_DIM = S3_BC_OBS_BASE_DIM + 2
S3_BC_OBS_EE23_DIM = 23
S3_BC_OBS_MOTION24_DIM = 24
S3_BC_EE23_NAMES = (
    "arm0",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "grip",
    "armvel0",
    "armvel1",
    "armvel2",
    "armvel3",
    "armvel4",
    "grip_vel",
    "base_vx",
    "base_vy",
    "base_wz",
    "ee_to_dest_body_x",
    "ee_to_dest_body_y",
    "ee_to_dest_body_z",
    "dest_rel_body_x",
    "dest_rel_body_y",
    "phase_a_flag",
    "release_phase_flag",
    "retract_started_flag",
)
S3_BC_MOTION24_NAMES = (
    "arm0",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "grip",
    "armvel0",
    "armvel1",
    "armvel2",
    "armvel3",
    "armvel4",
    "grip_vel",
    "base_vx",
    "base_vy",
    "base_wz",
    "ee_to_dest_body_x",
    "ee_to_dest_body_y",
    "dest_rel_body_x",
    "dest_rel_body_y",
    "dest_rel_body_z",
    "ee_z",
    "phase_a_flag",
    "release_phase_flag",
    "retract_started_flag",
)
S3_BC_EXTRA_NAMES = (
    "source_rel_body_x",
    "source_rel_body_y",
    "source_rel_body_z",
    "source_to_dest_body_x",
    "source_to_dest_body_y",
    "source_to_dest_body_z",
    "source_to_dest_xy",
    "source_height",
    "source_upright",
    "source_holding",
    "source_up_body_x",
    "source_up_body_y",
    "source_up_body_z",
    "source_to_gripper_body_x",
    "source_to_gripper_body_y",
    "source_to_gripper_body_z",
)
S3_BC_OBS_EXTENDED_DIM = S3_BC_OBS_BASE_DIM + len(S3_BC_EXTRA_NAMES)
S3_BC_PLACE_FLAG_NAME = "place_open_phase_flag"
S3_BC_OBS_PHASED_DIM = S3_BC_OBS_EXTENDED_DIM + 1
# 53D phased + release_phase_flag + retract_started_flag
S3_BC_OBS_PHASED55_DIM = S3_BC_OBS_PHASED_DIM + 2

S3_PLACE_FLAG_DIST_MAX = 0.18
S3_PLACE_FLAG_HEIGHT_MIN = 0.028
S3_PLACE_FLAG_HEIGHT_MAX = 0.055
S3_PLACE_FLAG_UPRIGHT_MIN = 0.85
S3_PLACE_FLAG_GRIP_MIN = 0.90


def _get_gripper_body_idx(env) -> int:
    idx = getattr(env, "_bc_obs_gripper_body_idx", None)
    if idx is not None:
        return int(idx)
    try:
        body_ids, _ = env.robot.find_bodies(["Moving_Jaw_08d_v1"])
        idx = int(body_ids[0])
    except (IndexError, RuntimeError, AttributeError):
        idx = 0
    env._bc_obs_gripper_body_idx = idx
    return idx


def _get_object_quat_w(env, num_envs: int, device: torch.device) -> torch.Tensor:
    if getattr(env, "_multi_object", False) and len(getattr(env, "object_rigids", [])) > 0:
        out = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        out[:, 0] = 1.0
        for oi, rigid in enumerate(env.object_rigids):
            mask = env.active_object_idx == oi
            if not mask.any():
                continue
            ids = mask.nonzero(as_tuple=False).squeeze(-1)
            out[ids] = rigid.data.root_quat_w[ids]
        return out
    if getattr(env, "object_rigid", None) is not None:
        return env.object_rigid.data.root_quat_w
    out = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    out[:, 0] = 1.0
    return out


def ee_world_pos(env) -> torch.Tensor:
    device = env.robot.data.root_pos_w.device
    num_envs = env.robot.data.root_pos_w.shape[0]
    jaw_idx = getattr(env, "_fixed_jaw_body_idx", -1)
    if jaw_idx is None or int(jaw_idx) < 0:
        try:
            fixed_jaw_ids, _ = env.robot.find_bodies(["Wrist_Roll_08c_v1"])
            jaw_idx = int(fixed_jaw_ids[0])
            env._fixed_jaw_body_idx = jaw_idx
        except (IndexError, RuntimeError, AttributeError):
            jaw_idx = -1
    if jaw_idx >= 0:
        wrist_pos = env.robot.data.body_pos_w[:, jaw_idx, :]
        wrist_quat = env.robot.data.body_quat_w[:, jaw_idx, :]
        ee_offset = getattr(env, "_ee_local_offset", None)
        if ee_offset is None:
            ee_offset = torch.zeros(1, 3, dtype=torch.float32, device=device)
        return wrist_pos + quat_apply(wrist_quat, ee_offset.expand(num_envs, -1))
    return env.robot.data.root_pos_w


def source_uprightness(env) -> torch.Tensor:
    device = env.robot.data.root_pos_w.device
    num_envs = env.robot.data.root_pos_w.shape[0]
    obj_quat = _get_object_quat_w(env, num_envs, device)
    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).expand(num_envs, -1)
    obj_up_world = quat_apply(obj_quat, world_up)
    return obj_up_world[:, 2].clamp(0.0, 1.0)


def compute_place_open_trigger(env, policy_obs: torch.Tensor, phase_a_flag) -> torch.Tensor:
    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(0)
    device = policy_obs.device
    num_envs = policy_obs.shape[0]

    if not torch.is_tensor(phase_a_flag):
        phase_a_flag = torch.full((num_envs, 1), float(phase_a_flag), dtype=torch.float32, device=device)
    else:
        phase_a_flag = phase_a_flag.to(device=device, dtype=torch.float32)
        if phase_a_flag.dim() == 0:
            phase_a_flag = phase_a_flag.view(1, 1).expand(num_envs, -1)
        elif phase_a_flag.dim() == 1:
            phase_a_flag = phase_a_flag.unsqueeze(-1)

    if policy_obs.shape[-1] not in (29, 30):
        raise ValueError(f"Unsupported policy_obs dim for place-open trigger: {policy_obs.shape[-1]}")

    src_pos = env.object_pos_w
    dst_pos = env.dest_object_pos_w
    src_dst_xy = torch.norm(src_pos[:, :2] - dst_pos[:, :2], dim=-1, keepdim=True)
    src_h = (src_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
    src_upright = source_uprightness(env).unsqueeze(-1)
    grip_pos = policy_obs[:, 5:6]
    phase_b = phase_a_flag < 0.5

    return (
        phase_b
        & (src_dst_xy <= S3_PLACE_FLAG_DIST_MAX)
        & (src_h >= S3_PLACE_FLAG_HEIGHT_MIN)
        & (src_h <= S3_PLACE_FLAG_HEIGHT_MAX)
        & (src_upright >= S3_PLACE_FLAG_UPRIGHT_MIN)
        & (grip_pos >= S3_PLACE_FLAG_GRIP_MIN)
    ).squeeze(-1)


def build_s3_bc_obs(
    env,
    policy_obs: torch.Tensor,
    init_pose6: torch.Tensor,
    phase_a_flag,
    obs_dim: int,
    place_open_flag=None,
    release_phase_flag=None,
    retract_started_flag=None,
    release_xy_thresh: float = 0.12,
    release_ee_z_thresh: float = 0.10,
    retract_grip_thresh: float = 0.55,
) -> torch.Tensor:
    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(0)
    device = policy_obs.device
    num_envs = policy_obs.shape[0]

    if init_pose6.dim() == 1:
        init_pose6 = init_pose6.unsqueeze(0).expand(num_envs, -1)
    if not torch.is_tensor(phase_a_flag):
        phase_a_flag = torch.full((num_envs, 1), float(phase_a_flag), dtype=torch.float32, device=device)
    else:
        phase_a_flag = phase_a_flag.to(device=device, dtype=torch.float32)
        if phase_a_flag.dim() == 0:
            phase_a_flag = phase_a_flag.view(1, 1).expand(num_envs, -1)
        elif phase_a_flag.dim() == 1:
            phase_a_flag = phase_a_flag.unsqueeze(-1)

    if policy_obs.shape[-1] == 29:
        s3_obs29 = policy_obs
    elif policy_obs.shape[-1] == 30:
        robot_pos = env.robot.data.root_pos_w
        robot_quat = env.robot.data.root_quat_w
        dest_pos = env.dest_object_pos_w
        dest_rel_body = quat_apply_inverse(robot_quat, dest_pos - robot_pos)
        contact_force = env._contact_force_per_env().unsqueeze(-1)
        s3_obs29 = torch.cat([
            policy_obs[:, 0:21],
            dest_rel_body,
            contact_force,
            policy_obs[:, 26:29],
            policy_obs[:, 29:30],
        ], dim=-1)
    else:
        raise ValueError(f"Unsupported policy_obs dim for Skill-3 BC obs build: {policy_obs.shape[-1]}")

    s3_obs = torch.cat([s3_obs29, init_pose6.to(device=device, dtype=torch.float32), phase_a_flag], dim=-1)
    if obs_dim == S3_BC_OBS_BASE_DIM:
        return s3_obs
    if obs_dim == S3_BC_OBS_RELEASE_DIM:
        src_pos = env.object_pos_w
        dst_pos = env.dest_object_pos_w
        grip_pos = policy_obs[:, 5:6]
        arm1 = policy_obs[:, 1:2]
        src_dst_xy = torch.norm(src_pos[:, :2] - dst_pos[:, :2], dim=-1, keepdim=True)
        src_h = (src_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
        phase_b = phase_a_flag < 0.5

        if hasattr(env, "object_grasped"):
            holding = env.object_grasped.float().unsqueeze(-1)
        else:
            holding = (env._contact_force_per_env() > float(env.cfg.grasp_contact_threshold)).float().unsqueeze(-1)

        if release_phase_flag is None:
            release_phase_flag = (
                phase_b
                & (arm1 >= 2.0)
                & (src_dst_xy <= 0.18)
            ).float().view(-1, 1)
        elif not torch.is_tensor(release_phase_flag):
            release_phase_flag = torch.full((num_envs, 1), float(release_phase_flag), dtype=torch.float32, device=device)
        else:
            release_phase_flag = release_phase_flag.to(device=device, dtype=torch.float32).view(-1, 1)

        if retract_started_flag is None:
            retract_started_flag = (
                (release_phase_flag.view(-1) > 0.5)
                & (grip_pos.view(-1) >= 0.55)
                & (src_dst_xy.view(-1) <= 0.18)
                & (src_h.view(-1) <= 0.055)
                & (holding.view(-1) < 0.5)
            ).float().view(-1, 1)
        elif not torch.is_tensor(retract_started_flag):
            retract_started_flag = torch.full((num_envs, 1), float(retract_started_flag), dtype=torch.float32, device=device)
        else:
            retract_started_flag = retract_started_flag.to(device=device, dtype=torch.float32).view(-1, 1)

        return torch.cat([s3_obs, release_phase_flag, retract_started_flag], dim=-1)

    if obs_dim not in (S3_BC_OBS_EXTENDED_DIM, S3_BC_OBS_PHASED_DIM, S3_BC_OBS_PHASED55_DIM):
        raise ValueError(
            f"Unsupported Skill-3 BC obs_dim={obs_dim}; expected "
            f"{S3_BC_OBS_BASE_DIM}, {S3_BC_OBS_RELEASE_DIM}, {S3_BC_OBS_EXTENDED_DIM}, "
            f"{S3_BC_OBS_PHASED_DIM}, or {S3_BC_OBS_PHASED55_DIM}"
        )

    robot_pos = env.robot.data.root_pos_w
    robot_quat = env.robot.data.root_quat_w
    src_pos = env.object_pos_w
    dst_pos = env.dest_object_pos_w
    src_rel_body = quat_apply_inverse(robot_quat, src_pos - robot_pos)
    src_to_dest_body = quat_apply_inverse(robot_quat, src_pos - dst_pos)
    src_dst_xy = torch.norm(src_pos[:, :2] - dst_pos[:, :2], dim=-1, keepdim=True)
    src_h = (src_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
    src_upright = source_uprightness(env).unsqueeze(-1)
    if hasattr(env, "object_grasped"):
        holding = env.object_grasped.float().unsqueeze(-1)
    else:
        holding = (env._contact_force_per_env() > 0.0).float().unsqueeze(-1)

    obj_quat = _get_object_quat_w(env, num_envs, device)
    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device).expand(num_envs, -1)
    obj_up_world = quat_apply(obj_quat, world_up)
    obj_up_body = quat_apply_inverse(robot_quat, obj_up_world)

    grip_body_idx = _get_gripper_body_idx(env)
    grip_pos_w = env.robot.data.body_pos_w[:, grip_body_idx]
    src_to_gripper_body = quat_apply_inverse(robot_quat, src_pos - grip_pos_w)

    extra = torch.cat([
        src_rel_body,
        src_to_dest_body,
        src_dst_xy,
        src_h,
        src_upright,
        holding,
        obj_up_body,
        src_to_gripper_body,
    ], dim=-1)
    out = torch.cat([s3_obs, extra], dim=-1)
    if obs_dim == S3_BC_OBS_EXTENDED_DIM:
        return out

    if place_open_flag is None:
        place_open_flag = compute_place_open_trigger(env, policy_obs, phase_a_flag).float()
    elif not torch.is_tensor(place_open_flag):
        place_open_flag = torch.full((num_envs,), float(place_open_flag), dtype=torch.float32, device=device)
    else:
        place_open_flag = place_open_flag.to(device=device, dtype=torch.float32).view(-1)

    out_53 = torch.cat([out, place_open_flag.unsqueeze(-1)], dim=-1)
    if obs_dim == S3_BC_OBS_PHASED_DIM:
        return out_53

    # 55D = 53D + release_phase_flag + retract_started_flag (build_s3_ee23_obs와 동일 로직)
    # 학습 데이터의 obs[53], obs[54]는 ee 기반으로 매 step 새로 계산되었으므로
    # eval에서도 동일하게 매 step 새로 계산 (latch 사용 X, deploy/train 일치)
    ee_pos = ee_world_pos(env)
    ee_to_dest_body = quat_apply_inverse(robot_quat, dst_pos - ee_pos)
    ee_z = (ee_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
    ee_xy = torch.norm(ee_to_dest_body[:, :2], dim=-1, keepdim=True)
    grip_pos = policy_obs[:, 5:6]
    phase_b = phase_a_flag < 0.5

    rel_flag = (
        phase_b
        & (ee_xy <= release_xy_thresh)
        & (ee_z <= release_ee_z_thresh)
    ).float().view(-1, 1)
    ret_flag = (
        (rel_flag > 0.5)
        & (grip_pos >= retract_grip_thresh)
    ).float().view(-1, 1)

    return torch.cat([out_53, rel_flag, ret_flag], dim=-1)


def build_s3_motion24_obs(
    env,
    policy_obs: torch.Tensor,
    phase_a_flag,
    release_phase_flag=None,
    retract_started_flag=None,
    release_xy_thresh: float = 0.12,
    release_ee_z_thresh: float = 0.10,
    retract_grip_thresh: float = 0.55,
) -> torch.Tensor:
    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(0)
    device = policy_obs.device
    num_envs = policy_obs.shape[0]

    if not torch.is_tensor(phase_a_flag):
        phase_a_flag = torch.full((num_envs, 1), float(phase_a_flag), dtype=torch.float32, device=device)
    else:
        phase_a_flag = phase_a_flag.to(device=device, dtype=torch.float32)
        if phase_a_flag.dim() == 0:
            phase_a_flag = phase_a_flag.view(1, 1).expand(num_envs, -1)
        elif phase_a_flag.dim() == 1:
            phase_a_flag = phase_a_flag.unsqueeze(-1)

    if policy_obs.shape[-1] == 29:
        dest_rel_body = policy_obs[:, 21:24]
    elif policy_obs.shape[-1] == 30:
        robot_pos = env.robot.data.root_pos_w
        robot_quat = env.robot.data.root_quat_w
        dest_pos = env.dest_object_pos_w
        dest_rel_body = quat_apply_inverse(robot_quat, dest_pos - robot_pos)
    else:
        raise ValueError(f"Unsupported policy_obs dim for Skill-3 motion24 obs build: {policy_obs.shape[-1]}")

    root_quat = env.robot.data.root_quat_w
    dest_pos = env.dest_object_pos_w
    ee_pos = ee_world_pos(env)
    ee_to_dest_body = quat_apply_inverse(root_quat, dest_pos - ee_pos)
    ee_z = (ee_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)

    grip_pos = policy_obs[:, 5:6]
    phase_b = phase_a_flag < 0.5
    ee_xy = torch.norm(ee_to_dest_body[:, :2], dim=-1, keepdim=True)

    if release_phase_flag is None:
        release_phase_flag = (
            phase_b
            & (ee_xy <= release_xy_thresh)
            & (ee_z <= release_ee_z_thresh)
        ).float()
    elif not torch.is_tensor(release_phase_flag):
        release_phase_flag = torch.full((num_envs, 1), float(release_phase_flag), dtype=torch.float32, device=device)
    else:
        release_phase_flag = release_phase_flag.to(device=device, dtype=torch.float32).view(-1, 1)

    if retract_started_flag is None:
        retract_started_flag = (
            (release_phase_flag > 0.5)
            & (grip_pos >= retract_grip_thresh)
        ).float()
    elif not torch.is_tensor(retract_started_flag):
        retract_started_flag = torch.full((num_envs, 1), float(retract_started_flag), dtype=torch.float32, device=device)
    else:
        retract_started_flag = retract_started_flag.to(device=device, dtype=torch.float32).view(-1, 1)

    motion24 = torch.cat([
        policy_obs[:, 0:6],
        policy_obs[:, 15:21],
        policy_obs[:, 6:9],
        ee_to_dest_body[:, :2],
        dest_rel_body,
        ee_z,
        phase_a_flag,
        release_phase_flag,
        retract_started_flag,
    ], dim=-1)
    if motion24.shape[-1] != S3_BC_OBS_MOTION24_DIM:
        raise RuntimeError(f"Skill-3 motion24 obs build produced {motion24.shape[-1]} dims, expected {S3_BC_OBS_MOTION24_DIM}")
    return motion24


def build_s3_ee23_obs(
    env,
    policy_obs: torch.Tensor,
    phase_a_flag,
    release_phase_flag=None,
    retract_started_flag=None,
    release_xy_thresh: float = 0.12,
    release_ee_z_thresh: float = 0.10,
    retract_grip_thresh: float = 0.55,
) -> torch.Tensor:
    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(0)
    device = policy_obs.device
    num_envs = policy_obs.shape[0]

    if not torch.is_tensor(phase_a_flag):
        phase_a_flag = torch.full((num_envs, 1), float(phase_a_flag), dtype=torch.float32, device=device)
    else:
        phase_a_flag = phase_a_flag.to(device=device, dtype=torch.float32)
        if phase_a_flag.dim() == 0:
            phase_a_flag = phase_a_flag.view(1, 1).expand(num_envs, -1)
        elif phase_a_flag.dim() == 1:
            phase_a_flag = phase_a_flag.unsqueeze(-1)

    if policy_obs.shape[-1] == 29:
        dest_rel_body = policy_obs[:, 21:24]
    elif policy_obs.shape[-1] == 30:
        robot_pos = env.robot.data.root_pos_w
        robot_quat = env.robot.data.root_quat_w
        dest_pos = env.dest_object_pos_w
        dest_rel_body = quat_apply_inverse(robot_quat, dest_pos - robot_pos)
    else:
        raise ValueError(f"Unsupported policy_obs dim for Skill-3 ee23 obs build: {policy_obs.shape[-1]}")

    root_quat = env.robot.data.root_quat_w
    dest_pos = env.dest_object_pos_w
    ee_pos = ee_world_pos(env)
    ee_to_dest_body = quat_apply_inverse(root_quat, dest_pos - ee_pos)
    ee_z = (ee_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)

    grip_pos = policy_obs[:, 5:6]
    phase_b = phase_a_flag < 0.5
    ee_xy = torch.norm(ee_to_dest_body[:, :2], dim=-1, keepdim=True)

    if release_phase_flag is None:
        release_phase_flag = (
            phase_b
            & (ee_xy <= release_xy_thresh)
            & (ee_z <= release_ee_z_thresh)
        ).float()
    elif not torch.is_tensor(release_phase_flag):
        release_phase_flag = torch.full((num_envs, 1), float(release_phase_flag), dtype=torch.float32, device=device)
    else:
        release_phase_flag = release_phase_flag.to(device=device, dtype=torch.float32).view(-1, 1)

    if retract_started_flag is None:
        retract_started_flag = (
            (release_phase_flag > 0.5)
            & (grip_pos >= retract_grip_thresh)
        ).float()
    elif not torch.is_tensor(retract_started_flag):
        retract_started_flag = torch.full((num_envs, 1), float(retract_started_flag), dtype=torch.float32, device=device)
    else:
        retract_started_flag = retract_started_flag.to(device=device, dtype=torch.float32).view(-1, 1)

    ee23 = torch.cat([
        policy_obs[:, 0:6],
        policy_obs[:, 15:21],
        policy_obs[:, 6:9],
        ee_to_dest_body,
        dest_rel_body[:, :2],
        phase_a_flag,
        release_phase_flag,
        retract_started_flag,
    ], dim=-1)
    if ee23.shape[-1] != S3_BC_OBS_EE23_DIM:
        raise RuntimeError(f"Skill-3 ee23 obs build produced {ee23.shape[-1]} dims, expected {S3_BC_OBS_EE23_DIM}")
    return ee23
