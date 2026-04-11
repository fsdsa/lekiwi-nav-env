from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


SCENE_PRIM_PATH = "/World/ProcTHORScene"
DEFAULT_SCENE_INSTALL_DIR = Path("~/molmospaces/assets/usd").expanduser()
FLOOR_OBSTACLE_MIN_TOP_Z = 0.05
FLOOR_OBSTACLE_MAX_BOTTOM_Z = 0.45


@dataclass(frozen=True)
class SceneSpawnCfg:
    min_robot_dist: float
    max_robot_dist: float
    clearance_radius: float


@dataclass(frozen=True)
class ScenePreset:
    default_robot_xy: tuple[float, float]
    default_robot_yaw_rad: float
    robot_clearance_radius: float
    source_spawn: SceneSpawnCfg
    dest_spawn: SceneSpawnCfg
    source_dest_min_separation: float
    support_floor_prim_path: str
    source_rest_z: float = 0.033
    dest_rest_z: float = 0.0


@dataclass(frozen=True)
class SceneTaskLayout:
    robot_xy: tuple[float, float]
    robot_yaw_rad: float
    source_xy: tuple[float, float]
    source_yaw_rad: float
    dest_xy: tuple[float, float]
    dest_yaw_rad: float
    floor_z: float = 0.0
    source_rest_z: float = 0.033
    dest_rest_z: float = 0.0


@dataclass(frozen=True)
class Footprint:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    path: str


SCENE_PRESETS: dict[int, ScenePreset] = {
    1302: ScenePreset(
        default_robot_xy=(10.8, 3.2),
        default_robot_yaw_rad=0.0,
        robot_clearance_radius=0.45,
        source_spawn=SceneSpawnCfg(
            min_robot_dist=4.0,
            max_robot_dist=20.0,
            clearance_radius=0.14,
        ),
        dest_spawn=SceneSpawnCfg(
            min_robot_dist=5.5,
            max_robot_dist=24.0,
            clearance_radius=0.18,
        ),
        source_dest_min_separation=3.0,
        support_floor_prim_path="/train_1302/Geometry/floor",
        source_rest_z=0.033,
        dest_rest_z=0.0,
    ),
}


def resolve_scene_usd(scene_idx: int, scene_usd: str, install_dir: str) -> Path | None:
    if scene_usd:
        path = Path(scene_usd).expanduser()
        return path if path.is_file() else None
    if scene_idx < 0:
        return None
    path = Path(install_dir).expanduser() / "scenes" / "procthor-10k-train" / f"train_{scene_idx}" / "scene.usda"
    return path if path.is_file() else None


def load_scene_reference(scene_usd: Path, prim_path: str = SCENE_PRIM_PATH) -> str:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    scene_prim = stage.DefinePrim(prim_path, "Xform")
    scene_prim.GetReferences().AddReference(str(scene_usd.resolve()))
    stage.Load()
    return prim_path


def sample_scene_task_layout(
    scene_idx: int,
    *,
    scene_usd: str | Path,
    robot_xy: tuple[float, float] | None = None,
    robot_yaw_rad: float | None = None,
    source_rest_z: float | None = None,
    dest_rest_z: float | None = None,
    floor_z: float | None = None,
    scene_scale: float = 1.0,
    rng: random.Random | None = None,
    source_spawn_override: SceneSpawnCfg | None = None,
    dest_spawn_override: SceneSpawnCfg | None = None,
    robot_faces_source: bool = False,
    robot_heading_noise_std: float = 0.35,
    robot_heading_max_rad: float = 0.76,
    randomize_robot_xy: bool = False,
) -> SceneTaskLayout:
    if scene_idx not in SCENE_PRESETS:
        raise KeyError(f"No scene layout preset for scene_idx={scene_idx}")

    preset = SCENE_PRESETS[scene_idx]
    rng = rng or random.Random()
    scene_path = Path(scene_usd).expanduser().resolve()

    src_rest_z = preset.source_rest_z if source_rest_z is None else float(source_rest_z)
    dst_rest_z = preset.dest_rest_z if dest_rest_z is None else float(dest_rest_z)

    support_floor_z = _load_support_floor_z(str(scene_path), preset.support_floor_prim_path)
    if floor_z is None:
        floor_z = support_floor_z
    else:
        floor_z = float(floor_z)

    obstacles = _load_scene_obstacles(str(scene_path))
    regions = _load_floor_regions(str(scene_path), support_floor_z=support_floor_z)
    if not regions:
        raise RuntimeError(f"No floor regions found for scene {scene_idx}")

    # Robot position: random in scene or fixed
    if randomize_robot_xy and robot_xy is None:
        # scene 내부 bounding box 제한 (검증된 좌표 범위, world 좌표 기준)
        # scene_scale 적용 전 원본 좌표로 변환하여 제한
        ss = float(scene_scale) if scene_scale > 0 else 1.0
        bbox_min = (2.0 / ss, 2.0 / ss)   # world (2.0, 2.0) → original
        bbox_max = (9.5 / ss, 16.0 / ss)  # world (9.5, 16.0) → original
        robot_xy = _sample_random_point_in_regions(
            regions, obstacles, preset.robot_clearance_radius, rng,
            bbox_min=bbox_min, bbox_max=bbox_max,
        )
    else:
        robot_xy = robot_xy or preset.default_robot_xy

    robot_yaw = preset.default_robot_yaw_rad if robot_yaw_rad is None else float(robot_yaw_rad)

    if not _point_is_clear(robot_xy, preset.robot_clearance_radius, obstacles, regions=regions):
        raise RuntimeError(
            f"Robot pose {robot_xy} is not clear in scene {scene_idx}. "
            "Update the robot pose or scene preset."
        )

    src_spawn = source_spawn_override or preset.source_spawn
    dst_spawn = dest_spawn_override or preset.dest_spawn

    source_xy = _sample_point_in_regions(
        robot_xy,
        src_spawn,
        regions,
        obstacles,
        occupied=[(robot_xy, preset.robot_clearance_radius)],
        rng=rng,
    )
    dest_xy = _sample_point_in_regions(
        robot_xy,
        dst_spawn,
        regions,
        obstacles,
        occupied=[
            (robot_xy, preset.robot_clearance_radius),
            (source_xy, preset.source_dest_min_separation),
        ],
        rng=rng,
    )

    # Robot yaw: face the source object (LeKiwi forward = +Y body axis)
    if robot_faces_source:
        dx = source_xy[0] - robot_xy[0]
        dy = source_xy[1] - robot_xy[1]
        angle_to_obj = math.atan2(dy, dx)
        yaw_for_forward_y = angle_to_obj - math.pi / 2
        heading_noise = rng.gauss(0.0, robot_heading_noise_std)
        heading_noise = max(-robot_heading_max_rad, min(robot_heading_max_rad, heading_noise))
        robot_yaw = math.atan2(
            math.sin(yaw_for_forward_y + heading_noise),
            math.cos(yaw_for_forward_y + heading_noise),
        )

    # Scene scale 적용: 모든 좌표를 scene_scale로 곱함
    # (USD Xform scale이 scene 전체를 축소하므로 world 좌표도 동일하게 축소)
    s = float(scene_scale)
    return SceneTaskLayout(
        robot_xy=(robot_xy[0] * s, robot_xy[1] * s),
        robot_yaw_rad=robot_yaw,
        source_xy=(source_xy[0] * s, source_xy[1] * s),
        source_yaw_rad=rng.uniform(-math.pi, math.pi),
        dest_xy=(dest_xy[0] * s, dest_xy[1] * s),
        dest_yaw_rad=rng.uniform(-math.pi, math.pi),
        floor_z=floor_z * s,
        source_rest_z=src_rest_z,
        dest_rest_z=dst_rest_z,
    )


def apply_scene_task_layout(env, layout: SceneTaskLayout) -> SceneTaskLayout:
    dev = env.device
    env_id = torch.tensor([0], device=dev)

    robot_state = env.robot.data.root_state_w.clone()
    robot_height = float(env.robot.data.default_root_state[0, 2].item())
    robot_state[0, 0] = float(layout.robot_xy[0])
    robot_state[0, 1] = float(layout.robot_xy[1])
    robot_state[0, 2] = float(layout.floor_z) + robot_height
    robot_state[0, 3:7] = _quat_from_yaw(layout.robot_yaw_rad, dev)
    robot_state[0, 7:] = 0.0
    env.robot.write_root_state_to_sim(robot_state, env_id)
    env.home_pos_w[0] = robot_state[0, :3]

    if getattr(env, "object_rigid", None) is not None:
        obj_pose = env.object_rigid.data.default_root_state[0:1, :7].clone()
        obj_pose[0, 0] = float(layout.source_xy[0])
        obj_pose[0, 1] = float(layout.source_xy[1])
        obj_pose[0, 2] = float(layout.floor_z) + float(layout.source_rest_z)
        obj_pose[0, 3:7] = _quat_from_yaw(layout.source_yaw_rad, dev)
        env.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=env_id)
        env.object_rigid.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=env_id)
        env.object_pos_w[0] = obj_pose[0, :3]

    dest_rigid = getattr(env, "_dest_object_rigid", None)
    if dest_rigid is not None:
        dest_pose = dest_rigid.data.default_root_state[0:1, :7].clone()
        dest_pose[0, 0] = float(layout.dest_xy[0])
        dest_pose[0, 1] = float(layout.dest_xy[1])
        dest_pose[0, 2] = float(layout.floor_z) + float(layout.dest_rest_z)
        dest_pose[0, 3:7] = _quat_from_yaw(layout.dest_yaw_rad, dev)
        dest_rigid.write_root_pose_to_sim(dest_pose, env_ids=env_id)
        if not bool(getattr(env.cfg, "dest_object_fixed", False)):
            dest_rigid.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=env_id)
        env.dest_object_pos_w[0] = dest_pose[0, :3]

    if hasattr(env, "object_grasped"):
        env.object_grasped[0] = False
    if hasattr(env, "task_success"):
        env.task_success[0] = False
    if hasattr(env, "just_grasped"):
        env.just_grasped[0] = False
    if hasattr(env, "just_dropped"):
        env.just_dropped[0] = False
    if hasattr(env, "prev_object_dist") and hasattr(env, "object_pos_w") and hasattr(env, "home_pos_w"):
        env.prev_object_dist[0] = torch.norm(env.object_pos_w[0, :2] - env.home_pos_w[0, :2])
    return layout


def estimate_spawn_clearance(scene_usd: str | Path, xy: tuple[float, float]) -> float:
    obstacles = _load_scene_obstacles(str(Path(scene_usd).expanduser().resolve()))
    if not obstacles:
        return float("inf")
    return min(_distance_point_to_aabb(xy, obs) for obs in obstacles)


@lru_cache(maxsize=None)
def _load_scene_obstacles(scene_usd: str) -> tuple[Footprint, ...]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_usd)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    obstacles: list[Footprint] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetName().lower()
        if "room_" in name:
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = bounds.GetMin()
        mx = bounds.GetMax()
        if mx[2] < FLOOR_OBSTACLE_MIN_TOP_Z:
            continue
        if mn[2] > FLOOR_OBSTACLE_MAX_BOTTOM_Z:
            continue
        obstacles.append(
            Footprint(
                min_x=float(mn[0]),
                min_y=float(mn[1]),
                max_x=float(mx[0]),
                max_y=float(mx[1]),
                path=str(prim.GetPath()),
            )
        )
    return tuple(obstacles)


def _find_robot_region(
    robot_xy: tuple[float, float],
    regions: tuple[Footprint, ...],
) -> Footprint | None:
    """로봇이 속한 region 반환. bbox 겹침 시 가장 작은 area 선택. 없으면 가장 가까운 region."""
    candidates = []
    for region in regions:
        if (region.min_x <= robot_xy[0] <= region.max_x
                and region.min_y <= robot_xy[1] <= region.max_y):
            area = (region.max_x - region.min_x) * (region.max_y - region.min_y)
            candidates.append((area, region))
    if candidates:
        candidates.sort(key=lambda x: x[0])  # 가장 작은 area 우선
        return candidates[0][1]
    # fallback: 가장 가까운 region
    best, best_dist = None, float("inf")
    for region in regions:
        cx = 0.5 * (region.min_x + region.max_x)
        cy = 0.5 * (region.min_y + region.max_y)
        d = math.dist(robot_xy, (cx, cy))
        if d < best_dist:
            best, best_dist = region, d
    return best


def _sample_point_in_regions(
    robot_xy: tuple[float, float],
    spawn_cfg: SceneSpawnCfg,
    regions: tuple[Footprint, ...],
    obstacles: tuple[Footprint, ...],
    *,
    occupied: list[tuple[tuple[float, float], float]],
    rng: random.Random,
    max_tries: int = 2000,
) -> tuple[float, float]:
    # 로봇이 속한 region에서만 스폰 (다른 방 스폰 방지)
    robot_region = _find_robot_region(robot_xy, regions)
    if robot_region is not None:
        same_room = tuple(
            r for r in regions
            if r.path == robot_region.path
            and (r.max_x - r.min_x) > (2.0 * spawn_cfg.clearance_radius)
            and (r.max_y - r.min_y) > (2.0 * spawn_cfg.clearance_radius)
        )
    else:
        same_room = ()

    # fallback: 같은 방에서 못 찾으면 전체 region 사용
    if not same_room:
        same_room = tuple(
            region for region in regions
            if (region.max_x - region.min_x) > (2.0 * spawn_cfg.clearance_radius)
            and (region.max_y - region.min_y) > (2.0 * spawn_cfg.clearance_radius)
        )
    if not same_room:
        raise RuntimeError("No floor regions large enough for spawn sampling.")
    weights = [(r.max_x - r.min_x) * (r.max_y - r.min_y) for r in same_room]

    for _ in range(max_tries):
        region = rng.choices(same_room, weights=weights, k=1)[0]
        cand = (
            rng.uniform(region.min_x + spawn_cfg.clearance_radius, region.max_x - spawn_cfg.clearance_radius),
            rng.uniform(region.min_y + spawn_cfg.clearance_radius, region.max_y - spawn_cfg.clearance_radius),
        )
        robot_dist = math.dist(cand, robot_xy)
        if robot_dist < spawn_cfg.min_robot_dist or robot_dist > spawn_cfg.max_robot_dist:
            continue
        if not _point_is_clear(cand, spawn_cfg.clearance_radius, obstacles, regions=regions):
            continue
        if any(math.dist(cand, other_xy) < min_sep for other_xy, min_sep in occupied):
            continue
        return cand

    raise RuntimeError("Failed to sample a collision-free spawn position in scene.")


def _sample_random_point_in_regions(
    regions: tuple[Footprint, ...],
    obstacles: tuple[Footprint, ...],
    clearance_radius: float,
    rng: random.Random,
    max_tries: int = 2000,
    bbox_min: tuple[float, float] | None = None,
    bbox_max: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Sample a random collision-free point on floor regions, optionally within bounding box."""
    valid = tuple(
        r for r in regions
        if (r.max_x - r.min_x) > 2.0 * clearance_radius
        and (r.max_y - r.min_y) > 2.0 * clearance_radius
    )
    if not valid:
        raise RuntimeError("No floor regions large enough for robot spawn.")
    weights = [(r.max_x - r.min_x) * (r.max_y - r.min_y) for r in valid]
    for _ in range(max_tries):
        region = rng.choices(valid, weights=weights, k=1)[0]
        cand = (
            rng.uniform(region.min_x + clearance_radius, region.max_x - clearance_radius),
            rng.uniform(region.min_y + clearance_radius, region.max_y - clearance_radius),
        )
        # bounding box 제한
        if bbox_min and bbox_max:
            if cand[0] < bbox_min[0] or cand[0] > bbox_max[0]:
                continue
            if cand[1] < bbox_min[1] or cand[1] > bbox_max[1]:
                continue
        if _point_is_clear(cand, clearance_radius, obstacles, regions=regions):
            return cand
    raise RuntimeError("Failed to sample collision-free robot position in scene.")


def _point_is_clear(
    xy: tuple[float, float],
    clearance_radius: float,
    obstacles: tuple[Footprint, ...],
    *,
    regions: tuple[Footprint, ...],
) -> bool:
    x, y = xy
    if not any(_point_in_region(xy, clearance_radius, region) for region in regions):
        return False
    return all(_distance_point_to_aabb(xy, obs) >= clearance_radius for obs in obstacles)


@lru_cache(maxsize=None)
def _load_floor_regions(scene_usd: str, *, support_floor_z: float) -> tuple[Footprint, ...]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_usd)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    regions: list[Footprint] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetName().lower()
        if not name.startswith("room_") or "_visual_" not in name:
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = bounds.GetMin()
        mx = bounds.GetMax()
        # visual room mesh는 실제 collider가 아니므로 XY 샘플링 영역으로만 사용한다.
        # 실제 지지면은 support_floor_prim_path collider가 담당한다.
        if abs(float(mx[2]) - support_floor_z) > 0.15 and abs(float(mn[2]) - support_floor_z) > 0.15:
            continue
        if (mx[0] - mn[0]) * (mx[1] - mn[1]) < 4.0:
            continue
        regions.append(
            Footprint(
                min_x=float(mn[0]),
                min_y=float(mn[1]),
                max_x=float(mx[0]),
                max_y=float(mx[1]),
                path=str(prim.GetPath()),
            )
        )
    return tuple(regions)


@lru_cache(maxsize=None)
def _load_support_floor_z(scene_usd: str, floor_prim_path: str) -> float:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(scene_usd)
    prim = stage.GetPrimAtPath(floor_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Support floor prim not found: {floor_prim_path}")
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Support floor prim has no collision API: {floor_prim_path}")

    xform = UsdGeom.Xformable(prim)
    local_to_world: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = local_to_world.ExtractTranslation()
    return float(translation[2])


def _distance_point_to_aabb(xy: tuple[float, float], footprint: Footprint) -> float:
    x, y = xy
    dx = max(footprint.min_x - x, 0.0, x - footprint.max_x)
    dy = max(footprint.min_y - y, 0.0, y - footprint.max_y)
    return math.hypot(dx, dy)


def _point_in_region(xy: tuple[float, float], clearance_radius: float, region: Footprint) -> bool:
    x, y = xy
    return (
        region.min_x + clearance_radius <= x <= region.max_x - clearance_radius
        and region.min_y + clearance_radius <= y <= region.max_y - clearance_radius
    )


def _quat_from_yaw(yaw_rad: float, device: torch.device | str) -> torch.Tensor:
    half = 0.5 * float(yaw_rad)
    return torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=torch.float32,
        device=device,
    )
