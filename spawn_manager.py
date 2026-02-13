"""
SpawnManager — Isaac Lab 환경에 물체를 동적으로 스폰/관리.

역할:
  1. env reset 시 object_pos_w 좌표에 랜덤 USD 물체 스폰 (1030종)
  2. 매 step object_pos_w 위치 추적 (GRASP 후 로봇 따라감)
  3. 조명 Domain Randomization
  4. per-env 메타데이터 관리 (object_name, instruction 등)
  5. 품질 필터링 (subtask 완성도, object별 cap)

사용법:
  collect_demos.py에서 import하여 사용.
  lekiwi_nav_env.py는 수정하지 않음.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

SPAWNED_ROOT = "/World/SpawnedObjects"
DR_LIGHTS_PATH = "/World/DR_Lights"

DEFAULT_OBJECT_SCALE = 0.7

# Phase → instruction template ("{name}" placeholder)
SUBTASK_TEMPLATES = {
    0: "look around to find the {name}",      # SEARCH
    1: "approach the {name}",                 # APPROACH
    2: "pick up the {name}",                  # GRASP
    3: "return to the starting position",     # RETURN
}
FULL_TASK_TEMPLATE = "find the {name} and bring it back"

# Lighting DR defaults
LIGHTING_CONFIG = {
    "intensity_range": (800.0, 1500.0),
    "color_temp_range": (5400, 7500),
    "position_z_range": (2.0, 3.0),
    "num_lights_range": (1, 2),
    "dome_intensity_range": (200, 400),
}

# RL phase IDs (lekiwi_nav_env.py와 동일)
PHASE_SEARCH = 0
PHASE_APPROACH = 1
PHASE_GRASP = 2
PHASE_RETURN = 3
NUM_PHASES = 4


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════


def _kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """색온도 → RGB (0~1)."""
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = max(0, 99.4708 * math.log(temp) - 161.1195)
    else:
        red = max(0, 329.6987 * ((temp - 60) ** -0.1332))
        green = max(0, 288.1222 * ((temp - 60) ** -0.0755))
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = max(0, 138.5177 * math.log(temp - 10) - 305.0448)
    return (min(255, red) / 255.0, min(255, green) / 255.0, min(255, blue) / 255.0)


# ═══════════════════════════════════════════════════════════════════════
#  SpawnManager
# ═══════════════════════════════════════════════════════════════════════


class SpawnManager:
    """Isaac Lab 환경의 물체 스폰 및 메타데이터 관리.

    lekiwi_nav_env.py를 수정하지 않고, 외부에서 USD stage를 조작한다.
    RL 학습에는 사용하지 않으며, 데모 수집(collect_demos.py) 전용.
    """

    def __init__(
        self,
        index_path: str | Path,
        num_envs: int,
        object_scale: float = DEFAULT_OBJECT_SCALE,
        object_cap: int = 0,
        lighting_config: dict | None = None,
    ):
        """
        Args:
            index_path: mujoco_obj_usd_index_all.jsonl 경로
            num_envs: 병렬 환경 수
            object_scale: 물체 스케일 (기본 0.7)
            object_cap: object별 최대 에피소드 수 (0=무제한)
            lighting_config: 조명 DR 설정 (None=기본값)
        """
        self.num_envs = num_envs
        self.object_scale = object_scale
        self.object_cap = object_cap
        self.light_cfg = lighting_config or LIGHTING_CONFIG

        # 물체 인덱스 로드
        self.objects_list = self._load_index(index_path)
        print(f"  [SpawnManager] {len(self.objects_list)}개 물체 로드: {index_path}")

        # Per-env 상태
        self.object_names: list[str] = [""] * num_envs
        self.object_usds: list[str] = [""] * num_envs
        self.object_scales: list[float] = [object_scale] * num_envs
        self.object_rotations: list[float] = [0.0] * num_envs
        self._spawned: list[bool] = [False] * num_envs

        # 품질 필터링용
        self.object_counts: dict[str, int] = defaultdict(int)
        self._lighting_seed: int = 0

    # ── 인덱스 로드 ──

    def _load_index(self, path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"물체 인덱스 파일 없음: {path}")
        objects = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("status") == "ok" and obj.get("usd") and obj.get("name"):
                        objects.append(obj)
                except (json.JSONDecodeError, KeyError):
                    pass
        if not objects:
            raise RuntimeError(f"인덱스에서 유효한 물체를 찾을 수 없음: {path}")
        return objects

    # ── USD Stage 접근 ──

    @staticmethod
    def _get_stage() -> Usd.Stage:
        return omni.usd.get_context().get_stage()

    def _prim_path(self, env_id: int) -> str:
        return f"{SPAWNED_ROOT}/obj_{env_id}"

    # ── 스폰 / 디스폰 ──

    def spawn_for_env(
        self,
        env_id: int,
        position: torch.Tensor | tuple,
    ) -> str:
        """env_id 환경에 랜덤 물체를 스폰.

        Args:
            env_id: 환경 인덱스
            position: (x, y, z) world 좌표 (env.object_pos_w[env_id])

        Returns:
            스폰된 물체 이름
        """
        stage = self._get_stage()

        # 이전 물체 제거
        self.despawn_for_env(env_id)

        # 랜덤 물체 선택
        obj_info = random.choice(self.objects_list)
        obj_name = obj_info["name"]
        obj_usd = obj_info["usd"]

        # 위치
        if isinstance(position, torch.Tensor):
            x, y, z = position[0].item(), position[1].item(), position[2].item()
        else:
            x, y, z = float(position[0]), float(position[1]), float(position[2])

        # 랜덤 회전 (Z축)
        rot_z = random.uniform(0, 360)

        # USD prim 생성 (world root에 배치, env scope 밖)
        prim_path = self._prim_path(env_id)

        # 부모 Xform 보장
        if not stage.GetPrimAtPath(SPAWNED_ROOT):
            stage.DefinePrim(SPAWNED_ROOT, "Xform")

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(obj_usd)

        # Transform 설정
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(rot_z)))
        s = float(self.object_scale)
        xformable.AddScaleOp().Set(Gf.Vec3f(s, s, s))

        # Kinematic body (시각만, 물리 시뮬레이션으로 움직이지 않음)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        rb_api.CreateKinematicEnabledAttr().Set(True)

        # 메타데이터 저장
        self.object_names[env_id] = obj_name
        self.object_usds[env_id] = obj_usd
        self.object_scales[env_id] = self.object_scale
        self.object_rotations[env_id] = rot_z
        self._spawned[env_id] = True

        return obj_name

    def despawn_for_env(self, env_id: int):
        """env_id의 스폰된 물체 제거."""
        if not self._spawned[env_id]:
            return
        stage = self._get_stage()
        prim_path = self._prim_path(env_id)
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            stage.RemovePrim(prim_path)
        self._spawned[env_id] = False
        self.object_names[env_id] = ""
        self.object_usds[env_id] = ""

    def respawn_for_env(self, env_id: int, position: torch.Tensor | tuple) -> str:
        """기존 물체 제거 후 새 물체 스폰."""
        return self.spawn_for_env(env_id, position)

    def spawn_all(self, positions: torch.Tensor):
        """모든 환경에 물체 스폰.

        Args:
            positions: (num_envs, 3) — env.object_pos_w
        """
        for i in range(self.num_envs):
            self.spawn_for_env(i, positions[i])

    # ── 위치 업데이트 ──

    def update_position(self, env_id: int, position: torch.Tensor | tuple):
        """env_id 물체의 위치를 업데이트 (GRASP 후 로봇 따라갈 때)."""
        if not self._spawned[env_id]:
            return
        stage = self._get_stage()
        prim_path = self._prim_path(env_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        if isinstance(position, torch.Tensor):
            x, y, z = position[0].item(), position[1].item(), position[2].item()
        else:
            x, y, z = float(position[0]), float(position[1]), float(position[2])

        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(x, y, z))
                return

    def update_all_positions(self, env, positions: torch.Tensor):
        """모든 물체 위치 업데이트.

        GRASP/RETURN 중에는 로봇 위치를 따라가도록 처리.

        Args:
            env: LeKiwiNavEnv (phase, object_grasped 참조)
            positions: (num_envs, 3) — env.object_pos_w
        """
        for i in range(self.num_envs):
            if not self._spawned[i]:
                continue
            # GRASP 이후: 로봇 위치를 따라감
            if hasattr(env, "object_grasped") and bool(env.object_grasped[i].item()):
                robot_pos = env.robot.data.root_pos_w[i]
                carried_pos = torch.tensor([robot_pos[0].item(), robot_pos[1].item(), 0.15])
                self.update_position(i, carried_pos)
            else:
                self.update_position(i, positions[i])

    # ── 조명 Domain Randomization ──

    def randomize_lighting(self):
        """글로벌 조명 랜덤화 (모든 env 공유)."""
        stage = self._get_stage()
        cfg = self.light_cfg

        # 기존 DR 조명 제거
        dr_prim = stage.GetPrimAtPath(DR_LIGHTS_PATH)
        if dr_prim and dr_prim.IsValid():
            stage.RemovePrim(DR_LIGHTS_PATH)
        stage.DefinePrim(DR_LIGHTS_PATH, "Xform")

        # Rect lights
        num_lights = random.randint(*cfg["num_lights_range"])
        for i in range(num_lights):
            light_path = f"{DR_LIGHTS_PATH}/Light_{i}"
            light = UsdLux.RectLight.Define(stage, light_path)

            intensity = random.uniform(*cfg["intensity_range"])
            light.GetIntensityAttr().Set(float(intensity))

            color_temp = random.uniform(*cfg["color_temp_range"])
            rgb = _kelvin_to_rgb(color_temp)
            light.GetColorAttr().Set(Gf.Vec3f(*rgb))

            light.GetWidthAttr().Set(float(random.uniform(0.8, 1.2)))
            light.GetHeightAttr().Set(float(random.uniform(0.2, 0.4)))

            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(*cfg["position_z_range"])

            xform = UsdGeom.Xformable(light)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, random.uniform(0, 360)))

        # Dome light (ambient)
        dome = UsdLux.DomeLight.Define(stage, f"{DR_LIGHTS_PATH}/Dome")
        dome.GetIntensityAttr().Set(float(random.uniform(*cfg["dome_intensity_range"])))
        dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        self._lighting_seed += 1

    # ── Instruction 생성 ──

    def get_object_name(self, env_id: int) -> str:
        return self.object_names[env_id]

    def get_subtask_instruction(self, env_id: int, subtask_id: int) -> str:
        """phase ID + 물체 이름 → 자연어 instruction."""
        name = self.object_names[env_id] or "target object"
        display_name = name.replace("_", " ")
        template = SUBTASK_TEMPLATES.get(subtask_id, "unknown subtask")
        return template.format(name=display_name)

    def get_full_task_instruction(self, env_id: int) -> str:
        """전체 태스크 instruction."""
        name = self.object_names[env_id] or "target object"
        display_name = name.replace("_", " ")
        return FULL_TASK_TEMPLATE.format(name=display_name)

    def get_spawn_metadata(self, env_id: int) -> dict:
        """에피소드 저장용 메타데이터."""
        return {
            "object_name": self.object_names[env_id],
            "object_usd": self.object_usds[env_id],
            "object_scale": self.object_scales[env_id],
            "object_rotation_z": self.object_rotations[env_id],
            "lighting_seed": self._lighting_seed,
        }

    # ── 품질 필터링 ──

    def check_quality(
        self,
        env_id: int,
        subtask_ids: list[int],
        min_steps: int = 20,
    ) -> tuple[bool, str]:
        """에피소드 품질 검사.

        Returns:
            (통과 여부, 사유)
        """
        # 1. 최소 step 수
        if len(subtask_ids) < min_steps:
            return False, f"too_short ({len(subtask_ids)} < {min_steps})"

        # 2. 4개 subtask 모두 포함
        unique_phases = set(subtask_ids)
        required = {PHASE_SEARCH, PHASE_APPROACH, PHASE_GRASP, PHASE_RETURN}
        missing = required - unique_phases
        if missing:
            phase_names = {0: "SEARCH", 1: "APPROACH", 2: "GRASP", 3: "RETURN"}
            missing_names = [phase_names[p] for p in missing]
            return False, f"missing_phases: {missing_names}"

        # 3. object별 cap
        obj_name = self.object_names[env_id]
        if self.object_cap > 0 and self.object_counts[obj_name] >= self.object_cap:
            return False, f"object_cap ({obj_name}: {self.object_counts[obj_name]}/{self.object_cap})"

        return True, "ok"

    def record_saved(self, env_id: int):
        """에피소드 저장 기록 (object별 카운터 증가)."""
        obj_name = self.object_names[env_id]
        if obj_name:
            self.object_counts[obj_name] += 1

    def get_object_stats(self) -> dict:
        """수집 통계."""
        return {
            "total_objects_used": len(self.object_counts),
            "total_library": len(self.objects_list),
            "top_5": sorted(self.object_counts.items(), key=lambda x: -x[1])[:5],
            "object_cap": self.object_cap,
        }

    # ── 정리 ──

    def despawn_all(self):
        """모든 스폰 물체 제거."""
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(SPAWNED_ROOT)
        if prim and prim.IsValid():
            stage.RemovePrim(SPAWNED_ROOT)
        for i in range(self.num_envs):
            self._spawned[i] = False
            self.object_names[i] = ""
            self.object_usds[i] = ""

        # DR 조명도 정리
        dr_prim = stage.GetPrimAtPath(DR_LIGHTS_PATH)
        if dr_prim and dr_prim.IsValid():
            stage.RemovePrim(DR_LIGHTS_PATH)


# ═══════════════════════════════════════════════════════════════════════
#  Subtask Transition Builder (SpawnManager 연동)
# ═══════════════════════════════════════════════════════════════════════


def build_subtask_transitions(
    subtask_ids: list[int],
    object_name: str = "",
) -> list[dict]:
    """subtask ID 시퀀스에서 phase 전이 로그를 생성 (물체 이름 포함).

    Args:
        subtask_ids: 매 step의 phase ID 리스트
        object_name: 물체 이름 ("" 이면 "target object" 사용)
    """
    if len(subtask_ids) == 0:
        return []

    display_name = (object_name.replace("_", " ")) if object_name else "target object"

    def _instruction(sid: int) -> str:
        template = SUBTASK_TEMPLATES.get(sid, "unknown subtask")
        return template.format(name=display_name)

    transitions: list[dict] = []
    prev = int(subtask_ids[0])
    transitions.append({
        "step": 0,
        "subtask_id": prev,
        "instruction": _instruction(prev),
    })

    for step, sid_raw in enumerate(subtask_ids[1:], start=1):
        sid = int(sid_raw)
        if sid != prev:
            transitions.append({
                "step": step,
                "from_subtask_id": prev,
                "to_subtask_id": sid,
                "instruction": _instruction(sid),
            })
            prev = sid

    return transitions
