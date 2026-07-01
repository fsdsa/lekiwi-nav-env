#!/bin/bash
# Scene eval launcher — Isaac Sim 환경 변수 포함
# 경로는 환경변수로 재정의 가능 (기본값은 기존 데스크탑 레이아웃과 동일):
#   ISAAC_PATH         Isaac Sim 설치 경로 (기본 $HOME/isaacsim)
#   ENV_ISAACLAB_DIR   env_isaaclab conda env 경로 (기본 $HOME/miniconda3/envs/env_isaaclab)
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PATH="${ISAAC_PATH:-$HOME/isaacsim}"
ENV_ISAACLAB_DIR="${ENV_ISAACLAB_DIR:-$HOME/miniconda3/envs/env_isaaclab}"

export CONDA_PREFIX="$ENV_ISAACLAB_DIR"
export ISAAC_PATH
export EXP_PATH="$ISAAC_PATH/apps"
export CARB_APP_PATH="$ISAAC_PATH/kit"
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$ISAAC_PATH/python_packages:$ISAAC_PATH/exts/isaacsim.simulation_app:$ISAAC_PATH/extsDeprecated/omni.isaac.kit:$ISAAC_PATH/kit/kernel/py:$ISAAC_PATH/kit/plugins/bindings-python:$ISAAC_PATH/exts/isaacsim.robot_motion.lula/pip_prebundle:$ISAAC_PATH/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$ISAAC_PATH/extscache/omni.kit.pip_archive-0.0.0+8131b85d.lx64.cp311/pip_prebundle:$ISAAC_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_PATH/exts/omni.pip.cloud/pip_prebundle"
export LD_LIBRARY_PATH="/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:$ISAAC_PATH/.:$ISAAC_PATH/exts/isaacsim.robot.schema/plugins/lib:$ISAAC_PATH/exts/isaacsim.robot_motion.lula/pip_prebundle:$ISAAC_PATH/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$ISAAC_PATH/kit:$ISAAC_PATH/kit/kernel/plugins:$ISAAC_PATH/kit/libs/iray:$ISAAC_PATH/kit/plugins:$ISAAC_PATH/kit/plugins/bindings-python:$ISAAC_PATH/kit/plugins/carb_gfx:$ISAAC_PATH/kit/plugins/rtx:$ISAAC_PATH/kit/plugins/gpu.foundation"
export PATH="$ENV_ISAACLAB_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$PROJ_DIR"
exec "$ENV_ISAACLAB_DIR/bin/python" vllm/record_teleop_scene.py "$@"
