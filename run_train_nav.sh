#!/bin/bash
# Navigate ResiP training launcher (uses env_isaaclab conda env vars)
export CONDA_PREFIX=/home/yubin11/miniconda3/envs/env_isaaclab
export ISAAC_PATH=/home/yubin11/isaacsim
export EXP_PATH=/home/yubin11/isaacsim/apps
export CARB_APP_PATH=/home/yubin11/isaacsim/kit
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/yubin11/isaacsim/python_packages:/home/yubin11/isaacsim/exts/isaacsim.simulation_app:/home/yubin11/isaacsim/extsDeprecated/omni.isaac.kit:/home/yubin11/isaacsim/kit/kernel/py:/home/yubin11/isaacsim/kit/plugins/bindings-python:/home/yubin11/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle:/home/yubin11/isaacsim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/home/yubin11/isaacsim/extscache/omni.kit.pip_archive-0.0.0+8131b85d.lx64.cp311/pip_prebundle:/home/yubin11/isaacsim/exts/omni.isaac.core_archive/pip_prebundle:/home/yubin11/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle:/home/yubin11/isaacsim/exts/omni.pip.compute/pip_prebundle:/home/yubin11/isaacsim/exts/omni.pip.cloud/pip_prebundle"
export LD_LIBRARY_PATH="/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/home/yubin11/isaacsim/.:/home/yubin11/isaacsim/exts/isaacsim.robot.schema/plugins/lib:/home/yubin11/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle:/home/yubin11/isaacsim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/home/yubin11/isaacsim/kit:/home/yubin11/isaacsim/kit/kernel/plugins:/home/yubin11/isaacsim/kit/libs/iray:/home/yubin11/isaacsim/kit/plugins:/home/yubin11/isaacsim/kit/plugins/bindings-python:/home/yubin11/isaacsim/kit/plugins/carb_gfx:/home/yubin11/isaacsim/kit/plugins/rtx:/home/yubin11/isaacsim/kit/plugins/gpu.foundation"
export PATH="/home/yubin11/miniconda3/envs/env_isaaclab/bin:$PATH"
export PYTHONUNBUFFERED=1

cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env

exec /home/yubin11/miniconda3/envs/env_isaaclab/bin/python train_resip.py "$@"
