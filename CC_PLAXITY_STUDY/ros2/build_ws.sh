#!/bin/bash
# Build the ROS 2 workspace on the board (no root): Autoware reference system + cc_lidar_ros.
ROOT=/home/arduino/ros
export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros
cd $ROOT/ws
colcon build --packages-up-to autoware_reference_system cc_lidar_ros --parallel-workers 2 \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release > build.log 2>&1
echo "BUILD_EXIT=$?" >> build.log
