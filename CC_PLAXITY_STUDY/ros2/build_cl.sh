#!/bin/bash
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros
cd $ROOT/ws && colcon build --packages-select cc_lidar_ros --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release > build_cl.log 2>&1
echo "BUILD_EXIT=$?" >> build_cl.log
