#!/bin/bash
# Pilot (seed 998, not study data): inline KP cost of lidar_ros_cl under K=3 reference-system instances, policy kp.
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros; source $ROOT/ws/install/setup.bash
REF=$ROOT/ws/install/autoware_reference_system/lib/autoware_reference_system/autoware_default_singlethreaded
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done
P=(); for i in 1 2 3; do ROS_DOMAIN_ID=$((20+i)) $REF > /dev/null 2>&1 & P+=($!); done; sleep 4
echo 3 > /tmp/regk3
ROS_DOMAIN_ID=1 timeout 80 $ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl /home/arduino/ccc/lidar_xyz_full /home/arduino/ccc/idx_val.txt /tmp/pk3.csv 400 100 val /tmp/regk3 998 $ROOT/g12_dummy_params.txt kp 0 > /dev/null 2>&1; echo rc=$?
for p in "${P[@]}"; do kill $p; wait $p; done 2>/dev/null; for p in "${SP[@]}"; do kill $p; done
