#!/bin/bash
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros; source $ROOT/ws/install/setup.bash
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
echo 0 > /tmp/regsm
for P in none kp toba random; do
  ROS_DOMAIN_ID=1 timeout 60 $ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl /home/arduino/ccc/lidar_xyz_full /home/arduino/ccc/idx_val.txt /tmp/smcl_$P.csv 150 100 val /tmp/regsm 998 $ROOT/g12_dummy_params.txt $P 0.3 > /tmp/smcl_$P.log 2>&1; echo "$P rc=$?"
done
for p in "${SP[@]}"; do kill $p; done
