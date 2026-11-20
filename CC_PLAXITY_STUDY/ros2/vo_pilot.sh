#!/bin/bash
# Addendum H Part 2 pilot (seed 189, never analysed for any hypothesis): K0 only, 200 calibration frames, 100 ms period.
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros; source $ROOT/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
mkdir -p $ROOT/h/vo_pilot; echo 0 > /tmp/vo_pilot_reg
SP=(); for c in 0 1 2 3; do python3 $ROOT/spin_nice.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
ROS_DOMAIN_ID=1 $ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/vo_ros_tm /home/arduino/vo/frames /home/arduino/vo/frames/idx_calib.txt \
  $ROOT/h/vo_pilot/vo_s189_pilot.csv 200 100 calib /tmp/vo_pilot_reg 189 > $ROOT/h/vo_pilot/log.txt 2>&1
echo "rc=$?" >> $ROOT/h/vo_pilot/log.txt
for p in "${SP[@]}"; do kill $p; done
