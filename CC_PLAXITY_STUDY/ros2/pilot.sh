#!/bin/bash
# Pilot (seed 999, not G11 data): choose the period. K in {0,2,3}, 300 frames each, clock held by spinners.
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros; source $ROOT/ws/install/setup.bash
REF=$ROOT/ws/install/autoware_reference_system/lib/autoware_reference_system/autoware_default_singlethreaded
PIPE=$ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros
PER=${1:-100}
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
echo "freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) period $PER"
echo 0 > /tmp/regsmoke
for K in 0 2 3; do
  P=(); for i in $(seq 1 $K); do ROS_DOMAIN_ID=$((20+i)) $REF > /dev/null 2>&1 & P+=($!); done; sleep 3
  ROS_DOMAIN_ID=1 timeout 120 $PIPE /home/arduino/ccc/lidar_xyz_full /home/arduino/ccc/idx_calib.txt /tmp/smk$K.csv 300 $PER calib /tmp/regsmoke 999 > /dev/null 2>&1
  for p in "${P[@]}"; do kill $p; wait $p; done 2>/dev/null
  awk -F, -v K=$K 'NR>51{n++; if($6==1){d++; s+=$16; a[d]=$16; t+=$9}} END{asort(a); print "K="K, n, d, "mean e2e", s/d, "p50", a[int(d*.5)], "p95", a[int(d*.95)], "max", a[d], "mean s2", t/d}' /tmp/smk$K.csv
done
for p in "${SP[@]}"; do kill $p; done
