#!/bin/bash
# Round G: telemetry runs. Usage: g13_all.sh <out_subdir> <seed> [<seed> ...]   (calib 3000 + val 3000 per seed, 100 ms period)
# One board-side script, no laptop driver loop. A lock file prevents two copies.
SUB=$1; shift; OUT=/home/arduino/ros/$SUB; mkdir -p $OUT; cd /home/arduino/ros
exec 9>/tmp/g13.lock; flock -n 9 || { echo "another g13_all is running"; exit 1; }
md5sum run_g13.sh spin_nice.py ws/src/cc_lidar_ros/src/lidar_ros_tm.cpp ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_tm > $OUT/md5.txt
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ros/spin_nice.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
( while [ ! -f $OUT/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '
' ' ')" >> $OUT/freq_log.txt; sleep 5; done ) &
for s in "$@"; do for sp in calib val; do
  echo "$s $sp start $(date)" >> $OUT/run_log.txt
  ./run_g13.sh $s $sp 3000 $OUT
  echo "$s $sp end $(date) $(tail -1 $OUT/log_s${s}_${sp}.txt)" >> $OUT/run_log.txt
done; done
for p in "${SP[@]}"; do kill $p; done
touch $OUT/done.flag
