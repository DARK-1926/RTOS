#!/bin/bash
# Addendum E, G11 on the board: spinners (clock hold), frequency log, seeds 111-113 (calib then val).
OUT=/home/arduino/ros/g11; mkdir -p $OUT; cd /home/arduino/ros
md5sum run_g11.sh ws/src/cc_lidar_ros/src/lidar_ros.cpp ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros > $OUT/md5.txt
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
( while [ ! -f $OUT/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ')" >> $OUT/freq_log.txt; sleep 5; done ) &
for s in 111 112 113; do for sp in calib val; do
  echo "$s $sp start $(date)" >> $OUT/run_log.txt
  ./run_g11.sh $s $sp 3000 $OUT
  echo "$s $sp end $(date) $(tail -1 $OUT/log_s${s}_${sp}.txt)" >> $OUT/run_log.txt
done; done
for p in "${SP[@]}"; do kill $p; done
echo DONE > $OUT/done.flag
