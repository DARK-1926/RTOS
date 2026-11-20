#!/bin/sh
# Addendum D, G10a phase 1: build, quality table, spinners, calibration runs (seeds 71-75)
cd /home/arduino/ccc; mkdir -p g10
md5sum lidar_cl.c > g10/md5.txt
/home/arduino/tc/gcc -O2 -Wall -o lidar_cl lidar_cl.c -lm -lpthread > g10/build.txt 2>&1; md5sum lidar_cl >> g10/md5.txt
./lidar_cl quality lidar_xyz_full g10/quality.csv
for c in 0 1 2 3; do nohup python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & done
sleep 3
( while [ ! -f g10/done1.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | sort -u | tr '\n' ' ') $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') spin=$(ps aux | grep -c '[s]pin.py')" >> g10/freq_log.txt; sleep 5; done ) &
for s in 71 72 73 74 75; do echo "cal $s start $(date)" >> g10/run_log.txt
  ./lidar_cl run lidar_xyz_full idx_calib.txt g10/cal_s$s.csv $s 3000 calib 50 1 - none > g10/log_cal_s$s.txt 2>&1
  echo "cal $s end rc=$? $(date)" >> g10/run_log.txt; done
echo DONE > g10/done1.flag
