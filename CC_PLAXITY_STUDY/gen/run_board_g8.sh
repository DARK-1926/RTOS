#!/bin/sh
# Addendum B board queue: performance governor (set by the user as root), no spinners.
cd /home/arduino/ccc; mkdir -p out
md5sum lidar_c.c > out/md5.txt; md5sum /home/arduino/ccrun/unoq_cc_confirm.py >> out/md5.txt
/home/arduino/tc/gcc -O2 -Wall -o lidar_c lidar_c.c -lm -lpthread > out/build.txt 2>&1; md5sum lidar_c >> out/md5.txt
( while [ ! -f out/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | sort -u | tr '\n' ' ') $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') spin=$(ps aux | grep -c '[s]pin.py')" >> out/gov_log.txt; sleep 5; done ) &
for s in 51 52 53 54 55; do echo "G8a $s start $(date)" >> out/run_log.txt
  ./lidar_c lidar_xyz_full out/G8a_c_lidar_perf_s$s.csv $s 3000 2000 50 1 > out/log_G8a_s$s.txt 2>&1
  echo "G8a $s end rc=$? $(date)" >> out/run_log.txt; done
cd /home/arduino/ccrun; mkdir -p armC
for s in 1 2 3 4 5; do echo "ArmC $s start $(date)" >> /home/arduino/ccc/out/run_log.txt
  python3 unoq_cc_confirm.py --out armC/cc_perfgov_seed$s.csv --seed $s > armC/log_seed$s.txt 2>&1
  echo "ArmC $s end rc=$? $(date)" >> /home/arduino/ccc/out/run_log.txt; done
echo ALLDONE > /home/arduino/ccc/out/done.flag
