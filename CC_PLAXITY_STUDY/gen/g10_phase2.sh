#!/bin/sh
# Addendum D, G10a phase 2: val runs (scene-disjoint), 5 policies per seed, same seed -> same contention schedule.
# Then G10s: frozen unoq_cc_confirm.py, seeds 101-103. Spinners from phase 1 keep running.
cd /home/arduino/ccc
( while [ ! -f g10/done2.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | sort -u | tr '\n' ' ') $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') spin=$(ps aux | grep -c '[s]pin.py')" >> g10/freq_log2.txt; sleep 5; done ) &
md5sum g10/tables_s*.bin >> g10/md5.txt
for s in 71 72 73 74 75; do for p in none toba cc ccsc naive; do
  echo "val $s $p start $(date)" >> g10/run_log.txt
  ./lidar_cl run lidar_xyz_full idx_val.txt g10/val_s${s}_$p.csv $s 2000 val 50 1 g10/tables_s$s.bin $p > g10/log_val_s${s}_$p.txt 2>&1
  echo "val $s $p end rc=$? $(date)" >> g10/run_log.txt
done; done
cd /home/arduino/ccrun; mkdir -p g10s; md5sum unoq_cc_confirm.py > g10s/md5.txt
for s in 101 102 103; do echo "G10s $s start $(date)" >> /home/arduino/ccc/g10/run_log.txt
  python3 unoq_cc_confirm.py --out g10s/cc_g10s_seed$s.csv --seed $s > g10s/log_seed$s.txt 2>&1
  echo "G10s $s end rc=$? $(date)" >> /home/arduino/ccc/g10/run_log.txt; done
echo DONE > /home/arduino/ccc/g10/done2.flag
