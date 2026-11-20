#!/bin/bash
# Addendum D, G10b: laptop, lidar_cl scene-disjoint, hog_mult 4, seeds 91-93
cd /d/RTKK/CC_PLAXITY_STUDY/gen; O=data/G10b
powercfg /getactivescheme > $O/power_state.txt 2>&1
for s in 91 92 93; do
  echo "cal $s start $(date)" >> $O/run_log.txt
  ./lidar_cl.exe run lidar_xyz_full idx_calib.txt $O/cal_s$s.csv $s 3000 calib 50 4 - none > $O/log_cal_s$s.txt 2>&1
  LOKY_MAX_CPU_COUNT=8 python export_tables.py $O/cal_s$s.csv $O/tables_s$s.bin 2>&1 | grep "^D=" >> $O/run_log.txt
  for p in none toba cc ccsc naive; do
    echo "val $s $p start $(date)" >> $O/run_log.txt
    ./lidar_cl.exe run lidar_xyz_full idx_val.txt $O/val_s${s}_$p.csv $s 2000 val 50 4 $O/tables_s$s.bin $p > $O/log_val_s${s}_$p.txt 2>&1
  done
done
echo ALLDONE >> $O/run_log.txt
