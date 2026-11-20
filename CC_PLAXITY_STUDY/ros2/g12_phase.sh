#!/bin/bash
# Addendum F, G12 on the board. Usage: g12_phase.sh cal <seed> | val <seed>
# cal: 3000 calibration frames, policy none (dummy params; monitors run but never act).
# val: 2000 frames each for policies none, kp, toba, random (p = degraded fraction of the kp run), same interference schedule.
PH=$1; S=$2; OUT=/home/arduino/ros/g12; mkdir -p $OUT; cd /home/arduino/ros
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
( while [ ! -f $OUT/phase_${PH}_$S.done ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ')" >> $OUT/freq_log.txt; sleep 5; done ) &
if [ "$PH" = cal ]; then
  ./run_g12.sh $S calib 3000 $OUT /home/arduino/ros/g12_dummy_params.txt none 0
else
  PRM=$OUT/params_s$S.txt
  for P in none kp toba; do ./run_g12.sh $S val 2000 $OUT $PRM $P 0; done
  PR=$(awk -F, 'NR>21 && $7==1 && $2!="K-1" {n++; if ($21>0) d++} END {printf "%.4f", d/n}' $OUT/g12_s${S}_val_kp.csv)
  echo "random p = $PR" >> $OUT/run_log.txt
  ./run_g12.sh $S val 2000 $OUT $PRM random $PR
fi
echo "$PH $S done $(date)" >> $OUT/run_log.txt
for p in "${SP[@]}"; do kill $p; done
touch $OUT/phase_${PH}_$S.done
