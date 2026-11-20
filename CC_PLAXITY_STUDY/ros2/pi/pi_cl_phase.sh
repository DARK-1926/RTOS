#!/bin/bash
# Addendum O on the Pi 5. Usage: pi_cl_phase.sh cal <seed> [ncal nrand] | val <seed> [nval]
# cal: calibration, policy none (dummy params; monitors never act) + a second calibration run with random degradation p = 0.5
#      (gives the degraded-mode median ME2E_D). val: policies none, kp (legacy), kpm, kpmwh, toba, then random with p = degraded
#      fraction of the kpm run. Same seeded interference schedule for every policy of a seed.
PH=$1; S=$2; . $HOME/ccpi/pi_env.sh; cd $CCPI; OUT=$CCPI/o/s$S; mkdir -p $OUT
exec 8>$CCPI/tmp/pi_cl.lock; flock -n 8 || { echo "pi_cl_phase already running"; exit 0; }
( while [ ! -f $OUT/phase_${PH}.done ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') T=$(cat /sys/class/thermal/thermal_zone0/temp)" >> $OUT/freq_log.txt; sleep 5; done ) &
if [ "$PH" = cal ]; then
  ./run_cl_pi.sh $S calib ${3:-3000} $OUT $CCPI/dummy_params.txt none 0
  ./run_cl_pi.sh $S calib ${4:-1500} $OUT $CCPI/dummy_params.txt random 0.5
else
  PRM=$OUT/params_s$S.txt; N=${3:-3000}
  md5sum run_cl_pi.sh pi_cl_phase.sh ws/src/cc_lidar_ros/src/lidar_ros_cl2.cpp ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl2 $PRM > $OUT/md5.txt
  for P in none kp kpm kpmwh toba; do ./run_cl_pi.sh $S val $N $OUT $PRM $P 0; done
  PR=$(awk -F, 'NR>21 && $7==1 && $2!="K-1" {n++; if ($21>0) d++} END {printf "%.4f", d/n}' $OUT/cl_s${S}_val_kpm.csv)
  echo "random p = $PR" >> $OUT/run_log.txt
  ./run_cl_pi.sh $S val $N $OUT $PRM random $PR
fi
echo "$PH $S done $(date)" >> $OUT/run_log.txt
touch $OUT/phase_${PH}.done
