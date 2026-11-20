#!/bin/bash
# Addendum R (Pi 5): fallback stress under two background-load configurations.
# Every frame at level-2 strength (random p = 1, CC_FB_DIV = 8) under constant contention K3 (CC_KFIX = 3), as Addendum Q.
#   arm base: interferers as in Addendum Q (SCHED_OTHER)
#   arm idle: interferers under SCHED_IDLE (CC_IDLE=1; unprivileged, the pipeline itself is unchanged)
# Usage: pi_r_stress.sh <seed> <nframes> <order: base-idle | idle-base> <outsub>
SEED=$1; N=$2; ORD=$3; SUB=${4:-r}
CCPI=$HOME/ccpi; OUT=$CCPI/$SUB/s$SEED; mkdir -p $OUT $CCPI/tmp
exec 8>$CCPI/tmp/pi_q.lock; flock -n 8 || { echo "pi phase already running"; exit 1; }
cd $CCPI
md5sum run_cl3_pi.sh pi_r_stress.sh ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl3 > $OUT/md5_code.txt
( while true; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tr '\n' ' ') $(cat /sys/class/thermal/thermal_zone0/temp)"; sleep 5; done ) > $OUT/freq_log.txt 2>&1 &
FP=$!
for ARM in ${ORD//-/ }; do
  if [ $ARM = idle ]; then IDV=1; else IDV=; fi
  CC_IDLE=$IDV CC_KFIX=3 CC_FB_DIV=8 ./run_cl3_pi.sh $SEED val $N $OUT $CCPI/dummy_params.txt random 1.0 stress_$ARM 8>&-
  echo "$ARM $SEED done $(date)" >> $OUT/run_log.txt
done
kill $FP
touch $OUT/phase_r.done
