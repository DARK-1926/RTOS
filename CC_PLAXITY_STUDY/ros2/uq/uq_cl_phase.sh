#!/bin/bash
# Addendum P on the UNO Q. One board-side phase at a time (flock). Clock held by SCHED_IDLE spinners, as in g14_all.sh.
#   cal <seed> <ncal> <nrand>        : calibration, policy none (dummy params) + random level-1 degradation p = 0.5 (gives ME2E_D)
#   fbtest <seed> <n> "<divs>"       : timeliness of candidate level-2 divisors: random p = 0.5, VALIDATION schedule (K3 included),
#                                      CC_FB_DIV = candidate; one run per divisor (pilot only)
#   val <seed> <n> <load>            : load = heavy | light, params_s<seed>_<load>.txt; policies none kp kpmn kpm kpmwh kpmd toba,
#                                      then random with p = degraded fraction of the kpm run under the same load
PH=$1; S=$2; ROOT=/home/arduino/ros; OUT=$ROOT/p/s$S; mkdir -p $OUT; cd $ROOT/uq
exec 8>/tmp/uq_cl.lock; flock -n 8 || { echo "uq_cl_phase already running"; exit 0; }
FLAG=$OUT/phase_${PH}${4:+_$4}.done; [ "$PH" = fbtest ] && FLAG=$OUT/phase_fbtest.done
SP=(); for c in 0 1 2 3; do python3 $ROOT/spin_nice.py $c > /dev/null 2>&1 8>&- & SP+=($!); done; sleep 2
( while [ ! -f $FLAG ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)" >> $OUT/freq_log.txt; sleep 5; done ) 8>&- &
md5sum run_cl_uq.sh uq_cl_phase.sh $ROOT/ws/src/cc_lidar_ros/src/lidar_ros_cl3.cpp $ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl3 >> $OUT/md5_code.txt
case $PH in
  cal)
    ./run_cl_uq.sh $S calib $3 $OUT $ROOT/p/dummy_params_cl3.txt none 0 none
    ./run_cl_uq.sh $S calib $4 $OUT $ROOT/p/dummy_params_cl3.txt random 0.5 random ;;
  fbtest)
    for d in $4; do CC_FB_DIV=$d ./run_cl_uq.sh $S val $3 $OUT $ROOT/p/dummy_params_cl3.txt random 0.5 fb$d; done ;;
  val)
    L=$4; PRM=$OUT/params_s${S}_$L.txt; md5sum $PRM >> $OUT/md5_code.txt
    for P in none kp kpmn kpm kpmwh kpmd toba; do ./run_cl_uq.sh $S val $3 $OUT $PRM $P 0 ${L}_$P; done
    PR=$(awk -F, 'NR>21 && $7==1 && $2!="K-1" {n++; if ($21>0) d++} END {printf "%.4f", d/n}' $OUT/cl_s${S}_val_${L}_kpm.csv)
    echo "$L random p = $PR" >> $OUT/run_log.txt
    ./run_cl_uq.sh $S val $3 $OUT $PRM random $PR ${L}_random ;;
  stress)   # Addendum S: every frame at level-2 strength (random p = 1, CC_FB_DIV = 8) under constant contention K3
    CC_KFIX=3 CC_FB_DIV=8 ./run_cl_uq.sh $S val $3 $OUT $ROOT/p/dummy_params_cl3.txt random 1.0 stressK3 ;;
esac
echo "$PH $S ${4:-} done $(date)" >> $OUT/run_log.txt
for p in "${SP[@]}"; do kill $p; done
sync; touch $FLAG; sync
