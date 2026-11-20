#!/bin/bash
# Addendum Q on the Pi 5 (closed loop v3, as Addendum P). One phase at a time (flock). Performance governor as in Addendum O.
#   cal <seed> <ncal> <nrand>       : calibration, policy none (dummy params) + random level-1 degradation p = 0.5 (gives ME2E_D)
#   fbtest <seed> <n> "<divs>"      : pilot only: random p = 0.5 on the validation schedule with CC_FB_DIV = candidate divisor
#   val <seed> <n> <load>           : load = heavy | light (params_s<seed>_<load>.txt); policies none kp kpmn kpm kpmwh kpmd toba,
#                                     then random at kpm's degraded fraction under the same load
#   stress <seed> <n3> <n4>         : every frame at the level-2 strength (random p = 1, CC_FB_DIV = FB2), constant contention
#                                     K3 for <n3> frames, then constant K4 (beyond anything in calibration or validation) for <n4>
PH=$1; S=$2; . $HOME/ccpi/pi_env.sh; cd $CCPI; OUT=$CCPI/q/s$S; mkdir -p $OUT
exec 8>$CCPI/tmp/pi_q.lock; flock -n 8 || { echo "pi_q_phase already running"; exit 0; }
FLAG=$OUT/phase_${PH}${4:+_$4}.done; [ "$PH" != val ] && FLAG=$OUT/phase_${PH}.done; FB2=${CC_FB2_DIV:-8}; export CC_FB2_DIV=$FB2
( while [ ! -f $FLAG ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') T=$(cat /sys/class/thermal/thermal_zone0/temp)" >> $OUT/freq_log.txt; sleep 5; done ) 8>&- &
md5sum run_cl3_pi.sh pi_q_phase.sh ws/src/cc_lidar_ros/src/lidar_ros_cl3.cpp ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl3 >> $OUT/md5_code.txt
case $PH in
  cal)
    ./run_cl3_pi.sh $S calib $3 $OUT $CCPI/dummy_params.txt none 0 none 8>&-
    ./run_cl3_pi.sh $S calib $4 $OUT $CCPI/dummy_params.txt random 0.5 random 8>&- ;;
  fbtest)
    for d in $4; do CC_FB_DIV=$d ./run_cl3_pi.sh $S val $3 $OUT $CCPI/dummy_params.txt random 0.5 fb$d 8>&-; done ;;
  val)
    L=$4; PRM=$OUT/params_s${S}_$L.txt; md5sum $PRM >> $OUT/md5_code.txt
    for P in none kp kpmn kpm kpmwh kpmd toba; do ./run_cl3_pi.sh $S val $3 $OUT $PRM $P 0 ${L}_$P 8>&-; done
    PR=$(awk -F, 'NR>21 && $7==1 && $2!="K-1" {n++; if ($21>0) d++} END {printf "%.4f", d/n}' $OUT/cl_s${S}_val_${L}_kpm.csv)
    echo "$L random p = $PR" >> $OUT/run_log.txt
    ./run_cl3_pi.sh $S val $3 $OUT $PRM random $PR ${L}_random 8>&- ;;
  stress)
    CC_KFIX=3 CC_FB_DIV=$FB2 ./run_cl3_pi.sh $S val $3 $OUT $CCPI/dummy_params.txt random 1.0 stressK3 8>&-
    CC_KFIX=4 CC_FB_DIV=$FB2 ./run_cl3_pi.sh $S val $4 $OUT $CCPI/dummy_params.txt random 1.0 stressK4 8>&- ;;
esac
echo "$PH $S done $(date)" >> $OUT/run_log.txt
touch $FLAG
