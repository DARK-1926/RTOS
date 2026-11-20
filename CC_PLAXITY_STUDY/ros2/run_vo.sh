#!/bin/bash
# G14 (critique C2): telemetry pipeline vo_ros_tm under REAL-APPLICATION interference instead of the Autoware reference system.
# Instance i (i = 0, 1, 2, ...) runs real programs in a loop, by type i mod 4:
#   0: C/C++ build   (gcc -O2 of lidar_cl.c, repeatedly)           compiler: CPU + I-cache + memory
#   1: gzip -6 of nuScenes scans (tar | gzip > /dev/null)          CPU + streaming memory
#   2: xz -3 of nuScenes scans                                     memory-intensive compression
#   3: python3 data processing (sort + hash of 2e6 floats)         interpreter, allocator, memory
# Calibration K in {0,1,2} = {none, compiler, compiler+gzip}; validation adds K=3 = +xz: an interferer TYPE never seen in calibration.
# Usage: run_g14.sh <seed> <calib|val> <nframes> <outdir>
# Regime dwell times ~ Exp(mean 8 s), drawn from a seeded RNG; K is written to a file the pipeline polls.

SEED=$1; SPLIT=$2; NFR=$3; OUT=$4
ROOT=/home/arduino/ros
export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros
source $ROOT/ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
JOB() {   # $1 = type
  case $1 in
    0) while true; do cc -O2 -o /tmp/lcl_$$ /home/arduino/ccc/lidar_cl.c -lm -lpthread 2>/dev/null; done ;;
    1) while true; do tar cf - -C /home/arduino/vo frames 2>/dev/null | gzip -6 > /dev/null; done ;;
    2) while true; do tar cf - -C /home/arduino/vo frames 2>/dev/null | xz -3 -T1 > /dev/null; done ;;
    3) while true; do python3 -c "import random,hashlib; a=[random.random() for _ in range(2000000)]; a.sort(); hashlib.sha256(str(a[:100000]).encode()).hexdigest()"; done ;;
  esac
}
PIPE=$ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/vo_ros_tm
REGF=/tmp/cc_regime_${SEED}_${SPLIT}; echo 0 > $REGF
mkdir -p $OUT
declare -a PIDS=()
set_k() {  # start/stop reference-system instances until exactly $1 run
  while [ ${#PIDS[@]} -lt $1 ]; do
    i=${#PIDS[@]}; setsid bash -c "$(declare -f JOB); JOB $((i % 4))" > /dev/null 2>&1 & PIDS+=($!)
  done
  while [ ${#PIDS[@]} -gt $1 ]; do
    last=$((${#PIDS[@]} - 1)); kill -- -${PIDS[$last]} 2>/dev/null; wait ${PIDS[$last]} 2>/dev/null; unset "PIDS[$last]"; PIDS=("${PIDS[@]}")
  done
}
TOP=2; [ "$SPLIT" = "val" ] && TOP=3
python3 - "$SEED" "$SPLIT" "$TOP" > $OUT/schedule_s${SEED}_${SPLIT}.txt <<'EOF'
import sys, random
seed, split, top = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
r = random.Random(seed * 1000 + (7 if split == "val" else 3)); k = 0
for _ in range(400):
    k = r.choice([x for x in range(top + 1) if x != k]); print(k, round(r.expovariate(1 / 8.0), 2))
EOF
ROS_DOMAIN_ID=1 $PIPE /home/arduino/vo/frames /home/arduino/vo/frames/idx_${SPLIT}.txt $OUT/vo_s${SEED}_${SPLIT}.csv \
  $NFR ${VO_PERIOD:-100} $SPLIT $REGF $SEED > $OUT/log_s${SEED}_${SPLIT}.txt 2>&1 &
PP=$!
# regime driver: the label changes only after the instance set is in place
while read K DW; do
  kill -0 $PP 2>/dev/null || break
  echo -1 > $REGF          # transition: frames released now are labelled K-1 and dropped from analysis
  set_k $K; sleep 1; echo $K > $REGF
  echo "$(date +%s.%N) K=$K dwell=$DW" >> $OUT/regime_log_s${SEED}_${SPLIT}.txt
  sleep $DW
done < $OUT/schedule_s${SEED}_${SPLIT}.txt
wait $PP; RC=$?
set_k 0
echo "rc=$RC" >> $OUT/log_s${SEED}_${SPLIT}.txt
