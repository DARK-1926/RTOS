#!/bin/bash
# Round G (G13): G11 design with the telemetry-logging pipeline lidar_ros_tm (same interference driver as G11).
# Usage: run_g13.sh <seed> <calib|val> <nframes> <outdir>
# Contention level K = number of concurrent autoware_default_singlethreaded instances (each in its own ROS domain,
# so they never exchange messages with each other or with the pipeline; they share CPUs, caches, memory and the
# DDS stack).  Calibration uses K in {0,1,2}; validation uses K in {0,1,2,3} (K3 never seen in calibration).
# Regime dwell times ~ Exp(mean 8 s), drawn from a seeded RNG; K is written to a file the pipeline polls.

SEED=$1; SPLIT=$2; NFR=$3; OUT=$4
ROOT=/home/arduino/ros
export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros
source $ROOT/ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
REF=$ROOT/ws/install/autoware_reference_system/lib/autoware_reference_system/autoware_default_singlethreaded
PIPE=$ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_tm
REGF=/tmp/cc_regime_${SEED}_${SPLIT}; echo 0 > $REGF
mkdir -p $OUT
declare -a PIDS=()
set_k() {  # start/stop reference-system instances until exactly $1 run
  while [ ${#PIDS[@]} -lt $1 ]; do
    i=${#PIDS[@]}; ROS_DOMAIN_ID=$((20 + i)) $REF > /dev/null 2>&1 & PIDS+=($!)
  done
  while [ ${#PIDS[@]} -gt $1 ]; do
    last=$((${#PIDS[@]} - 1)); kill ${PIDS[$last]} 2>/dev/null; wait ${PIDS[$last]} 2>/dev/null; unset "PIDS[$last]"; PIDS=("${PIDS[@]}")
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
ROS_DOMAIN_ID=1 $PIPE /home/arduino/ccc/lidar_xyz_full /home/arduino/ccc/idx_${SPLIT}.txt $OUT/g13_s${SEED}_${SPLIT}.csv \
  $NFR 100 $SPLIT $REGF $SEED > $OUT/log_s${SEED}_${SPLIT}.txt 2>&1 &
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
