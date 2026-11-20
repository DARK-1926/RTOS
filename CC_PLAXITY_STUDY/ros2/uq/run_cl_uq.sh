#!/bin/bash
# Addendum P on the Arduino UNO Q: closed loop v3 (lidar_ros_cl3) under the G14 REAL-APPLICATION interference.
# Identical interferers, schedule RNG and period to run_g14.sh; pipeline = lidar_ros_cl3 (inline monitor + degradation policy).
# Degradation strength from the environment: CC_FB_DIV (level 1), CC_FB2_DIV (level 2). The ROS domain is 1, as in run_g14.sh.
# Addendum S: env CC_KFIX=k replaces the seeded schedule by constant contention k; the dwell sleep wakes when the pipeline ends.
# Usage: run_cl_uq.sh <seed> <calib|val> <nframes> <outdir> <params> <policy> <p_random> <tag>
SEED=$1; SPLIT=$2; NFR=$3; OUT=$4; PARAMS=$5; POL=$6; PR=$7; TAG=$8
ROOT=/home/arduino/ros
export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros
source $ROOT/ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export CC_FB_DIV=${CC_FB_DIV:-2} CC_FB2_DIV=${CC_FB2_DIV:-8}
JOB() {   # $1 = type
  case $1 in
    0) while true; do cc -O2 -o /tmp/lcl_$$ /home/arduino/ccc/lidar_cl.c -lm -lpthread 2>/dev/null; done ;;
    1) while true; do tar cf - -C /home/arduino/ccc lidar_xyz_full 2>/dev/null | gzip -6 > /dev/null; done ;;
    2) while true; do tar cf - -C /home/arduino/ccc lidar_xyz_full 2>/dev/null | xz -3 -T1 > /dev/null; done ;;
    3) while true; do python3 -c "import random,hashlib; a=[random.random() for _ in range(2000000)]; a.sort(); hashlib.sha256(str(a[:100000]).encode()).hexdigest()"; done ;;
  esac
}
PIPE=$ROOT/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl3
REGF=/tmp/cc_regime_p_${SEED}_${SPLIT}_${TAG}; echo 0 > $REGF
mkdir -p $OUT
declare -a PIDS=()
set_k() {
  while [ ${#PIDS[@]} -lt $1 ]; do
    i=${#PIDS[@]}; setsid bash -c "$(declare -f JOB); JOB $((i % 4))" > /dev/null 2>&1 & PIDS+=($!)
  done
  while [ ${#PIDS[@]} -gt $1 ]; do
    last=$((${#PIDS[@]} - 1)); kill -- -${PIDS[$last]} 2>/dev/null; wait ${PIDS[$last]} 2>/dev/null; unset "PIDS[$last]"; PIDS=("${PIDS[@]}")
  done
}
trap 'set_k 0' EXIT
TOP=2; [ "$SPLIT" = "val" ] && TOP=3
python3 - "$SEED" "$SPLIT" "$TOP" > $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt <<'PYEOF'
import sys, random
seed, split, top = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
r = random.Random(seed * 1000 + (7 if split == "val" else 3)); k = 0
for _ in range(400):
    k = r.choice([x for x in range(top + 1) if x != k]); print(k, round(r.expovariate(1 / 8.0), 2))
PYEOF
[ -n "${CC_KFIX:-}" ] && echo "$CC_KFIX 100000" > $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt   # Addendum S: constant contention
ROS_DOMAIN_ID=1 $PIPE /home/arduino/ccc/lidar_xyz_full /home/arduino/ccc/idx_${SPLIT}.txt $OUT/cl_s${SEED}_${SPLIT}_${TAG}.csv \
  $NFR 100 $SPLIT $REGF $SEED $PARAMS $POL $PR > $OUT/log_s${SEED}_${SPLIT}_${TAG}.txt 2>&1 &
PP=$!
while read K DW; do
  kill -0 $PP 2>/dev/null || break
  echo -1 > $REGF
  set_k $K; sleep 1; echo $K > $REGF
  echo "$(date +%s.%N) K=$K dwell=$DW" >> $OUT/regime_log_s${SEED}_${SPLIT}_${TAG}.txt
  sleep $DW & SLP=$!                                   # wake as soon as the pipeline ends (Addendum S; same as the Pi script)
  while kill -0 $PP 2>/dev/null && kill -0 $SLP 2>/dev/null; do sleep 0.5; done; kill $SLP 2>/dev/null
done < $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt
wait $PP; RC=$?
set_k 0
echo "rc=$RC fb=$CC_FB_DIV fb2=$CC_FB2_DIV" >> $OUT/log_s${SEED}_${SPLIT}_${TAG}.txt
