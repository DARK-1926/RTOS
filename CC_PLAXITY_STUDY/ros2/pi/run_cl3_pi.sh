#!/bin/bash
# Raspberry Pi 5 port of run_g14.sh / run_vo.sh (Addendum L). Identical pipeline, interferers, schedule RNG and periods;
# only paths differ, and every file stays inside ~/ccpi (borrowed board: nothing outside that folder is written).
#   kind g14: lidar_ros_tm, nuScenes scans, 100 ms period      kind vo: vo_ros_tm, EuRoC frames, 200 ms period
# Interferer i runs type i mod 4: 0 gcc -O2 build, 1 gzip -6, 2 xz -3 (unseen in calibration), 3 python sort+hash.
# CLOSED-LOOP v3 (Addendum Q): pipeline lidar_ros_cl3, as run_cl_uq.sh (Addendum P) but on the Pi 5. Degradation strength from
# env CC_FB_DIV (level 1, default 2) and CC_FB2_DIV (level 2, default 8). Env CC_KFIX=k replaces the seeded schedule by a
# constant contention level k (stress runs only). Env CC_ISO=1 pins interferers to cores 0-2 and the pipeline to core 3.
# Usage: run_cl3_pi.sh <seed> <calib|val> <nframes> <outdir> <params> <policy> <p_random> <tag>
KIND=g14; SEED=$1; SPLIT=$2; NFR=$3; OUT=$4; PARAMS=$5; POL=$6; PR=$7; TAG=$8
. $HOME/ccpi/pi_env.sh
eval "$($CCPI/bin/micromamba shell hook -s bash)"; micromamba activate ros
source $CCPI/ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export CC_FB_DIV=${CC_FB_DIV:-2} CC_FB2_DIV=${CC_FB2_DIV:-8}; KFIX=${CC_KFIX:-}; ISO=${CC_ISO:-}   # CC_ISO=1: interferers pinned to cores 0-2, pipeline to core 3 (user-space taskset, no privileges)
TSI=; TSP=; [ -n "$ISO" ] && { TSI="taskset -c 0-2"; TSP="taskset -c 3"; }
IDL=${CC_IDLE:-}; [ -n "$IDL" ] && TSI="$TSI chrt -i 0"   # CC_IDLE=1: interferers run under SCHED_IDLE (unprivileged; pipeline keeps SCHED_OTHER)
if [ $KIND = vo ]; then SRC=$CCPI/data/vo; SUBD=frames; PIPE=vo_ros_tm; DATA=$CCPI/data/vo/frames; IDX=$CCPI/data/vo/frames/idx_${SPLIT}.txt; PER=200
else SRC=$CCPI/data/ccc; SUBD=lidar_xyz_full; PIPE=lidar_ros_tm; DATA=$CCPI/data/ccc/lidar_xyz_full; IDX=$CCPI/data/ccc/idx_${SPLIT}.txt; PER=100; fi
JOB() {   # $1 = type
  case $1 in
    0) while true; do cc -O2 -o $CCPI/tmp/lcl_$$ $CCPI/data/ccc/lidar_cl.c -lm -lpthread 2>/dev/null; done ;;
    1) while true; do tar cf - -C $SRC $SUBD 2>/dev/null | gzip -6 > /dev/null; done ;;
    2) while true; do tar cf - -C $SRC $SUBD 2>/dev/null | xz -3 -T1 > /dev/null; done ;;
    3) while true; do /usr/bin/python3 -c "import random,hashlib; a=[random.random() for _ in range(2000000)]; a.sort(); hashlib.sha256(str(a[:100000]).encode()).hexdigest()"; done ;;
  esac
}
PIPEX=$CCPI/ws/install/cc_lidar_ros/lib/cc_lidar_ros/lidar_ros_cl3
REGF=$CCPI/tmp/cc_regime_q_${SEED}_${SPLIT}_${TAG}; echo 0 > $REGF
mkdir -p $OUT
declare -a PIDS=()
set_k() {
  while [ ${#PIDS[@]} -lt $1 ]; do
    i=${#PIDS[@]}; setsid $TSI bash -c "CCPI=$CCPI SRC=$SRC SUBD=$SUBD; $(declare -f JOB); JOB $((i % 4))" > /dev/null 2>&1 & PIDS+=($!)
  done
  while [ ${#PIDS[@]} -gt $1 ]; do
    last=$((${#PIDS[@]} - 1)); kill -- -${PIDS[$last]} 2>/dev/null; wait ${PIDS[$last]} 2>/dev/null; unset "PIDS[$last]"; PIDS=("${PIDS[@]}")
  done
}
trap 'set_k 0' EXIT
TOP=2; [ "$SPLIT" = "val" ] && TOP=3
if [ -n "$KFIX" ]; then echo "$KFIX 100000" > $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt; else
/usr/bin/python3 - "$SEED" "$SPLIT" "$TOP" > $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt <<'EOF'
import sys, random
seed, split, top = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
r = random.Random(seed * 1000 + (7 if split == "val" else 3)); k = 0
for _ in range(400):
    k = r.choice([x for x in range(top + 1) if x != k]); print(k, round(r.expovariate(1 / 8.0), 2))
EOF
fi
$TSP $PIPEX $DATA $IDX $OUT/cl_s${SEED}_${SPLIT}_${TAG}.csv $NFR $PER $SPLIT $REGF $SEED $PARAMS $POL $PR > $OUT/log_s${SEED}_${SPLIT}_${TAG}.txt 2>&1 &
PP=$!
while read K DW; do
  kill -0 $PP 2>/dev/null || break
  echo -1 > $REGF
  set_k $K; sleep 1; echo $K > $REGF
  echo "$(date +%s.%N) K=$K dwell=$DW" >> $OUT/regime_log_s${SEED}_${SPLIT}_${TAG}.txt
  sleep $DW & SP=$!   # wake early when the pipeline ends (constant-contention runs have one very long dwell)
  while kill -0 $PP 2>/dev/null && kill -0 $SP 2>/dev/null; do sleep 0.5; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null
done < $OUT/schedule_s${SEED}_${SPLIT}_${TAG}.txt
wait $PP; RC=$?
set_k 0
echo "rc=$RC" >> $OUT/log_s${SEED}_${SPLIT}_${TAG}.txt
