#!/bin/bash
# Raspberry Pi 5 port of run_g14.sh / run_vo.sh (Addendum L). Identical pipeline, interferers, schedule RNG and periods;
# only paths differ, and every file stays inside ~/ccpi (borrowed board: nothing outside that folder is written).
#   kind g14: lidar_ros_tm, nuScenes scans, 100 ms period      kind vo: vo_ros_tm, EuRoC frames, 200 ms period
# Interferer i runs type i mod 4: 0 gcc -O2 build, 1 gzip -6, 2 xz -3 (unseen in calibration), 3 python sort+hash.
# Usage: run_pi.sh <g14|vo> <seed> <calib|val> <nframes> <outdir>
KIND=$1; SEED=$2; SPLIT=$3; NFR=$4; OUT=$5
. $HOME/ccpi/pi_env.sh
eval "$($CCPI/bin/micromamba shell hook -s bash)"; micromamba activate ros
source $CCPI/ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
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
PIPEX=$CCPI/ws/install/cc_lidar_ros/lib/cc_lidar_ros/$PIPE
REGF=$CCPI/tmp/cc_regime_${KIND}_${SEED}_${SPLIT}; echo 0 > $REGF
mkdir -p $OUT
declare -a PIDS=()
set_k() {
  while [ ${#PIDS[@]} -lt $1 ]; do
    i=${#PIDS[@]}; setsid bash -c "CCPI=$CCPI SRC=$SRC SUBD=$SUBD; $(declare -f JOB); JOB $((i % 4))" > /dev/null 2>&1 & PIDS+=($!)
  done
  while [ ${#PIDS[@]} -gt $1 ]; do
    last=$((${#PIDS[@]} - 1)); kill -- -${PIDS[$last]} 2>/dev/null; wait ${PIDS[$last]} 2>/dev/null; unset "PIDS[$last]"; PIDS=("${PIDS[@]}")
  done
}
trap 'set_k 0' EXIT
TOP=2; [ "$SPLIT" = "val" ] && TOP=3
/usr/bin/python3 - "$SEED" "$SPLIT" "$TOP" > $OUT/schedule_s${SEED}_${SPLIT}.txt <<'EOF'
import sys, random
seed, split, top = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
r = random.Random(seed * 1000 + (7 if split == "val" else 3)); k = 0
for _ in range(400):
    k = r.choice([x for x in range(top + 1) if x != k]); print(k, round(r.expovariate(1 / 8.0), 2))
EOF
$PIPEX $DATA $IDX $OUT/${KIND}_s${SEED}_${SPLIT}.csv $NFR $PER $SPLIT $REGF $SEED > $OUT/log_s${SEED}_${SPLIT}.txt 2>&1 &
PP=$!
while read K DW; do
  kill -0 $PP 2>/dev/null || break
  echo -1 > $REGF
  set_k $K; sleep 1; echo $K > $REGF
  echo "$(date +%s.%N) K=$K dwell=$DW" >> $OUT/regime_log_s${SEED}_${SPLIT}.txt
  sleep $DW
done < $OUT/schedule_s${SEED}_${SPLIT}.txt
wait $PP; RC=$?
set_k 0
echo "rc=$RC" >> $OUT/log_s${SEED}_${SPLIT}.txt
