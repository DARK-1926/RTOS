#!/bin/bash
# Laptop-side driver for Addendum Q (Pi 5, closed loop v3). Per seed: cal -> pull + export heavy params -> push -> [fbtest] -> val -> stress -> pull.
# Usage: bash pi_q_driver.sh <data_subdir> "<seeds>" <ncal> <nrand> <nval> <n_stress_K3> <n_stress_K4> [fbtest_n "fbtest_divs"]
SUB=$1; SEEDS=$2; NC=$3; NR=$4; NV=$5; N3=$6; N4=$7; FBN=${8:-}; FBD=${9:-}
S="ssh -i $HOME/.ssh/id_ed25519_pi -o BatchMode=yes -o ConnectTimeout=10 prollysamz@10.0.9.145"; SCP="scp -q -i $HOME/.ssh/id_ed25519_pi"
G=/d/RTKK/CC_PLAXITY_STUDY/gen; D=$G/data/$SUB; mkdir -p $D
launch() { timeout 30 $S "cd ~/ccpi && setsid nohup ./pi_q_phase.sh $2 $1 $3 >> q_stdout.log 2>&1 < /dev/null &"; }
phase() {  # seed phase "args" flagname
  F=ccpi/q/s$1/phase_$4.done
  timeout 30 $S "test -f $F" || launch $1 $2 "$3"
  until timeout 30 $S "test -f $F" 2>/dev/null; do
    timeout 30 $S "pgrep -f '[p]i_q_phase' > /dev/null || test -f $F || echo IDLE" 2>/dev/null | grep -q IDLE && { echo "phase $4 of seed $1 not running and not done: relaunching $(date)"; launch $1 $2 "$3"; }
    sleep 30; done; echo "seed $1 $4 done $(date)"; }
pull() {  # seed tag
  f=cl_s$1_$2.csv; $SCP prollysamz@10.0.9.145:ccpi/q/s$1/$f $D/ || { echo "PULL FAILED $f"; return 1; }
  r=$(timeout 30 $S "md5sum ccpi/q/s$1/$f" | cut -c1-32); l=$(md5sum $D/$f | cut -c1-32)
  echo "$f $r $l $([ -n "$r" ] && [ "$r" = "$l" ] && echo OK || echo MISMATCH)" >> $D/md5_data.txt; }
for X in $SEEDS; do
  phase $X cal "$NC $NR" cal; pull $X calib_none; pull $X calib_random
  for L in heavy:0.85 light:m1.6; do
    (cd $G && python kp_cl3_export.py data/$SUB/cl_s${X}_calib_none.csv data/$SUB/cl_s${X}_calib_random.csv data/$SUB/params_s${X}_${L%%:*}.txt ${L#*:}) >> $D/export_log.txt 2>&1 || { echo "EXPORT FAILED $X"; exit 1; }
    $SCP $D/params_s${X}_${L%%:*}.txt prollysamz@10.0.9.145:ccpi/q/s$X/ || { echo "PUSH FAILED $X"; exit 1; }
    echo "seed $X ${L%%:*} params: $(tail -n 1 $D/export_log.txt)"; done
  if [ -n "$FBN" ]; then phase $X fbtest "$FBN \"$FBD\"" fbtest; for d in $FBD; do pull $X val_fb$d; done; fi
  for L in heavy light; do phase $X val "$NV $L" val_$L; for p in none kp kpmn kpm kpmwh kpmd toba random; do pull $X val_${L}_$p; done; done
  phase $X stress "$N3 $N4" stress; pull $X val_stressK3; pull $X val_stressK4
  for f in freq_log run_log md5_code; do $SCP prollysamz@10.0.9.145:ccpi/q/s$X/$f.txt $D/${f}_s$X.txt; done
  echo "seed $X all done $(date)"
done
echo ALL_DONE
