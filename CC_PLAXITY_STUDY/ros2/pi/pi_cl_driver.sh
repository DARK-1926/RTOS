#!/bin/bash
# Laptop-side driver for Addendum O: calibration on the Pi -> export params on the laptop -> validation on the Pi -> pull.
# Usage: bash pi_cl_driver.sh <data_subdir> "<seeds>" [ncal nrand nval]
SUB=$1; SEEDS=$2; NC=${3:-3000}; NR=${4:-1500}; NV=${5:-3000}
S="ssh -i $HOME/.ssh/id_ed25519_pi -o BatchMode=yes -o ConnectTimeout=10 prollysamz@10.0.9.145"; SCP="scp -q -i $HOME/.ssh/id_ed25519_pi"
G=/d/RTKK/CC_PLAXITY_STUDY/gen; D=$G/data/$SUB; mkdir -p $D
wait_flag() { until timeout 30 $S "test -f ~/ccpi/o/s$1/phase_$2.done" 2>/dev/null; do
    timeout 30 $S "pgrep -f '[p]i_cl_phase' > /dev/null" 2>/dev/null; [ $? -eq 1 ] && { echo "phase $2 of seed $1 not running and not done: relaunching $(date)"; launch $1 $2; }
    sleep 30; done; }
launch() { if [ $2 = cal ]; then A="$NC $NR"; else A="$NV"; fi
  timeout 30 $S "cd ~/ccpi && setsid nohup ./pi_cl_phase.sh $2 $1 $A >> o_phase_stdout.log 2>&1 < /dev/null &"; }
for X in $SEEDS; do
  timeout 30 $S "test -f ~/ccpi/o/s$X/phase_cal.done" || launch $X cal
  wait_flag $X cal
  for p in none random; do $SCP prollysamz@10.0.9.145:ccpi/o/s$X/cl_s${X}_calib_$p.csv $D/ || { echo "PULL FAILED $X"; exit 1; }; done
  (cd $G && python kp_cl2_export.py $D/cl_s${X}_calib_none.csv $D/cl_s${X}_calib_random.csv $D/params_s$X.txt) >> $D/export_log.txt 2>&1 || { echo "EXPORT FAILED $X"; exit 1; }
  $SCP $D/params_s$X.txt prollysamz@10.0.9.145:ccpi/o/s$X/params_s$X.txt || { echo "PUSH FAILED $X"; exit 1; }
  echo "seed $X params exported $(date): $(tail -n 1 $D/export_log.txt)"
  timeout 30 $S "test -f ~/ccpi/o/s$X/phase_val.done" || launch $X val
  wait_flag $X val
  for p in none kp kpm kpmwh toba random; do $SCP prollysamz@10.0.9.145:ccpi/o/s$X/cl_s${X}_val_$p.csv $D/ || { echo "PULL FAILED $X"; exit 1; }
    r=$(timeout 30 $S "md5sum ccpi/o/s$X/cl_s${X}_val_$p.csv" | cut -c1-32); l=$(md5sum $D/cl_s${X}_val_$p.csv | cut -c1-32); echo "cl_s${X}_val_$p.csv $r $l $([ -n "$r" ] && [ "$r" = "$l" ] && echo OK || echo MISMATCH)" >> $D/md5_data.txt; done
  $SCP prollysamz@10.0.9.145:ccpi/o/s$X/freq_log.txt $D/freq_log_s$X.txt; $SCP prollysamz@10.0.9.145:ccpi/o/s$X/run_log.txt $D/run_log_s$X.txt
  echo "seed $X val done $(date)"
done
echo ALL_DONE
