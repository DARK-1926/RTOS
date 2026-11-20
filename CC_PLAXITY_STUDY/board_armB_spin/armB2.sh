#!/bin/sh
# Arm B (no root): frozen script unchanged; clock held at max by 4 SCHED_IDLE spinners (one per core).
cd /home/arduino/ccrun; mkdir -p armB; rm -f armB/status.txt armB/governor_log.txt
for c in 0 1 2 3; do python3 spin.py $c & done
( while [ ! -f armB/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ')" >> armB/freq_log.txt; sleep 5; done ) &
sleep 2
for s in 1 2 3 4 5; do
  echo "seed $s start $(date)" >> armB/run_log.txt
  python3 unoq_cc_confirm.py --out armB/cc_perf_seed$s.csv --seed $s > armB/log_seed$s.txt 2>&1
  echo "seed $s end rc=$? $(date)" >> armB/run_log.txt
done
echo ALLDONE > armB/done.flag
pkill -f "[s]pin.py"
