#!/bin/bash
# Resume after the board reboot: re-run only the interrupted seed-122 random run (same seed, schedule and p).
OUT=/home/arduino/ros/g12; cd /home/arduino/ros
mkdir -p $OUT/interrupted; mv $OUT/log_s122_val_random.txt $OUT/regime_log_s122_val_random.txt $OUT/interrupted/ 2>/dev/null
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done; sleep 2
( while [ ! -f $OUT/phase_val_122.done ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr "
" " ")" >> $OUT/freq_log.txt; sleep 5; done ) &
echo "resume after reboot: random p = 0.8547 (from the kp run)" >> $OUT/run_log.txt
./run_g12.sh 122 val 2000 $OUT $OUT/params_s122.txt random 0.8547
echo "val 122 done (resumed) $(date)" >> $OUT/run_log.txt
for p in "${SP[@]}"; do kill $p; done
touch $OUT/phase_val_122.done
