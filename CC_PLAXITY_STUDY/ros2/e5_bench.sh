#!/bin/bash
# Addendum E, E5: KP C monitor cost on the board while K=3 reference-system instances + spinners run.
ROOT=/home/arduino/ros; export MAMBA_ROOT_PREFIX=$ROOT/mamba
eval "$($ROOT/bin/micromamba shell hook -s bash)"; micromamba activate ros; source $ROOT/ws/install/setup.bash
cd $ROOT/kp; md5sum kp_monitor.c > md5.txt
cc -O2 -o kp_monitor kp_monitor.c -lm && md5sum kp_monitor >> md5.txt
REF=$ROOT/ws/install/autoware_reference_system/lib/autoware_reference_system/autoware_default_singlethreaded
SP=(); for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c > /dev/null 2>&1 & SP+=($!); done
P=(); for i in 1 2 3; do ROS_DOMAIN_ID=$((20+i)) $REF > /dev/null 2>&1 & P+=($!); done; sleep 5
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq > freq.txt
./kp_monitor g11_s111_params.txt g11_s111_frames.csv $(cat g11_s111_D.txt) kp_c_board_K3.csv; echo rc=$? >> md5.txt
for p in "${P[@]}"; do kill $p; wait $p; done 2>/dev/null
./kp_monitor g11_s111_params.txt g11_s111_frames.csv $(cat g11_s111_D.txt) kp_c_board_K0.csv
for p in "${SP[@]}"; do kill $p; done
echo DONE > done.flag
