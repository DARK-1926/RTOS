#!/bin/bash
# Addendum L pilot (seed 199, never analysed for any hypothesis): checks the pipelines run on the Pi and records K0 / K3 latency.
# 300 calibration frames per kind with the normal calibration schedule (K0-K2), then 300 validation frames (K0-K3).
. $HOME/ccpi/pi_env.sh; cd $CCPI
for k in g14 vo; do for sp in calib val; do ./run_pi.sh $k 199 $sp 300 h/pilot_$k; done; done
echo PILOT_DONE > h/pilot.done
