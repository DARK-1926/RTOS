#!/bin/bash
for p in $(pgrep -f "g12_phase.sh") $(pgrep -f "run_g12.sh"); do [ "$p" != "$$" ] && kill $p 2>/dev/null; done
/home/arduino/ros/stopall.sh
cd /home/arduino/ros/g12 && rm -f log_s121_val_* regime_log_s121_val_* schedule_s121_val.txt g12_s121_val_*.csv
rm -f /home/arduino/ros/g12_val_121.log
ls /home/arduino/ros/g12
