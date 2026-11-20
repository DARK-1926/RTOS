#!/bin/bash
for p in $(pgrep -f "g13_all.sh") $(pgrep -f "run_g13.sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "spin_nice.py"); do kill $p; done
/home/arduino/ros/stopall.sh
ps aux | grep -cE "[l]idar_ros|[a]utoware|[s]pin"
