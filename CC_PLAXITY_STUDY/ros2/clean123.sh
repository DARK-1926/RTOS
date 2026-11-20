#!/bin/bash
for p in $(pgrep -f "g12_phase.sh") $(pgrep -f "run_g12.sh") $(pgrep -f "g12_resume"); do kill $p 2>/dev/null; done
/home/arduino/ros/stopall.sh
cd /home/arduino/ros/g12 && mkdir -p discarded_s123 && mv *s123* phase_*_123.done discarded_s123/ 2>/dev/null
ls | grep -c 123
