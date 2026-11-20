#!/bin/bash
# Addendum S driver (laptop side): stress on seeds 261-263, then the Addendum P chain on seeds 261-267 into data/S.
export MSYS_NO_PATHCONV=1
A="C:/Users/mohit/AppData/Local/Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"
G=/d/RTKK/CC_PLAXITY_STUDY/gen; D=$G/data/S; mkdir -p $D; DW=$(cygpath -m $D)
sh_() { timeout 40 "$A" shell "$1"; }
for X in 261 262 263; do
  sh_ "mkdir -p /home/arduino/ros/p/s$X"
  until sh_ "test -f /home/arduino/ros/p/s$X/phase_stress.done" 2>/dev/null; do
    if [ "$(sh_ "ps aux | grep -c '[u]q_cl_phase'" | tr -d '\r')" = "0" ]; then echo "launch stress $X $(date)"
      sh_ "cd /home/arduino/ros/uq && setsid nohup ./uq_cl_phase.sh stress $X 4000 > /dev/null 2>&1 < /dev/null &"; fi
    sleep 30; done
  f=cl_s${X}_val_stressK3.csv; "$A" pull /home/arduino/ros/p/s$X/$f "$DW/$f" > /dev/null
  r=$(sh_ "md5sum /home/arduino/ros/p/s$X/$f" | cut -c1-32); l=$(md5sum $D/$f | cut -c1-32)
  echo "$f $r $l $([ "$r" = "$l" ] && echo OK || echo MISMATCH)" >> $D/md5_data.txt; echo "stress $X done $(date)"
done
cd /d/RTKK/CC_PLAXITY_STUDY/ros2/uq && bash uq_cl_driver.sh S "261 262 263 264 265 266 267" 3000 1500 2000 m1.6
