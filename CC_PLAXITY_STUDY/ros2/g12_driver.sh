#!/bin/bash
# Laptop-side driver for G12: calibration on the board -> export params on the laptop -> validation on the board.
# Local paths are given to adb.exe in Windows form (D:/...): with MSYS_NO_PATHCONV=1 a /d/... path is not converted.
# A calibration that already completed (phase_cal_S.done on the board) is not repeated.
# The driver stops if a pull or an export fails, so a validation never runs without its parameters.
export MSYS_NO_PATHCONV=1
ADB="/c/Users/mohit/AppData/Local/Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"
G=/d/RTKK/CC_PLAXITY_STUDY/gen; GW=D:/RTKK/CC_PLAXITY_STUDY/gen; mkdir -p $G/data/G12
LOCK=$G/data/G12/.driver.lock
mkdir "$LOCK" 2>/dev/null || { echo "ANOTHER DRIVER IS RUNNING (lock $LOCK); exiting"; exit 1; }
trap 'rmdir "$LOCK"' EXIT
for S in 121 122 123; do
  if "$ADB" shell "test -f ~/ros/g12/phase_val_$S.done"; then echo "seed $S already complete, skipped"; continue; fi
  if ! "$ADB" shell "test -f ~/ros/g12/phase_cal_$S.done"; then
    "$ADB" shell "cd ~/ros && setsid nohup ./g12_phase.sh cal $S > g12_cal_$S.log 2>&1 < /dev/null &"
    until "$ADB" shell "test -f ~/ros/g12/phase_cal_$S.done"; do sleep 20; done
  fi
  "$ADB" pull /home/arduino/ros/g12/g12_s${S}_calib_none.csv $GW/data/G12/g12_s${S}_calib_none.csv > /dev/null || { echo "PULL FAILED $S"; exit 1; }
  (cd $G && python kp_cl_export.py data/G12/g12_s${S}_calib_none.csv data/G12/params_s$S.txt) >> $G/data/G12/export_log.txt 2>&1 \
    || { echo "EXPORT FAILED $S"; exit 1; }
  "$ADB" push $GW/data/G12/params_s$S.txt /home/arduino/ros/g12/params_s$S.txt > /dev/null || { echo "PUSH FAILED $S"; exit 1; }
  echo "seed $S params exported $(date)"
  "$ADB" shell "cd ~/ros && setsid nohup ./g12_phase.sh val $S > g12_val_$S.log 2>&1 < /dev/null &"
  until "$ADB" shell "test -f ~/ros/g12/phase_val_$S.done"; do sleep 20; done
  echo "seed $S val done $(date)"
done
"$ADB" pull /home/arduino/ros/g12 $GW/data/G12_board_raw > /dev/null; cp $G/data/G12_board_raw/g12_s*_val_*.csv $G/data/G12/
echo ALL_DONE
