#!/bin/bash
# Laptop-side driver for Addendum P on the UNO Q (adb): calibration -> export heavy/light params -> val heavy -> val light -> pull.
# Usage: bash uq_cl_driver.sh <data_subdir> "<seeds>" <ncal> <nrand> <nval> <q_light>
SUB=$1; SEEDS=$2; NC=$3; NR=$4; NV=$5; QL=$6
export MSYS_NO_PATHCONV=1
A="C:/Users/mohit/AppData/Local/Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"
G=/d/RTKK/CC_PLAXITY_STUDY/gen; D=$G/data/$SUB; mkdir -p $D; DW=$(cygpath -m $D)
sh_() { timeout 40 "$A" shell "$1"; }
running() { sh_ "ps aux | grep -c '[u]q_cl_phase'" 2>/dev/null | tr -d '\r'; }
launch() { sh_ "cd /home/arduino/ros/uq && setsid nohup ./uq_cl_phase.sh $2 $1 $3 > /dev/null 2>&1 < /dev/null &"; }
wait_flag() {  # seed phase flagname args...
  until sh_ "test -f /home/arduino/ros/p/s$1/$3" 2>/dev/null; do
    r=$(running); if [ "$r" = "0" ]; then echo "phase $2 of seed $1 not running and not done: (re)launching $(date)"; launch $1 $2 "$4"; fi
    sleep 60; done; }
pull() { "$A" pull /home/arduino/ros/p/s$1/$2 "$DW/$2" > /dev/null || { echo "PULL FAILED $2"; exit 1; }
  r=$(sh_ "md5sum /home/arduino/ros/p/s$1/$2" | cut -c1-32); l=$(md5sum $D/$2 | cut -c1-32)
  echo "$2 $r $l $([ -n "$r" ] && [ "$r" = "$l" ] && echo OK || echo MISMATCH)" >> $D/md5_data.txt; }
for X in $SEEDS; do
  sh_ "mkdir -p /home/arduino/ros/p/s$X"
  wait_flag $X cal phase_cal_$NR.done "$NC $NR"
  pull $X cl_s${X}_calib_none.csv; pull $X cl_s${X}_calib_random.csv
  for L in heavy light; do q=0.85; [ $L = light ] && q=$QL
    (cd $G && python kp_cl3_export.py data/$SUB/cl_s${X}_calib_none.csv data/$SUB/cl_s${X}_calib_random.csv data/$SUB/params_s${X}_$L.txt $q) >> $D/export_log.txt 2>&1 || { echo "EXPORT FAILED $X $L"; exit 1; }
    "$A" push "$DW/params_s${X}_$L.txt" /home/arduino/ros/p/s$X/params_s${X}_$L.txt > /dev/null || { echo "PUSH FAILED"; exit 1; }
    echo "seed $X $L params: $(tail -n 1 $D/export_log.txt) $(date)"; done
  for L in heavy light; do
    wait_flag $X val phase_val_$L.done "$NV $L"
    for p in none kp kpmn kpm kpmwh kpmd toba random; do pull $X cl_s${X}_val_${L}_$p.csv; done
    echo "seed $X val $L done $(date)"; done
  for f in freq_log.txt run_log.txt md5_code.txt; do "$A" pull /home/arduino/ros/p/s$X/$f "$DW/${f%.txt}_s$X.txt" > /dev/null; done
done
echo ALL_DONE
