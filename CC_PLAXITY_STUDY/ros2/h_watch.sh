#!/bin/bash
# Laptop-side watchdog (v2). Every 60 s, with a timeout on every adb call (v1 hung on adb while the board was offline):
# if h/<part>.done exists -> exit; else if h_chain.sh is not running on the board -> (re)launch it (it is idempotent).
export MSYS_NO_PATHCONV=1
ADB="/c/Users/mohit/AppData/Local/Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"
PART=${1:-part1}; JOBS=${2:-}
while true; do
  timeout 30 "$ADB" wait-for-device >/dev/null 2>&1
  st=$(timeout 30 "$ADB" shell "test -f ~/ros/h/$PART.done && echo DONE || (pgrep -f [h]_chain[.]sh >/dev/null && echo RUN || echo IDLE)" 2>/dev/null | tr -d '\r')
  case "$st" in
    DONE) echo "DONE $PART $(date)"; exit 0 ;;
    IDLE) echo "chain not running at $(date): launching"
          timeout 30 "$ADB" shell "cd ~/ros && (VO_PERIOD=${VO_PERIOD:-100} H_DONE=$PART H_JOBS='$JOBS' setsid nohup ./h_chain.sh >> h/chain_stdout.log 2>&1 < /dev/null &)" ;;
    RUN) ;;
    *) echo "board unreachable $(date)" ;;
  esac
  sleep 60
done
