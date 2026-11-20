#!/bin/bash
# Laptop-side watcher: on a board reboot (uptime decreases), wait for boot and launch the idempotent board-side autoresume.
export MSYS_NO_PATHCONV=1
ADB="/c/Users/mohit/AppData/Local/Arduino15/packages/arduino/tools/adb/32.0.0/adb.exe"
last=0
while true; do
  if "$ADB" shell 'test -f ~/ros/gconfirm.done' 2>/dev/null; then echo "DONE $(date)"; exit 0; fi
  up=$("$ADB" shell 'cut -d. -f1 /proc/uptime' 2>/dev/null | tr -d '\r')
  if [[ "$up" =~ ^[0-9]+$ ]]; then
    if [ "$up" -lt "$last" ]; then
      echo "REBOOT detected at $(date) (uptime $up < $last); resuming in 60 s"
      sleep 60
      "$ADB" shell 'cd ~/ros && (setsid nohup ./g14_autoresume.sh >> g14_autoresume.log 2>&1 < /dev/null &)'
    fi
    last=$up
  fi
  sleep 30
done
