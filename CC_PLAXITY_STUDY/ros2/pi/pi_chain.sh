#!/bin/bash
# Addendum L chain on the Pi 5 (idempotent, safe to re-launch). PI_JOBS = "kind:seed ..." (kind g14 | vo), PI_DONE = flag name.
# A seed is complete when its val CSV has 3001 lines and its log ends rc=0; partial seeds go to h/discarded_<ts>/ and are re-run.
# Logs CPU clock, SoC temperature and throttle state every 5 s (read-only sysfs). No spinners: the governor is already 'performance'.
. $HOME/ccpi/pi_env.sh; cd $CCPI
exec 7>$CCPI/tmp/pi_chain.lock; flock -n 7 || { echo "pi_chain already running"; exit 0; }
mkdir -p h; TS=$(date +%s)
done_ok() { [ -f $1/${2}_s${3}_val.csv ] && [ "$(wc -l < $1/${2}_s${3}_val.csv)" -eq 3001 ] && grep -q "rc=0" $1/log_s${3}_val.txt 2>/dev/null; }
for ks in $PI_JOBS; do
  k=${ks%%:*}; s=${ks##*:}; d=h/${k}_$s
  done_ok $d $k $s && continue
  if [ -d $d ]; then mkdir -p h/discarded_$TS; mv $d h/discarded_$TS/; echo "$(date) discarded partial $d" >> h/chain_log.txt; fi
  mkdir -p $d; md5sum run_pi.sh pi_chain.sh ws/src/cc_lidar_ros/src/*_tm.cpp ws/install/cc_lidar_ros/lib/cc_lidar_ros/*_tm > $d/md5.txt
  ( while [ ! -f $d/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') T=$(cat /sys/class/thermal/thermal_zone0/temp) fan=$(cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input 2>/dev/null)" >> $d/freq_log.txt; sleep 5; done ) &
  LG=$!
  echo "$(date) start $k $s" >> h/chain_log.txt
  for sp in calib val; do ./run_pi.sh $k $s $sp 3000 $d; done
  touch $d/done.flag; wait $LG 2>/dev/null
  echo "$(date) end $k $s $(tail -1 $d/log_s${s}_val.txt 2>/dev/null)" >> h/chain_log.txt
done
ALL=1; for ks in $PI_JOBS; do k=${ks%%:*}; s=${ks##*:}; done_ok h/${k}_$s $k $s || ALL=0; done
[ $ALL = 1 ] && touch h/${PI_DONE:-partL}.done || echo "$(date) chain finished with incomplete seeds" >> h/chain_log.txt
