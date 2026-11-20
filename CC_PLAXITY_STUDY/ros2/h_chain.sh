#!/bin/bash
# Addendum H board chain (idempotent; safe to re-launch after a reboot).
# For each "kind:seed": complete = val CSV has 3001 lines AND its log ends rc=0. Otherwise any partial files of that seed are moved to
# h/discarded_<ts>/ (never analysed) and the seed is re-run from calibration. Kinds: g13 (Autoware), g14 (real apps), vo (EuRoC, Part 2).
cd /home/arduino/ros
exec 7>/tmp/h_chain.lock; flock -n 7 || { echo "h_chain already running"; exit 0; }
mkdir -p h; TS=$(date +%s)
for ks in ${H_JOBS:-g13:161 g13:162 g13:163 g14:171 g14:172 g14:173}; do
  k=${ks%%:*}; s=${ks##*:}; d=h/${k}_$s
  if [ -f $d/${k}_s${s}_val.csv ] && [ "$(wc -l < $d/${k}_s${s}_val.csv)" -eq 3001 ] && grep -q "rc=0" $d/log_s${s}_val.txt 2>/dev/null; then continue; fi
  if [ -d $d ]; then mkdir -p h/discarded_$TS; mv $d h/discarded_$TS/; echo "$(date) discarded partial $d" >> h/chain_log.txt; fi
  for i in $(seq 1 60); do flock -n /tmp/$( [ $k = vo ] && echo vo || echo $k ).lock true && break; sleep 2; done   # wait for the previous run's logger to release the lock
  echo "$(date) start $k $s" >> h/chain_log.txt
  case $k in g13) ./g13_all.sh $d $s ;; g14) ./g14_all.sh $d $s ;; vo) ./vo_all.sh $d $s ;; esac
  echo "$(date) end $k $s $(tail -1 $d/log_s${s}_val.txt 2>/dev/null)" >> h/chain_log.txt
done
ALL=1; for ks in ${H_JOBS:-g13:161 g13:162 g13:163 g14:171 g14:172 g14:173}; do k=${ks%%:*}; s=${ks##*:}; d=h/${k}_$s
  { [ -f $d/${k}_s${s}_val.csv ] && [ "$(wc -l < $d/${k}_s${s}_val.csv)" -eq 3001 ] && grep -q "rc=0" $d/log_s${s}_val.txt; } || ALL=0; done
[ $ALL = 1 ] && touch h/${H_DONE:-part1}.done || echo "$(date) chain finished with incomplete seeds; watchdog will relaunch" >> h/chain_log.txt
