#!/bin/bash
# Idempotent resume for the Addendum G G14 seeds after a reboot. Complete = val CSV with 3001 lines AND its log ends rc=0.
# Incomplete seeds: partial files -> g14/discarded_<timestamp>/ (never analysed), then re-run from calibration.
cd /home/arduino/ros
exec 8>/tmp/g14_autoresume.lock; flock -n 8 || { echo "autoresume already running"; exit 0; }
TS=$(date +%s); MISSING=()
for s in 152 153; do
  f=g14/g14_s${s}_val.csv; l=g14/log_s${s}_val.txt
  if [ -f $f ] && [ "$(wc -l < $f)" -eq 3001 ] && grep -q "rc=0" g14resume*/log_s${s}_val.txt $l 2>/dev/null; then continue; fi
  MISSING+=($s)
done
if [ ${#MISSING[@]} -eq 0 ]; then touch gconfirm.done; echo "all complete"; exit 0; fi
mkdir -p g14/discarded_$TS
for s in "${MISSING[@]}"; do mv g14/*s${s}_* g14/discarded_$TS/ 2>/dev/null; done
for d in g14resume*; do [ -d $d ] && for s in "${MISSING[@]}"; do mv $d/*s${s}_* g14/discarded_$TS/ 2>/dev/null; done; done
echo "autoresume $TS: re-running ${MISSING[*]} $(date)" >> g14/run_log.txt
OUTD=g14resume_$TS
./g14_all.sh $OUTD "${MISSING[@]}"
for s in "${MISSING[@]}"; do cp $OUTD/g14_s${s}_*.csv $OUTD/log_s${s}_*.txt g14/ 2>/dev/null; done
cat $OUTD/run_log.txt >> g14/run_log.txt
bash $0   # re-check; touches gconfirm.done when everything is complete
