#!/bin/bash
# Laptop-side driver for Addendum R: start the three seeds on the Pi in one detached chain, wait, pull with md5.
S="ssh -i $HOME/.ssh/id_ed25519_pi -o BatchMode=yes -o ConnectTimeout=10 prollysamz@10.0.9.145"; SCP="scp -q -i $HOME/.ssh/id_ed25519_pi"
D=/d/RTKK/CC_PLAXITY_STUDY/gen/data/R; mkdir -p $D
timeout 30 $S "cd ~/ccpi && setsid nohup bash -c './pi_r_stress.sh 251 3000 base-idle r; ./pi_r_stress.sh 252 3000 idle-base r; ./pi_r_stress.sh 253 3000 base-idle r' > r_stdout.log 2>&1 < /dev/null &"
echo "launched $(date)"
until timeout 30 $S "test -f ccpi/r/s253/phase_r.done" 2>/dev/null; do sleep 60; done
for s in 251 252 253; do for a in base idle; do f=cl_s${s}_val_stress_$a.csv
  $SCP prollysamz@10.0.9.145:ccpi/r/s$s/$f $D/ || echo "PULL FAILED $f"
  r=$(timeout 30 $S "md5sum ccpi/r/s$s/$f" | cut -c1-32); l=$(md5sum $D/$f | cut -c1-32)
  echo "$f $r $l $([ -n "$r" ] && [ "$r" = "$l" ] && echo OK || echo MISMATCH)" >> $D/md5_data.txt; done
  for g in run_log md5_code freq_log; do $SCP prollysamz@10.0.9.145:ccpi/r/s$s/$g.txt $D/${g}_s$s.txt; done; done
echo ALL_DONE $(date)
