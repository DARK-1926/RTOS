#!/bin/bash
# Laptop-side watchdog for the Pi 5 chain (as h_watch.sh v2): every 60 s, timeout on every ssh call.
# Usage: bash pi_watch.sh <done_flag_name> "<kind:seed ...>"
PART=$1; JOBS=$2
S="ssh -i $HOME/.ssh/id_ed25519_pi -o BatchMode=yes -o ConnectTimeout=10 prollysamz@10.0.9.145"
while true; do
  st=$(timeout 30 $S "[ -f ~/ccpi/h/$PART.done ] && echo DONE || { pgrep -f '[p]i_chain[.]sh' >/dev/null && echo RUN || echo IDLE; }" 2>/dev/null)
  case "$st" in
    DONE) echo "DONE $PART $(date)"; exit 0 ;;
    IDLE) echo "chain not running at $(date): launching"
          timeout 30 $S "cd ~/ccpi && PI_DONE=$PART PI_JOBS='$JOBS' setsid nohup ./pi_chain.sh >> h_chain_stdout.log 2>&1 < /dev/null &" ;;
    RUN) ;;
    *) echo "Pi unreachable at $(date)" ;;
  esac
  sleep 60
done
