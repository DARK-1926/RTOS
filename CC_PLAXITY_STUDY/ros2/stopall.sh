#!/bin/bash
for p in $(pgrep -f "ccrun/spin.py") $(pgrep -f "smoke2.sh") $(pgrep -f autoware_default) $(pgrep -f lidar_ros); do kill $p 2>/dev/null; done
sleep 1; echo "left: $(pgrep -fc 'ccrun/spin.py') spin, $(pgrep -fc autoware_default) ref"
