#!/bin/bash
# Resume after reboot #2: discard the partial seed-152 calibration, re-run G14 seeds 152-153, then signal completion.
cd /home/arduino/ros
mkdir -p g14/discarded_reboot2 && mv g14/*s152* g14/discarded_reboot2/ 2>/dev/null
echo "reboot during 152 calib; partial files moved to discarded_reboot2; resumed $(date)" >> g14/run_log.txt
./g14_all.sh g14resume 152 153
for f in g14resume/g14_s15[23]_*.csv; do cp $f g14/; done
cat g14resume/run_log.txt >> g14/run_log.txt
touch /home/arduino/ros/gconfirm.done
