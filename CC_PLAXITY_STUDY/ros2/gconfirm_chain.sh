#!/bin/bash
# Addendum G confirmatory chain: G13c (Autoware, 141-143) then G14 (real apps, 151-153). One script, sequential.
cd /home/arduino/ros
./g13_all.sh g13c 141 142 143
./g14_all.sh g14 151 152 153
touch /home/arduino/ros/gconfirm.done
