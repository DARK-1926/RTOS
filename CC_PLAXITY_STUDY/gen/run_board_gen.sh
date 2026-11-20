#!/bin/sh
# G-study board queue. G1 and G2 with the clock held (SCHED_IDLE spinners), G4x with schedutil (spinners off).
cd /home/arduino/ccgen; mkdir -p out
md5sum cc_general.py > out/md5_board.txt
spin_on()  { for c in 0 1 2 3; do python3 /home/arduino/ccrun/spin.py $c & done; sleep 2; }
spin_off() { pkill -f "python3 /home/arduino/ccrun/sp""in.py"; sleep 2; }
( while [ ! -f out/done.flag ]; do echo "$(date +%s) $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' ' ') spin=$(ps aux | grep -c '[s]pin.py')" >> out/freq_log.txt; sleep 5; done ) &
run() { echo "$1 seed $2 start $(date)" >> out/run_log.txt; shift 0; }
spin_on
for s in 11 12 13 14 15; do echo "G1 $s start $(date)" >> out/run_log.txt
  python3 cc_general.py --workload lidar --hog mem --lidar_dir lidar_xyz --calib_frames 2400 --val_frames 1600 --period_ms 150 --seed $s --out out/G1_lidar_mem_held_s$s.csv > out/log_G1_s$s.txt 2>&1
  echo "G1 $s end rc=$? $(date)" >> out/run_log.txt; done
for s in 21 22 23; do echo "G2 $s start $(date)" >> out/run_log.txt
  python3 cc_general.py --workload synthetic --hog cpu --calib_frames 3000 --val_frames 2000 --period_ms 100 --seed $s --out out/G2_synth_cpu_held_s$s.csv > out/log_G2_s$s.txt 2>&1
  echo "G2 $s end rc=$? $(date)" >> out/run_log.txt; done
spin_off
for s in 41 42; do echo "G4x $s start $(date)" >> out/run_log.txt
  python3 cc_general.py --workload lidar --hog mem --lidar_dir lidar_xyz --calib_frames 2400 --val_frames 1600 --period_ms 150 --seed $s --out out/G4x_lidar_mem_schedutil_s$s.csv > out/log_G4x_s$s.txt 2>&1
  echo "G4x $s end rc=$? $(date)" >> out/run_log.txt; done
echo ALLDONE > out/done.flag
