import os, sys
# Clock-holding spinner, as ccrun/spin.py, plus nice 19 so its CPU time is accounted as 'nice' in /proc/stat
# (SCHED_IDLE scheduling is unchanged; nice only affects accounting here). Used from Round G onward.
os.sched_setaffinity(0, {int(sys.argv[1])})
os.nice(19)
os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
while True:
    pass
