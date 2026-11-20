import os, sys
os.sched_setaffinity(0, {int(sys.argv[1])})
os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
while True:
    pass
