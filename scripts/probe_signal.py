"""Does a DataSphere cancel reach the Python process through `timeout` (graceful shutdown)?

Run: python scripts/ds.py run jobs/probe-signal.yaml, wait for "heartbeat", then
python scripts/ds.py cancel <job_id>. The CLI stops streaming at the cancel, so read the finish
time in `ds.py get <job_id>`: ~60 s after the cancel means SIGTERM reached Python and DataSphere
waited (graceful shutdown); ~2 min (the graceful timeout) means it did not; seconds: no grace.
Result 2026-10-08 (c1.4, jobs bt1t910p6a2uc0vj2em1, bt1q3dh8vu9dhto9m2so): finished ~20 s after
the cancel with both a 5 s and a 60 s handler, so DataSphere does not wait for a graceful exit.
"""

import signal
import sys
import time


def handle(signum, _frame):
    print(f"received {signal.Signals(signum).name}: saving would happen now", flush=True)
    time.sleep(60)  # the job's finish time tells: ~60 s after cancel = signal reached Python
    print("clean exit after the signal", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle)
signal.signal(signal.SIGINT, handle)
for i in range(120):
    print(f"heartbeat {i}", flush=True)
    time.sleep(10)
print("no signal within 20 minutes", flush=True)
