"""Refresh the offline progress dashboard while long measurements are running."""
import time,traceback
from .common import *
from .report import build

def run():
    while not (OUT/'execution_complete.json').exists() and not (OUT/'report_halt.json').exists():
        try:build()
        except Exception:traceback.print_exc()
        time.sleep(60)
    write(OUT/'live_stopped.json',dict(at=now(),stopped=True))

if __name__=='__main__':run()
