"""Refresh the offline progress dashboard while long measurements are running."""
import time,traceback
from .common import *
from .report import build

def run():
    while not (OUT/'execution_complete.json').exists():
        try:build()
        except Exception:traceback.print_exc()
        time.sleep(60)

if __name__=='__main__':run()
