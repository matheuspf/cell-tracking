"""Read-only resource monitor for the current study process trees."""
import csv
import shutil
import subprocess
import time
import psutil
from datetime import datetime
from .common import now,write_json

def run(ctx,args=None):
    stop=ctx.out/'resource_monitor.stop';start=time.monotonic()
    history=ctx.out/'resource_samples.csv'
    if history.exists():
        with history.open() as source:
            first=next(csv.DictReader(source),None)
        if first:
            start-=max(0,(datetime.fromisoformat(now())-datetime.fromisoformat(first['utc'])).total_seconds())
    with (ctx.out/'resource_samples.csv').open('a',newline='') as f:
        keys=['utc','elapsed_seconds','rss_bytes','processes','gpu_used_mib','gpu_util_percent','free_disk_bytes']
        w=csv.DictWriter(f,fieldnames=keys)
        if not f.tell():w.writeheader()
        while not stop.exists():
            processes={}
            for p in psutil.process_iter(['cmdline']):
                try:
                    command=' '.join(p.info['cmdline'] or [])
                    if any(marker in command for marker in ['strong_tracker_v3','strong-tracker-v3']):
                        processes[p.pid]=p
                        processes.update({c.pid:c for c in p.children(recursive=True)})
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            rss=0
            for p in processes.values():
                try:rss+=p.memory_info().rss
                except psutil.NoSuchProcess:pass
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            row=dict(utc=now(),elapsed_seconds=time.monotonic()-start,rss_bytes=rss,processes=len(processes),
                gpu_used_mib=float(gpu[0]),gpu_util_percent=float(gpu[1]),free_disk_bytes=shutil.disk_usage(ctx.out).free)
            w.writerow(row);f.flush()
            if rss>32*1024**3 or float(gpu[0])>20*1024 or row['free_disk_bytes']<18*1024**3:
                write_json(ctx.out/'resource_alert.json',row)
            time.sleep(10)
