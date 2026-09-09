"""Measured v2 resource use; does not terminate unrelated workloads."""
import csv
import shutil
import subprocess
import time

import psutil

from .common import OUT, now, write_json


def run(args):
    stop=OUT/'resource_monitor.stop'
    start=time.monotonic()
    with (OUT/'resource_samples.csv').open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['utc','elapsed_seconds','rss_bytes','processes','gpu_used_mib','gpu_util_percent','free_disk_bytes'])
        if not f.tell():w.writeheader()
        while not stop.exists():
            processes=[]
            for p in psutil.process_iter(['cmdline','memory_info']):
                try:
                    if 'strong_tracker_v2' in ' '.join(p.info['cmdline'] or []):
                        processes.append(p)
                        processes.extend(p.children(recursive=True))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            rss=sum(p.memory_info().rss for p in {p.pid:p for p in processes}.values() if p.is_running())
            s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            free=shutil.disk_usage(OUT).free
            w.writerow(dict(utc=now(),elapsed_seconds=time.monotonic()-start,rss_bytes=rss,processes=len(processes),
                gpu_used_mib=float(s[0]),gpu_util_percent=float(s[1]),free_disk_bytes=free));f.flush()
            if rss>128*1024**3 or float(s[0])>20*1024 or free<20*1024**3:
                write_json(OUT/'resource_limit_alert.json',dict(utc=now(),rss_bytes=rss,gpu_used_mib=float(s[0]),free_disk_bytes=free))
            time.sleep(10)
