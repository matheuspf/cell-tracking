"""Append bounded-run resource measurements without changing any user process."""
import csv
import os
import subprocess
import sys
import time

import psutil

from .common import OUT,now,write_json


def output_bytes():
    total=0
    for root,dirs,files in os.walk(OUT,followlinks=False):
        for name in files:
            path=os.path.join(root,name)
            if not os.path.islink(path):
                try:total+=os.stat(path).st_size
                except FileNotFoundError:pass
    return total


def run(args):
    path=OUT/'resource_samples.csv';new=not path.exists()
    write_json(OUT/'resource_monitor_scope.json',dict(started=now(),executable=sys.executable,
               sampling_seconds=15,output_size_sampling_seconds=300,
               note='Summed RSS of processes using the isolated study interpreter; shared pages can be counted repeatedly. Whole-device VRAM includes unrelated desktop use. Prior appended samples used the main CLI/public/supplement process-name scope.'))
    size=output_bytes();i=0
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['utc','study_processes','summed_rss_gib','device_used_mib','gpu_utilization','output_bytes'])
        if new:writer.writeheader()
        while True:
            rss=[]
            for p in psutil.process_iter(['pid','cmdline','memory_info']):
                cmd=p.info['cmdline'] or []
                if p.pid!=os.getpid() and cmd and cmd[0]==sys.executable and p.info['memory_info']:
                    rss.append(p.info['memory_info'].rss)
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            if i%20==0:size=output_bytes()
            writer.writerow(dict(utc=now(),study_processes=len(rss),summed_rss_gib=sum(rss)/2**30,
                         device_used_mib=float(gpu[0]),gpu_utilization=float(gpu[1]),output_bytes=size));f.flush()
            if (OUT/'resource_monitor.stop').exists():break
            i+=1;time.sleep(15)
    write_json(OUT/'resource_monitor_stopped.json',dict(stopped=now(),pid=os.getpid(),csv_closed=True))
