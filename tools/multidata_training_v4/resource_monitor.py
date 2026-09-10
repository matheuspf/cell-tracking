"""Local read-only resource sampling while the authorized study is active."""
import subprocess
import time
from datetime import datetime
from .common import *

def run():
    import psutil
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    with (OUT/'resource_samples.jsonl').open('a') as f:
        while time.time()<deadline:
            processes=[]
            for proc in psutil.process_iter(['pid','cmdline','memory_info','cpu_times']):
                try:
                    cmd=' '.join(proc.info['cmdline'] or [])
                    if 'multidata_training_v4' not in cmd or proc.pid==os.getpid():continue
                    if proc.name() not in ['python','python3','python3.12']:continue
                    processes.append(dict(pid=proc.pid,rss_gib=proc.info['memory_info'].rss/2**30,
                        cpu_seconds=sum(proc.info['cpu_times'][:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(',')
            row=dict(created=now(),gpu_memory_mib=float(q[0]),gpu_utilization_percent=float(q[1]),gpu_power_watts=float(q[2]),
                processes=processes,total_observed_process_rss_gib=sum(p['rss_gib'] for p in processes))
            f.write(json.dumps(row)+'\n');f.flush()
            if (OUT/'status.json').exists() and read(OUT/'status.json').get('state')=='completed':break
            time.sleep(30)
