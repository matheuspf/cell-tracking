import subprocess,time,psutil
from .common import *

def run():
    while True:
        processes=[]
        for p in psutil.process_iter(['pid','cmdline','memory_info','num_threads','cpu_times']):
            try:
                cmd=' '.join(p.info['cmdline'] or [])
                if '-m image_native_tracking_v5.' in cmd and 'monitor' not in cmd:
                    processes.append(dict(pid=p.pid,rss_gib=p.info['memory_info'].rss/2**30,threads=p.info['num_threads'],
                        cpu_seconds=sum(p.info['cpu_times'][:2]),module=cmd.split('image_native_tracking_v5.')[-1].split()[0]))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        record=dict(at=now(),gpu_mib=int(gpu[0]),gpu_util_percent=int(gpu[1]),temperature_c=int(gpu[2]),
            summed_process_rss_gib=sum(p['rss_gib'] for p in processes),free_gib=shutil.disk_usage(ROOT).free/2**30,processes=processes)
        p=OUT/'resources.jsonl'
        with p.open('a') as f:f.write(json.dumps(record)+'\n')
        write(OUT/'resource_current.json',record)
        if (OUT/'execution_complete.json').exists():return
        time.sleep(30)

if __name__=='__main__':run()
