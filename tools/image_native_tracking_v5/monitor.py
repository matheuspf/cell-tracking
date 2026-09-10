import subprocess,time,psutil
from .common import *

def run():
    while True:
        processes=[];tracked={}
        for p in psutil.process_iter(['pid','cmdline','memory_info','num_threads','cpu_times']):
            try:
                cmd=p.info['cmdline'] or []
                modules=[x for x in cmd if x.startswith('image_native_tracking_v5.')]
                if modules and modules[0]!='image_native_tracking_v5.monitor':
                    tracked[p.pid]=(p,modules[0].split('.')[-1])
                    for child in p.children(recursive=True):tracked.setdefault(child.pid,(child,'inference_or_worker_child'))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        for pid,(p,module) in tracked.items():
            try:processes.append(dict(pid=pid,rss_gib=p.memory_info().rss/2**30,threads=p.num_threads(),cpu_seconds=sum(p.cpu_times()[:2]),module=module))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        record=dict(at=now(),gpu_mib=int(gpu[0]),gpu_util_percent=int(gpu[1]),temperature_c=int(gpu[2]),
            summed_process_rss_gib=sum(p['rss_gib'] for p in processes),free_gib=shutil.disk_usage(ROOT).free/2**30,processes=processes,
            RSS_scope='all v5 Python roots and recursive children, deduplicated; early samples tracked named v5 roots only')
        p=OUT/'resources.jsonl'
        with p.open('a') as f:f.write(json.dumps(record)+'\n')
        write(OUT/'resource_current.json',record)
        if (OUT/'execution_complete.json').exists():return
        time.sleep(30)

if __name__=='__main__':run()
