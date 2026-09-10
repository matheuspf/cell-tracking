"""Read-only job/status watch, plus an observed CPU scheduling receipt."""
import time
import psutil
from .common import *


def run():
    last_scores=None
    cpu_samples=[]
    while True:
        current=read(OUT/'resource_current.json')
        processes=list(psutil.process_iter(['pid','ppid','cmdline','status']))
        by_pid={p.pid:p for p in processes}
        roots=[p for p in processes if 'image_native_tracking_v5.predict_batch' in (p.info['cmdline'] or [])
            and p.info['ppid'] in by_pid and 'image_native_tracking_v5.predict_batch' not in (by_pid[p.info['ppid']].info['cmdline'] or [])]
        children={}
        for p in roots:
            try:
                kids={q.pid for q in p.children() if q.status()!=psutil.STATUS_ZOMBIE}
                if len(kids)==10:children=kids;break
            except psutil.NoSuchProcess:pass
        observed={p['pid'] for p in current['processes']}
        if children and children<=observed and (not cpu_samples or cpu_samples[-1]['at']!=current['at']):
            cpu_samples.append(dict(at=current['at'],summed_RSS_gib=current['summed_process_rss_gib'],
                GPU_gib=current['gpu_mib']/1024,free_gib=current['free_gib'],pool_workers=10,
                pool_RSS_gib=sum(p['rss_gib'] for p in current['processes'] if p['pid'] in children)))
            write(OUT/'CPU_ten_worker_validation.json',dict(at=now(),observed=True,samples=len(cpu_samples),
                max_observed_summed_RSS_gib=max(r['summed_RSS_gib'] for r in cpu_samples),
                max_observed_GPU_gib=max(r['GPU_gib'] for r in cpu_samples),
                minimum_observed_free_gib=min(r['free_gib'] for r in cpu_samples),records=cpu_samples,
                scope='Sampled ten-worker production batches; all model and solver settings unchanged.'))
        native={p.stem:read(p) for p in (OUT/'models/native').glob('*.json') if not read(p).get('tiny')}
        progress={r['key']:f"{r['steps']}/{r['target_steps']}" for p in (OUT/'training').glob('*progress.json')
            if not (r:=read(p))['key'].endswith('_tiny') and r['key'] not in native}
        prediction_counts={p.name:len(list(p.glob('*.json'))) for p in (OUT/'prediction_receipts').iterdir()}
        evaluated={p.name:len(list(p.glob('*.json'))) for p in (OUT/'evaluation').iterdir()}
        failed={name:r for name,r in read(OUT/'supervisor_state.json').get('jobs',{}).items() if r['state']=='failed'}
        state=dict(at=now(),native_fits=len(native),training=progress,decoded=prediction_counts,
            scored={k:v for k,v in evaluated.items() if v==199},GPU_gib=round(current['gpu_mib']/1024,2),
            RSS_gib=round(current['summed_process_rss_gib'],2),free_gib=round(current['free_gib'],2),failed=failed)
        from .report import csv_rows
        scores={r['variant']:r['score'] for r in csv_rows('ablation_scores.csv') if r['embryo']=='pooled'}
        if scores!=last_scores:state['scores']=scores;last_scores=scores
        if (OUT/'fresh_validation_progress.json').exists():
            r=read(OUT/'fresh_validation_progress.json')
            state['fresh']={k:v for k,v in r.items() if k in ['image_clips_executed','compared_clips','status']}
        print(json.dumps(state),flush=True)
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(current['at'])).total_seconds()
        assert 0<=age<120,('Resource monitor stale',age)
        assert current['gpu_mib']<=20*1024 and current['summed_process_rss_gib']<=28 and current['free_gib']>=8,state
        assert not failed,failed
        complete=read(OUT/'execution_complete.json') if (OUT/'execution_complete.json').exists() else {}
        finalizing=any('image_native_tracking_v5.finalize' in (p.info['cmdline'] or []) for p in processes)
        if complete.get('complete') and not finalizing:return
        time.sleep(50)


if __name__=='__main__':run()
