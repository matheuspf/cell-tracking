"""Read-only lightweight process/progress view; never opens images or labels."""
from collections import Counter
from datetime import datetime,timezone
import json
from .common import WORK,read,now
from .report import alive


def optional(path):
    try:return read(path)
    except FileNotFoundError:return None


def run():
    jobs=Counter();running=[];resources=[]
    for path in (WORK/'controller/jobs').glob('*.json'):
        r=optional(path)
        if r is None:continue
        if path.name.endswith('.resources.json'):
            resources.append(r);continue
        jobs[r['stage']+'/'+r['status']]+=1
        if r['status']!='running':continue
        item={k:r.get(k) for k in ('stage','source','seed','arm','clip','part','pid')}
        item['process_verified_alive']=alive(r.get('pid'))
        if r['stage']=='predict':
            progress=optional(WORK/'predictions'/r['arm']/r['source']/str(r['seed'])/r['clip']/'progress.json')
            if progress:
                item.update(frames=progress.get('frames'),nodes=progress.get('nodes'),edges=progress.get('edges'),
                    progress_age_seconds=round((datetime.now(timezone.utc)-datetime.fromisoformat(progress['updated_utc'])).total_seconds(),1))
            policy=optional(WORK/'predictions'/r['arm']/r['source']/str(r['seed'])/r['clip']/'policy_progress.json')
            if policy:item['policy_progress']=policy
        elif r['stage']=='prepare-actions':
            item['bank_progress']=optional(WORK/'banks'/r['source']/str(r['seed'])/r['part']/r['clip']/'progress.json')
        elif r['stage']=='calibration-predict':
            item['policy_progress']=optional(WORK/'fits'/r['source']/str(r['seed'])/'calibration'/r['arm']/r['clip']/'progress.json')
        elif r['stage']=='mine':
            item['policy_progress']=optional(WORK/'fits'/r['source']/str(r['seed'])/'compact/mining'/r['clip']/'progress.json')
        running.append(item)
    fits=[]
    for source in ('44b6','6bba'):
        for seed in (20260918,314159):
            for kind in ('upstream','compact'):
                folder=WORK/'fits'/source/str(seed)/kind
                final=optional(folder/'final.json');progress=optional(folder/'progress.json')
                if final or progress:
                    r=final or progress
                    fits.append(dict(source=source,seed=seed,kind=kind,status=r['status'],
                        step=r.get('updates',r.get('step')),process_verified_alive=alive((progress or {}).get('pid'))))
    prepared=[]
    for path in (WORK/'controller/source-prefetch').glob('*/*/progress.json'):
        r=optional(path)
        if r:prepared.append({**r,'process_verified_alive':alive(r.get('pid'))})
    heads=[]
    for path in (WORK/'controller/head-prefetch').glob('*/*/progress.json'):
        r=optional(path)
        if r:heads.append({**r,'process_verified_alive':alive(r.get('pid'))})
    return dict(utc=now(),pipeline=optional(WORK/'controller/pipeline.json'),jobs=dict(jobs),fits=fits,running_jobs=running,source_preparation=prepared,
        head_preparation=heads,
        sampled_study_rss_peak_gib=max((r['study_rss_peak_bytes']/2**30 for r in resources),default=None),
        sampled_host_available_min_gib=min((r['host_available_min_bytes']/2**30 for r in resources),default=None))


if __name__=='__main__':print(json.dumps(run(),indent=2))
