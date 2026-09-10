"""Predeclared density-stratified fresh-image follow-up within the local window."""
from .common import *

def lock():
    p=OUT/'additional_pilot_lock.json'
    if p.exists():return read(p)
    fixed={r['dataset'] for r in read(V3/'fresh_pilot_lock.json')['selected']};selected=[]
    for embryo in ['44b6','6bba']:
        counts=sorted((len(arrays(V3/'selected_predictions'/f'{r["dataset"]}.npz')['nodes']),r['dataset']) for r in inputs() if r['embryo']==embryo and r['dataset'] not in fixed)
        for quantile in [.5,.9]:
            count,name=counts[int((len(counts)-1)*quantile)]
            selected.append(dict(dataset=name,embryo=embryo,quantile=quantile,incumbent_nodes=count))
    result=dict(created=now(),selected=selected,rule='Median and 90th percentile frozen incumbent node count per embryo, excluding required pilots',
        target_scores_read=False,labels_read=False,prediction_policy_changes=False)
    write(p,result);return result

def run():
    import time
    from datetime import datetime
    from .delivery import pilots
    p=lock();schedule=OUT/'additional_pilot_schedule.json'
    if not schedule.exists():
        deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
        observed=read(OUT/'inference_receipt.json')['fresh_image_pilots'];predicted=[]
        for row in p['selected']:
            estimate=0.
            for pilot in observed:
                if not pilot['dataset'].startswith(row['embryo']):continue
                count=len(arrays(V3/'selected_predictions'/f'{pilot["dataset"]}.npz')['nodes'])
                estimate+=pilot['seconds']*max(1,row['incumbent_nodes']/count)*1.2+5
            predicted.append(dict(**row,estimated_seconds=estimate))
        available=deadline-time.time()-900;chosen=predicted
        if sum(r['estimated_seconds'] for r in chosen)>available:chosen=[r for r in predicted if r['quantile']==.9]
        if sum(r['estimated_seconds'] for r in chosen)>available:chosen=[]
        write(schedule,dict(created=now(),selected=chosen,all_candidates=predicted,available_seconds=available,
            decision_uses_runtime_and_frozen_density_only=True,final_validation_reserve_seconds=900))
    selected=read(schedule)['selected']
    if not selected:
        write(OUT/'additional_pilot_receipt.json',dict(completed=False,optional_skipped=True,reason='Measured runtime estimate would consume final validation reserve'));return
    pilots(OUT/'inference_package',[r['dataset'] for r in selected],'additional_pilot_receipt.json')
    receipt=read(OUT/'additional_pilot_receipt.json');receipt['schedule']=read(schedule);write(OUT/'additional_pilot_receipt.json',receipt)
