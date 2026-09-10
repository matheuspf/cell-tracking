"""Post-outcome W470 diagnostic; frozen weights, no calibration or policy changes."""
import pandas as pd
import torch
from .common import *
from .calibrate import sample,load_model,scores,diagnostics

def run():
    receipt=OUT/'rendered_holdout_receipt.json'
    if not read(OUT/'rendering_trial_receipt.json').get('primary_measured'):
        write(receipt,dict(completed=False,optional_skipped=True,reason='No completed C7 transfer comparison'));return
    if receipt.exists() and read(receipt).get('completed'):
        for name,digest in read(receipt)['model_sha256'].items():assert sha(OUT/'models'/f'{name}.pt')==digest
        return
    torch.set_num_threads(2);rows=[];hashes={}
    for source in ['44b6','6bba']:
        for partition in ['validation','test']:
            data=sample('rendered_zoo_'+source,'I',partition)
            for arm in ['C4','C7']:
                name=f'I_{arm}_{source}';m=load_model(name)
                metrics=diagnostics(scores(m,data),data['target'],1.)
                rows.append(dict(source=source,partition=partition,arm=arm,model=name,
                    temperature=1.,**{k:v for k,v in metrics.items() if k!='bins'}))
                hashes[name]=sha(OUT/'models'/f'{name}.pt');del m
                print(source,partition,arm,metrics['positive_pair_correct'],metrics['positive'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'rendered_holdout_diagnostics.csv',index=False)
    write(OUT/'rendered_holdout_receipt.json',dict(completed=True,created=now(),comparisons=len(rows),
        model_sha256=hashes,post_outcome_diagnostic=True,used_to_tune=False,optimizer_updates=0,
        calibration_or_threshold_changes=False,
        scope='Weak graph labels from purged time blocks of the same Zoo-fish acquisition; explicitly rendered images; conditional sampling, not biological prevalence'))

if __name__=='__main__':run()
