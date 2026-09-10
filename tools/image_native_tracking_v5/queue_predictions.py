"""Bounded pipeline for registered complete graph families, independent of target outcomes."""
import time,subprocess,sys
from .common import *

def call(module,args,label):
    with (OUT/'logs'/f'{label}.log').open('a') as log:
        subprocess.run([sys.executable,'-m',f'image_native_tracking_v5.{module}',*args],check=True,stdout=log,stderr=subprocess.STDOUT)

def banks():
    from .banks import cached
    from . import native_adapter as native
    model=native.load()
    for row in inventory():
        name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6'
        while not (OUT/'observations'/f'{name}.npz').exists():time.sleep(10)
        for expanded in [False,True]:cached(model,name,source,expanded)
        print('target common banks',name,flush=True)

def controls():
    for source in ['44b6','6bba']:call('joint_calibration',[source],f'{source}_joint_calibration')
    variants=['J_native_frozen','P_union_native_J','P_DC_native_J','P_image_ablation']
    call('predict_batch',['decode','--variants',*variants],'decode_controls')
    call('evaluate',variants,'evaluate_controls')

def native(stage,seed):
    while any(not (OUT/'models/native'/f'{source}_{stage}_{seed}.json').exists() for source in ['44b6','6bba']):time.sleep(15)
    with gpu_aux():call('predict_batch',['native','--stage',stage,'--seed',str(seed)],f'score_{stage}_{seed}')
    if stage=='N1':variants=['N_head_J']
    elif seed==20260910:variants=['N_backbone_J','P_union_N_J','P_DC_N_J']
    else:variants=['N_final_seed2','P_final_seed2','P_DC_final_seed2']
    call('predict_batch',['decode','--variants',*variants],f'decode_{stage}_{seed}')
    call('evaluate',variants,f'evaluate_{stage}_{seed}')

def hoct():
    while any(not (OUT/'models/hoct'/f'{source}_probe_{seed}.json').exists() for source in ['44b6','6bba'] for seed in [20260910,314159]):time.sleep(15)
    with gpu_aux():
        for source in ['44b6','6bba']:call('calibrate_ctc',[source],f'{source}_ctc_calibration')
        call('predict_batch',['hoct'],'score_H')
    variants=['H_general_J','H_probe_J','H_probe_native_J','H_final_seed2']
    call('predict_batch',['decode','--variants',*variants],'decode_H')
    call('evaluate',variants,'evaluate_H')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['banks','controls','native','hoct']);p.add_argument('--stage',default='N2');p.add_argument('--seed',type=int,default=20260910)
    a=p.parse_args()
    if a.action=='banks':banks()
    elif a.action=='controls':controls()
    elif a.action=='native':native(a.stage,a.seed)
    else:hoct()
