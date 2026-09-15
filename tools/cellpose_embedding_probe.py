"""Native image embeddings at new Cellpose points; frozen pilot diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from tools.cellpose_models_probe import REPO, WORK, OUT, SCREEN, GPU_LOCK, read, write, sha, deny_labels

NATIVE = Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/repo')
PRIMARY = NATIVE.parent / 'weights/unet_transformer/split_0/edge_predictor_best.pth'
SECONDARY = Path('/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth')
METHODS = ('incumbent', 'cpdino-vitb', 'cpdino', 'cpsam_v2', 'vitb+vitl', 'vitb+sam', 'all_cellpose', 'incumbent+vitb', 'incumbent+all_cellpose')
SPACING = np.array([1.625, .40625, .40625])
FULL_CP = REPO / 'work/cellpose-ultrack-20260914'
FULL_NATIVE = Path('/kaggle/working/cell-tracking/annotation-selection-v1/public_harmonic_full/inputs')


def panel():
    first=read(OUT/'pilot-plan.json'); second=read(OUT/'adjacent/pilot-plan.json')
    frames=sorted(first['frames']+second['frames'],key=lambda r:r['key'])
    pairs=[{'key':r['key'], 'dataset':r['dataset'], 'time':r['time'],
            'frames':[r,next(n for n in second['frames'] if n['dataset']==r['dataset'] and n['time']==r['time']+1)]}
           for r in first['frames']]
    return {'frames':frames,'pairs':pairs,'image_sha256':{**first['image_sha256'],**second['image_sha256']}}


def source_path(method, key):
    if method in ('incumbent', 'cpdino-vitb'):
        path=SCREEN / 'predictions' / ('cellpose_cpdino_vitb' if method == 'cpdino-vitb' else method) / f'{key}.npz'
        if path.exists():return path
        if method=='cpdino-vitb':return FULL_CP/'predictions/cellpose_cpdino_vitb'/f'{key}.npz'
        return FULL_NATIVE/f"pre_ilp_{key.rsplit('-t',1)[0]}.npz"
    path=WORK / 'predictions' / method / f'{key}.npz'
    return path if path.exists() else WORK/'adjacent/predictions'/method/f'{key}.npz'


def prepare():
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'pilot_plan_sha256': sha(OUT / 'pilot-plan.json'),
            'adjacent_plan_sha256':sha(OUT/'adjacent/pilot-plan.json'),
            'protocol_revision':'v1: correct nonadjacent t25/t75 pilot by adding t26/t76; v0 was never executed. No GT opened for revision.',
            'panel':panel(),
            'weights': {k: {'path':str(p), 'sha256':sha(p)} for k,p in [('primary',PRIMARY),('secondary',SECONDARY)]},
            'native_source': {str(p):sha(p) for p in sorted(NATIVE.rglob('*.py'))},
            'embeddings': '32-dimensional trained TemporalUNet features at each bank-specific center. Encoder is run once per frame pair and weight; node-transformer is run separately per bank with all its candidates.',
            'coordinates': 'Integer native exported point divided by (1,4,4); original floor-and-clamp feature indexing, sinusoidal positions, original-voxel edge coordinates. Strided native input and original full-movie quantile normalization.',
            'controls': ['negative physical distance', 'primary trained encoder + trained transformer', 'primary zero image features + same trained transformer', 'fixed notebook low-margin ensemble, image and zero-image', 'secondary trained encoder + trained transformer', 'secondary zero image features + same trained transformer'],
            'link_gate_um': 15., 'union_suppression_um': 2., 'methods': METHODS,
            'evaluation': 'Detector recall1-7um and close-pair recovery. Continuation top1 child ranking, top1 parent ranking for all GT edges, and true-division top2 recovery, using actual7um per-frame assignment. Unknown cells remain candidate competitors. No learned calibration or target fitting.',
            'scope': '24 frames, twelve adjacent pairs, six reused pilot clips. No complete-clip competition score. Encoders inherit incumbent exposure.',
            'script_sha256':sha(__file__)}
    path = OUT / 'embedding-plan.json'
    if path.exists():
        old = read(path)
        assert {k:v for k,v in old.items() if k!='created_utc'} == {k:v for k,v in plan.items() if k!='created_utc'}
    else:write(path,plan)
    print(json.dumps({'plan':str(path),'methods':len(METHODS)}),flush=True)


def union(base, additions):
    centers = list(base['centers_zyx']); scores = list(base['scores'])
    for added in additions:
        for index in np.argsort(-added['scores'],kind='stable'):
            p = added['centers_zyx'][index]
            if centers and np.min(np.linalg.norm((np.asarray(centers)-p)*SPACING,axis=1)) <= 2.:continue
            centers.append(p);scores.append(added['scores'][index])
    return {'centers_zyx':np.asarray(centers,np.float32).reshape(-1,3),'scores':np.asarray(scores,np.float32)}


def banks():
    deny_labels()
    plan=panel()
    assert sha(__file__)==read(OUT/'embedding-plan.json')['script_sha256']
    receipts=[]
    for row in plan['frames']:
        key=row['key']; values={}
        for method in METHODS[:4]:
            path=source_path(method,key)
            if method in ('cpdino','cpsam_v2'):
                receipt=read(path.with_suffix('.json'))
                origin_plan=OUT/'adjacent/pilot-plan.json' if '/adjacent/' in str(path) else OUT/'pilot-plan.json'
                assert receipt['plan_sha256']==sha(origin_plan) and receipt['prediction_sha256']==sha(path)
            elif method=='cpdino-vitb':
                receipt=read(path.with_suffix('.json'))
                assert receipt['input_sha256']==plan['image_sha256'][key]
            with np.load(path,allow_pickle=False) as f:
                if 'centers_zyx' in f:
                    values[method]={'centers_zyx':np.asarray(f['centers_zyx'],np.float32),'scores':np.asarray(f['scores'],np.float32)}
                else:
                    assert method=='incumbent'; chosen=f['coords'][:,0]==row['time']
                    values[method]={'centers_zyx':np.asarray(f['coords'][chosen,1:],np.float32),'scores':np.asarray(f['node_probabilities'][chosen],np.float32)}
            receipts.append({'source_only':True,'method':method,'key':key,'path':str(path),'sha256':sha(path)})
        for name,others in [('vitb+vitl',['cpdino']),('vitb+sam',['cpsam_v2']),('all_cellpose',['cpdino','cpsam_v2'])]:
            values[name]=union(values['cpdino-vitb'],[values[k] for k in others])
        values['incumbent+vitb']=union(values['incumbent'],[values['cpdino-vitb']])
        values['incumbent+all_cellpose']=union(values['incumbent'],[values[k] for k in ('cpdino-vitb','cpdino','cpsam_v2')])
        for method,value in values.items():
            shape=np.asarray(row['shape']);centers=np.clip(np.rint(value['centers_zyx']),0,shape-1).astype(np.int64)
            assert len(centers)==len(value['scores']) and np.isfinite(value['scores']).all()
            path=WORK/'banks'/method/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,centers_zyx=centers,centers_float_zyx=value['centers_zyx'],scores=value['scores'])
            receipts.append({'method':method,'key':key,'sha256':sha(path),'candidates':len(centers)})
    write(OUT/'banks-lock.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'embedding_plan_sha256':sha(OUT/'embedding-plan.json'),'files':receipts})
    print(json.dumps({'banks':sum(not r.get('source_only') for r in receipts),'candidates':sum(r.get('candidates',0) for r in receipts)}),flush=True)


def native_inference():
    deny_labels()
    import torch
    from tools.detector_screen.cellpose_adapter import gpu_lock
    for p in (NATIVE/'src', NATIVE/'scripts'):sys.path.insert(0,str(p))
    import predict_unet_transformer as prediction
    import train_unet_transformer as training
    torch.set_num_threads(2)
    plan=read(OUT/'embedding-plan.json'); pilot=panel()
    assert plan['script_sha256']==sha(__file__)
    for p,h in plan['native_source'].items():assert sha(p)==h
    bank_lock=read(OUT/'banks-lock.json')
    assert bank_lock['embedding_plan_sha256']==sha(OUT/'embedding-plan.json')
    for r in bank_lock['files']:
        path=Path(r['path']) if r.get('source_only') else WORK/'banks'/r['method']/f"{r['key']}.npz"
        assert sha(path)==r['sha256']
    receipts=[]
    for weight,record in plan['weights'].items():
        assert sha(record['path'])==record['sha256']
        model,window,ds=prediction.load_model(Path(record['path']),torch.device('cpu'))
        assert window==2 and tuple(ds)==(1,4,4)
        for pair in pilot['pairs']:
            dataset=pair['dataset']; rows=pair['frames']; pair_key=pair['key']
            assert len(rows)==2 and rows[1]['time']==rows[0]['time']+1
            meta=read(REPO/'data/train'/f'{dataset}.zarr/zarr.json')['attributes']['image_statistics']['quantiles']
            low,high=float(meta['0.001']),float(meta['0.999'])
            frames=[]
            for row in rows:
                assert sha(row['image_path'])==pilot['image_sha256'][row['key']]
                volume=np.load(row['image_path'],allow_pickle=False)[::1,::4,::4].astype(np.float32)
                frames.append(torch.from_numpy(np.maximum((volume-low)/(high-low+1e-6),0)))
            assert all(tuple(frame.shape)==(64,64,64) for frame in frames)
            began=time.perf_counter()
            with gpu_lock(GPU_LOCK) as queue:
                torch.cuda.reset_peak_memory_stats();model.to('cuda').eval()
                try:
                    with torch.inference_mode():
                        feature,_=model.encode(torch.stack(frames)[None].to('cuda'))
                        assert list(feature.shape)==[1,2,32,64,64,64]
                        for method in METHODS:
                            coords=[];pos=[];mask=[];embeddings=[]
                            for t,row in enumerate(rows):
                                with np.load(WORK/'banks'/method/f"{row['key']}.npz",allow_pickle=False) as f:xyz=np.asarray(f['centers_zyx'],np.float32)
                                grid=xyz/np.array(ds,np.float32)
                                c=torch.tensor(grid,device='cuda')[None]; m=torch.ones((1,len(c[0])),dtype=torch.bool,device='cuda')
                                p=torch.tensor(training.extract_pos_features(np.column_stack([np.full(len(grid),t),grid]),(2,64,64,64)),device='cuda')[None]
                                coords.append(c*torch.tensor(ds,device='cuda'));pos.append(p);mask.append(m)
                                embeddings.append(model._index_features(feature[:,t],c,m))
                            if all(e.shape[1] for e in embeddings):
                                logits=model.predict_edges(*embeddings,*coords,*pos,*mask)[0]
                                zero=model.predict_edges(*[torch.zeros_like(e) for e in embeddings],*coords,*pos,*mask)[0]
                            else:
                                logits=torch.empty((embeddings[0].shape[1],embeddings[1].shape[1]),device='cuda');zero=logits.clone()
                            arrays={'src_features':embeddings[0][0].cpu().numpy(),'dst_features':embeddings[1][0].cpu().numpy(),'image_logits':logits.cpu().numpy(),'zero_image_logits':zero.cpu().numpy()}
                            assert all(np.isfinite(a).all() for a in arrays.values())
                            path=WORK/'native'/weight/method/f'{pair_key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
                            receipts.append({'weight':weight,'method':method,'dataset':dataset,'key':pair_key,'sha256':sha(path),'shape':list(arrays['image_logits'].shape),'mean_abs_feature':float(np.mean(np.abs(arrays['src_features']))) if len(arrays['src_features']) else None,'mean_abs_logit_change_from_zero':float(np.mean(np.abs(arrays['image_logits']-arrays['zero_image_logits']))) if arrays['image_logits'].size else None})
                        torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()
                        del feature
                finally:model.to('cpu');torch.cuda.empty_cache()
            print(json.dumps({'pair':pair_key,'dataset':dataset,'weight':weight,'processing_seconds':time.perf_counter()-began-queue,'queue_seconds':queue,'peak_reserved_bytes':peak}),flush=True)
    write(OUT/'native-lock.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'embedding-plan.json'),'banks_lock_sha256':sha(OUT/'banks-lock.json'),'predictions':receipts})


def score_variants(arrays, distance):
    """Fixed controls plus the notebook's calibrated low-margin edge blend."""
    from scipy.special import softmax
    result={'distance':-distance}
    for weight in ('primary','secondary'):
        result[weight+'_image']=arrays[weight]['image_logits']
        result[weight+'_zero']=arrays[weight]['zero_image_logits']
    for kind in ('image','zero'):
        primary=result['primary_'+kind]; secondary=result['secondary_'+kind]
        if primary.size==0:
            result['ensemble_'+kind]=primary.copy(); continue
        pc=primary.mean(axis=0,keepdims=True); sc=secondary.mean(axis=0,keepdims=True)
        ratio=np.clip(np.maximum(primary.std(axis=0,keepdims=True),1e-4)/np.maximum(secondary.std(axis=0,keepdims=True),1e-4),.5,2.)
        calibrated=(secondary-sc)*ratio+pc
        weight=np.zeros((1,primary.shape[1]),np.float32)
        if primary.shape[0]>=2:
            probabilities=softmax(primary,axis=0)
            top2=np.sort(probabilities,axis=0)[-2:]
            uncertainty=np.clip((.35-(top2[1]-top2[0]))/.35,0,1)
            weight[0]=.15*uncertainty*(np.argmax(primary,axis=0)==np.argmax(calibrated,axis=0))
        result['ensemble_'+kind]=(1-weight)*primary+weight*calibrated
    return result


def evaluate():
    import logging,warnings
    logging.disable(logging.WARNING); warnings.filterwarnings('ignore')
    from scipy.spatial.distance import cdist
    from tools.detector_screen.evaluate import matches
    pilot=panel(); native=read(OUT/'native-lock.json'); bank_lock=read(OUT/'banks-lock.json')
    assert read(OUT/'embedding-plan.json')['script_sha256']==sha(__file__)
    assert native['plan_sha256']==sha(OUT/'embedding-plan.json') and native['banks_lock_sha256']==sha(OUT/'banks-lock.json')
    for r in bank_lock['files']:
        if not r.get('source_only'):assert sha(WORK/'banks'/r['method']/f"{r['key']}.npz")==r['sha256']
    for r in native['predictions']:assert sha(WORK/'native'/r['weight']/r['method']/f"{r['key']}.npz")==r['sha256']
    opening=datetime.now(timezone.utc).isoformat()
    gt=read(SCREEN/'evaluation/ground_truth.json')
    details=[]
    for method in METHODS:
        for pair in pilot['pairs']:
            dataset=pair['dataset']; rows=pair['frames']; centers=[]; maps=[]
            gnodes=np.asarray(gt['clips'][dataset]['nodes'],np.int64).reshape(-1,5)
            gedges=np.asarray(gt['clips'][dataset]['edges'],np.int64).reshape(-1,2)
            nodes=0; recalls={str(r):0 for r in range(1,8)}
            crowded={'close_gt_pairs':0,'close_pairs_both_at_3':0,'close_pairs_both_at_7':0,'gt_with_multiple_candidates_at_7':0,'duplicates_integer':0}
            for row in rows:
                with np.load(WORK/'banks'/method/f"{row['key']}.npz",allow_pickle=False) as f:c=np.asarray(f['centers_zyx'],np.int64)
                centers.append(c);truth=gnodes[gnodes[:,1]==row['time']];nodes+=len(truth)
                assignments={}
                for radius in range(1,8):
                    assignments[radius]=matches(c,truth,row['time'],float(radius))
                    recalls[str(radius)]+=len(assignments[radius])
                maps.append({v:k for k,v in assignments[7].items()})
                distances=cdist(truth[:,2:]*SPACING,c*SPACING)
                crowded['gt_with_multiple_candidates_at_7']+=int(((distances<=7).sum(axis=1)>1).sum())
                crowded['duplicates_integer']+=len(c)-len(np.unique(c,axis=0))
                td=cdist(truth[:,2:]*SPACING,truth[:,2:]*SPACING)
                for i,j in zip(*np.where(np.triu(td<=7,k=1))):
                    crowded['close_gt_pairs']+=1
                    for radius in (3,7):
                        ids=set(assignments[radius].values())
                        crowded[f'close_pairs_both_at_{radius}']+=int(truth[i,0] in ids and truth[j,0] in ids)
            times={int(r[0]):int(r[1]) for r in gnodes};children=defaultdict(list)
            for a,b in gedges:children[int(a)].append(int(b))
            relevant=[(int(a),int(b)) for a,b in gedges if times[int(a)]==rows[0]['time'] and times[int(b)]==rows[1]['time']]
            continuation=[(a,b) for a,b in relevant if len(children[a])==1]
            supported=[(maps[0][a],maps[1][b]) for a,b in continuation if a in maps[0] and b in maps[1]]
            supported_all=[(maps[0][a],maps[1][b]) for a,b in relevant if a in maps[0] and b in maps[1]]
            divisions=[(a,bs) for a,bs in children.items() if times[a]==rows[0]['time'] and len(bs)==2 and all(times[b]==rows[1]['time'] for b in bs)]
            supported_divisions=[(maps[0][a],[maps[1][b] for b in bs]) for a,bs in divisions if a in maps[0] and all(b in maps[1] for b in bs)]
            distance=cdist(centers[0]*SPACING,centers[1]*SPACING); gate=distance<=15.
            arrays={}
            for weight in ('primary','secondary'):
                with np.load(WORK/'native'/weight/method/f"{pair['key']}.npz",allow_pickle=False) as f:
                    arrays[weight]={k:f[k] for k in ('image_logits','zero_image_logits')}
            scores=score_variants(arrays,distance); metrics={}
            for name,score in scores.items():
                score=np.where(gate,score,-np.inf); correct=within=parent_correct=div_correct=0
                for a,b in supported:
                    if gate[a,b]:
                        within+=1;correct+=int(np.argmax(score[a])==b)
                for a,b in supported_all:
                    parent_correct+=int(gate[a,b] and np.argmax(score[:,b])==a)
                for a,bs in supported_divisions:
                    div_correct+=int(all(gate[a,b] for b in bs) and set(np.argsort(-score[a],kind='stable')[:2])==set(bs))
                metrics[name]={'correct_top1':correct,'supported_edges':len(supported),'true_edge_within_gate':within,'eligible_continuation_gt_edges':len(continuation),
                               'correct_parent_top1':parent_correct,'supported_all_edges':len(supported_all),'eligible_gt_edges':len(relevant),
                               'correct_division_top2':div_correct,'supported_divisions':len(supported_divisions),'eligible_gt_divisions':len(divisions)}
            details.append({'method':method,'dataset':dataset,'key':pair['key'],'embryo':dataset[:4],'frames':2,'gt_nodes':nodes,'candidates':sum(map(len,centers)),
                            'matches':recalls,'crowded':crowded,'eligible_gt_edges':len(relevant),'available_gt_edge_endpoints':len(supported_all),'association':metrics})
    groups=[]
    for embryo in ('pooled','44b6','6bba'):
        for method in METHODS:
            rs=[r for r in details if r['method']==method and (embryo=='pooled' or r['embryo']==embryo)]
            n=sum(r['gt_nodes'] for r in rs)
            groups.append({'embryo':embryo,'method':method,'frames':sum(r['frames'] for r in rs),'gt_nodes':n,
                'candidates':sum(r['candidates'] for r in rs),'matches':{str(k):sum(r['matches'][str(k)] for r in rs) for k in range(1,8)},
                'recall':{str(k):sum(r['matches'][str(k)] for r in rs)/n if n else None for k in range(1,8)},
                'crowded':{k:sum(r['crowded'][k] for r in rs) for k in rs[0]['crowded']},
                'association':{name:{field:sum(r['association'][name][field] for r in rs) for field in rs[0]['association'][name]} for name in rs[0]['association']}})
    write(OUT/'pilot-metrics.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'gt_opened_utc':opening,'gt_sha256':sha(SCREEN/'evaluation/ground_truth.json'),
        'native_lock_sha256':sha(OUT/'native-lock.json'),'groups':groups,'per_pair':details,
        'scope':'Exploratory, training-exposed 24-frame pilot on six clips/two embryos. No complete-clip competition score. Top1 ranking does not enforce consistent temporal assignments or detect divisions. Zero-image control preserves coordinates/positions and the trained head; it is not a separately recalibrated model.'})
    print(json.dumps([r for r in groups if r['embryo']=='pooled']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','banks','native','evaluate'));args=parser.parse_args()
    {'prepare':prepare,'banks':banks,'native':native_inference,'evaluate':evaluate}[args.action]()
