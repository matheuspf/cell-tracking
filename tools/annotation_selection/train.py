"""Source-only fitting. Both directional models are locked before outer reveal."""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import OUT,SEED,digest,load_graph,now,read_json,sha,stage,write_json
from .features import FEATURES,GROUPS,assert_features


def configurations():
    result=[dict(id='constant',family='constant',group='all')]
    for group in GROUPS:
        for c in [.1,1.,10.]:
            result.append(dict(id=f'logistic_{group}_C{c:g}',family='logistic',group=group,C=c))
        for leaves in [7,15]:
            result.append(dict(id=f'hgb_{group}_leaf{leaves}',family='hgb',group=group,leaves=leaves))
    for suffix in ['ambiguous_excluded','high_confidence','shuffled','synthetic']:
        result.append(dict(id=f'hgb_all_leaf7_{suffix}',family='hgb',group='all',leaves=7,sensitivity=suffix))
    return result


def control_targets(labels,units,seed,kind):
    rng=np.random.default_rng(seed)
    order_units=np.argsort(units,kind='stable')
    groups=np.split(order_units,np.flatnonzero(np.diff(units[order_units]))+1)
    y=np.empty(len(labels),np.uint8)
    if kind=='synthetic':
        prevalence=float(labels.mean())
        for g in groups:y[g]=rng.random()<prevalence
    else:
        order=rng.permutation(len(groups))
        for g,j in zip(groups,order):
            src=labels[groups[j]]
            y[g]=src[np.minimum((np.arange(len(g))*len(src)/len(g)).astype(int),len(src)-1)]
    return y


def source_data(names):
    xs=[];ys=[];amb=[];units=[];refs=[];offset=0
    for name in names:
        b=load_graph(OUT/'baseline/clean'/f'{name}.npz')
        with np.load(OUT/'evaluation/membership'/f'{name}.npz') as f:
            labels=f['annotation_label'];ambiguous=f['ambiguous']
        xs.append(b['features']);ys.append(labels);amb.append(ambiguous)
        units.append(b['tracklet']+offset);offset+=int(b['tracklet'].max(initial=-1))+1
        refs.extend((name,i) for i in range(len(labels)))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(amb),np.concatenate(units),refs


def source_manifest(direction):
    return dict(source=direction['source'],outer=direction['outer'],source_samples=direction['source_samples'],
                baseline_hashes={n:sha(OUT/'baseline/clean'/f'{n}.npz') for n in direction['source_samples']},
                label_hashes={n:sha(OUT/'evaluation/membership'/f'{n}.npz') for n in direction['source_samples']},
                feature_columns=FEATURES,upstream_supervision='none',provenance_lane='clean',
                split_hash=sha(OUT/'fold_manifest.json'),preregistration_hash=sha(OUT/'preregistration.json'))


def check_source_direction(direction):
    if direction['source']==direction['outer'] or set(direction['source_samples'])&set(direction['outer_samples']):
        raise ValueError('Source/outer contamination')
    if any(n.split('_')[0]!=direction['source'] for n in direction['source_samples']):
        raise ValueError('Target labels in source manifest')
    if any(n.split('_')[0]!=direction['outer'] for n in direction['outer_samples']):
        raise ValueError('Wrong outer embryo')


def run(args):
    config=read_json(OUT/'preregistration.json');manifest=read_json(OUT/'fold_manifest.json')
    stage('S050','running',outer_results_revealed=False)
    for direction in manifest['directions']:
        check_source_direction(direction)
        source=direction['source']
        if args.source and source!=args.source:continue
        root=OUT/'models'/source;root.mkdir(parents=True,exist_ok=True)
        if (root/'tabular_complete.json').exists():
            print(f'Resume tabular {source}',flush=True);continue
        x,y,amb,units,refs=source_data(direction['source_samples'])
        rng=np.random.default_rng(SEED)
        sample=rng.choice(len(y),min(config['tabular_max_fit_candidates'],len(y)),replace=False)
        fits=[]
        for cfg in configurations():
            start=time.perf_counter();target=y;ix=sample
            sensitivity=cfg.get('sensitivity')
            if sensitivity=='ambiguous_excluded':ix=ix[~amb[ix]]
            if sensitivity=='high_confidence':ix=ix[(x[ix,5]>=.05)&(x[ix,10]>=3)]
            if sensitivity in ('shuffled','synthetic'):
                target=control_targets(y,units,SEED,sensitivity)
            cols=GROUPS[cfg['group']];assert_features(cols);col_ix=[FEATURES.index(c) for c in cols]
            if cfg['family']=='constant':model=DummyClassifier(strategy='prior')
            elif cfg['family']=='logistic':model=make_pipeline(StandardScaler(),LogisticRegression(C=cfg['C'],max_iter=500,random_state=SEED))
            else:model=HistGradientBoostingClassifier(max_leaf_nodes=cfg['leaves'],l2_regularization=1.,max_iter=150,early_stopping=False,random_state=SEED)
            model.fit(x[ix][:,col_ix],target[ix])
            artifact=dict(model=model,columns=cols,config=cfg)
            path=root/(cfg['id']+'.joblib');joblib.dump(artifact,path)
            fits.append(dict(**cfg,fit_candidates=len(ix),natural_source_candidates=len(y),sampling_fraction=len(sample)/len(y),
                             sampling='uniform natural-prevalence; no class weighting',seconds=time.perf_counter()-start,
                             sha256=sha(path),training_index_hash=digest(ix.tolist())))
            print(f'Trained {source} {cfg["id"]} ({len(ix)} observations, {fits[-1]["seconds"]:.1f}s)',flush=True)
        write_json(root/'tabular_complete.json',dict(created=now(),manifest=source_manifest(direction),fits=fits))
    stage('S050','complete',directions=len(manifest['directions']),models_per_direction=len(configurations()),selection='fixed; source inner grouping unavailable')


def image(args):
    import torch
    from .image_model import PatchModel,predict_image
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('RTX4090/CUDA unavailable')
    config=read_json(OUT/'preregistration.json');manifest=read_json(OUT/'fold_manifest.json')
    stage('S060','running',gpu=torch.cuda.get_device_name(),outer_results_revealed=False)
    for direction in manifest['directions']:
        check_source_direction(direction)
        source=direction['source']
        if args.source and source!=args.source:continue
        root=OUT/'models'/source
        if (root/'image_complete.json').exists():continue
        x,y,amb,units,refs=source_data(direction['source_samples'])
        rng=np.random.default_rng(SEED)
        selected=rng.choice(len(y),min(config['image_fit_candidates'],len(y)),replace=False)
        start_load=time.perf_counter()
        cached={n:np.load(OUT/'patches'/f'{n}.npy',mmap_mode='r') for n in direction['source_samples']}
        patches=np.stack([cached[refs[i][0]][refs[i][1]] for i in selected]).astype(np.float32)/255.
        xx=x[selected];yy=y[selected].astype(np.float32)
        image_mean=float(patches.mean());image_std=max(float(patches.std()),.01)
        tab_mean=xx.mean(axis=0);tab_std=np.maximum(xx.std(axis=0),.01)
        load_seconds=time.perf_counter()-start_load
        image_tensor=torch.from_numpy((patches-image_mean)/image_std)
        tab_tensor=torch.from_numpy((xx-tab_mean)/tab_std)
        targets=torch.from_numpy(yy)
        fits=[]
        for seed in config['image_seeds']:
            for plus_tabular in [False,True]:
                torch.manual_seed(seed);np.random.seed(seed)
                model_id=f'image_{"plus_tabular" if plus_tabular else "only"}_seed{seed}'
                model=PatchModel(len(FEATURES) if plus_tabular else 0).cuda()
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                epochs=[];bs=config['image_batch_size']
                for ep in range(config['image_epochs']):
                    model.train();total=0.
                    order=torch.randperm(len(targets))
                    for ix in order.split(bs):
                        im=image_tensor[ix].cuda();tab=tab_tensor[ix].cuda();target=targets[ix].cuda()
                        if torch.rand(())<.5:im=im.flip(-1)
                        if torch.rand(())<.5:im=im.flip(-2)
                        shift=tuple(int(v) for v in torch.randint(-1,2,(2,)))
                        im=torch.roll(im,shift,(-2,-1))
                        opt.zero_grad(set_to_none=True)
                        logits=model(im,tab)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
                        loss.backward();opt.step();total+=float(loss.detach())*len(ix)
                    epochs.append(dict(epoch=ep+1,training_bce=total/len(targets)))
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                checkpoint=dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},plus_tabular=plus_tabular,
                                image_mean=image_mean,image_std=image_std,tab_mean=tab_mean.tolist(),tab_std=tab_std.tolist(),
                                seed=seed,epochs=config['image_epochs'],model_id=model_id)
                path=root/(model_id+'.pt');torch.save(checkpoint,path)
                fits.append(dict(id=model_id,seed=seed,epochs=epochs,fit_candidates=len(selected),gpu_seconds=seconds,
                                 peak_vram_bytes=torch.cuda.max_memory_allocated(),observations_per_second=len(selected)*config['image_epochs']/seconds,
                                 loader_seconds=load_seconds,checkpoint_choice='fixed last epoch; no target/inner outcomes used',sha256=sha(path)))
                print(f'GPU {source} {model_id}: {seconds:.1f}s, {fits[-1]["peak_vram_bytes"]/2**20:.0f} MiB',flush=True)
                del model,opt;torch.cuda.empty_cache()
        write_json(root/'image_complete.json',dict(created=now(),manifest=source_manifest(direction),fits=fits))
    stage('S060','complete',gpu='RTX4090',directions=len(manifest['directions']),image_fits_per_direction=4)


def freeze(args):
    manifest=read_json(OUT/'fold_manifest.json')
    files={}
    for d in manifest['directions']:
        root=OUT/'models'/d['source']
        if not (root/'tabular_complete.json').exists() or not (root/'image_complete.json').exists():
            raise ValueError('Both directions must finish tabular and image fits before lock')
        for p in sorted(root.iterdir()):files[str(p.relative_to(OUT))]=sha(p)
    data=dict(created=now(),models=files,preregistration_hash=sha(OUT/'preregistration.json'),
              split_hash=sha(OUT/'fold_manifest.json'),selection=dict(model='hgb_all_leaf7',policy='membership_tracklets',keep=.9),
              source_selection_available=False,reason='No source inner groups certified; preregistered fixed-setting fallback',
              outer_scores_revealed=False)
    write_json(OUT/'model_lock.json',data,immutable=True)
    stage('S080','models_locked_prediction_verification_pending',directions=2)
