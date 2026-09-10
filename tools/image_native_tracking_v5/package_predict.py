"""Standalone image-to-submission entry; process-start sitecustomize supplies read/socket guards."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--v1',required=True,type=Path);p.add_argument('--v2',required=True,type=Path)
    p.add_argument('--source-model',required=True,choices=['44b6','6bba']);p.add_argument('--variant');p.add_argument('--also-variant',action='append',default=[])
    p.add_argument('--disable-new-heads',action='store_true');a=p.parse_args()
    package=Path(__file__).resolve().parents[2];images=a.images.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Output must be a new directory')
    for protected in [package,images,a.v1.resolve(),a.v2.resolve(),*[q.resolve() for q in images.glob('*.zarr')]]:
        if out==protected or out.is_relative_to(protected) or protected.is_relative_to(out):raise ValueError('Output overlaps protected inputs')
    os.environ.update(V5_V1=str(a.v1.resolve()),V5_V2=str(a.v2.resolve()),V5_OUTPUT=str(package),V5_STUDIES=str(out.parent),V5_FRESH_OUTPUT=str(out),V5_AUDIT_DIR=str(out/'read_audit'))
    out.mkdir(parents=True)
    import sitecustomize
    assert sitecustomize.INSTALLED_BEFORE_NUMERICAL
    selected='C0' if a.disable_new_heads else (a.variant or json.loads((package/'winning_config.json').read_text())['variant'])
    variants=list(dict.fromkeys([selected,*a.also_variant]));start=time.monotonic()
    with (out/'base.log').open('w') as log:
        subprocess.run([str(package/'base/run.sh'),'--python',sys.executable,'--images',str(images),'--output',str(out/'base'),
            '--v1',str(a.v1),'--v2',str(a.v2),'--source-model',a.source_model],check=True,stdout=log,stderr=subprocess.STDOUT)
    from image_native_tracking_v5.common import read,sha,arrays,save,write
    from image_native_tracking_v5 import native_adapter as native
    from image_native_tracking_v5.fresh_observations import observe
    from image_native_tracking_v5.banks import build
    from image_native_tracking_v5.score_models import encode_at_nodes,native_scores,hoct_scores
    from image_native_tracking_v5.hoct_adapter import load as hoct_load
    from image_native_tracking_v5.predict_batch import VARIANTS,transform
    from image_native_tracking_v5.deepcenter import load as dc_load,score_nodes
    from strong_tracker_v3.common import validate,graph_hash
    import torch,zarr,resource,numpy as np
    torch.set_num_threads(2)
    manifest=read(package/'manifest.json')
    new=[v for v in variants if v!='C0']
    for rel,h in manifest['files'].items():
        if not new and rel.split('/')[0] in ['models','calibration','python']:continue
        assert sha(package/rel)==h,rel
    source=a.source_model;joint=read(package/'calibration'/f'{source}_J.json') if new else None
    motion=read(package/'motion_config.json')[source] if new else None
    frozen=native.load() if new else None
    native_models={};heads={};general=None;ctc=None;dc_bundle=None;records=[]
    for image_path in sorted(images.glob('*.zarr')):
        name=image_path.stem;base=arrays(out/'base/predictions'/f'{name}.npz');shape=zarr.open_group(str(image_path),mode='r')['0'].shape;clip_start=time.monotonic()
        c=observe(frozen,base['nodes'],image_path) if new else None;banks={};model_score_cache={};dc=None
        for variant in variants:
            if variant=='C0':n,e=base['nodes'],base['edges'];receipt=dict(fallback=True,model_executed=False)
            else:
                cfg=VARIANTS[variant];family=cfg['family'];seed=cfg['seed'];expanded=cfg['pop']!='P0';pop='P1' if expanded else 'P0'
                mask=np.ones(len(c['nodes']),bool) if expanded else c['oldmask']
                if pop not in banks:
                    pairs,logits,br=build(frozen,c['nodes'][mask],base['edges'],c['features_source'][mask],c['features_target'][mask],motion['distance_um'])
                    banks[pop]=dict(pairs=pairs,native_logits=logits)
                common=banks[pop];cal=None;ms={}
                if family in ['N1','N2']:
                    key=(family,seed)
                    if key not in native_models:native_models[key]=native.load(package/'models/native'/f'{source}_{family}_{seed}.pt')
                    model=native_models[key]
                    if key not in model_score_cache:
                        fs,ft=encode_at_nodes(model,c['nodes'],image_path) if family=='N2' else (c['features_source'],c['features_target'])
                        model_score_cache[key]=dict(fs=fs,ft=ft)
                    cache=model_score_cache[key]
                    if pop not in cache:cache[pop]=native_scores(model,c['nodes'][mask],common['pairs'],cache['fs'][mask],cache['ft'][mask])
                    ms={pop:cache[pop]};cal=read(package/'calibration'/f'{source}_{family}_{seed}.json')['calibration']
                elif family.startswith('H'):
                    if family=='Hctc':
                        if ctc is None:ctc=hoct_load('ctc_v0')
                        hs,hr=hoct_scores(ctc,c['nodes'][mask],common['pairs'],c['properties'][mask],c['valid_region'][mask]);ms={'Hctc':hs['H0']}
                        cal=read(package/'calibration'/f'{source}_Hctc.json')['calibration']
                    else:
                        if general is None:general=hoct_load()
                        if seed not in heads:heads[seed]=torch.load(package/'models/hoct'/f'{source}_probe_{seed}.pt',weights_only=False,map_location='cpu')
                        key=('H',seed,pop)
                        if key not in model_score_cache:
                            hs,hr=hoct_scores(general,c['nodes'][mask],common['pairs'],c['properties'][mask],c['valid_region'][mask],[(f'H1_{seed}',heads[seed])])
                            model_score_cache[key]=hs
                        ms=model_score_cache[key];cal=read(package/'calibration'/f'{source}_H_{seed}.json')['models'][family]
                if cfg['pop']=='PDC' and dc is None:
                    if dc_bundle is None:dc_bundle=dc_load()
                    dc,_=score_nodes(*dc_bundle,c['nodes'],image_path)
                n,e,receipt=transform(c,base,common,ms,cfg,joint,cal,dc)
                receipt.update(model_executed=True,native_image_encoder_executed=family=='N2',full_frame_proposals_executed=True,
                    hoct_backbone_executed=family.startswith('H'),deepcenter_full_frames_executed=cfg['pop']=='PDC')
            validate(n,e,shape);save(out/'predictions'/variant/f'{name}.npz',nodes=n,edges=e)
            records.append(dict(dataset=name,variant=variant,source=source,graph_hash=graph_hash(n,e),nodes=len(n),edges=len(e),
                clip_elapsed_seconds=time.monotonic()-clip_start,**receipt))
    columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id'];index=0
    for variant in variants:
        dest=out/('submission.csv' if variant==selected else f'submission_{variant}.csv');index=0
        with dest.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(columns)
            for image_path in sorted(images.glob('*.zarr')):
                name=image_path.stem;g=arrays(out/'predictions'/variant/f'{name}.npz')
                for n in g['nodes']:writer.writerow([index,name,'node',*map(int,n),-1,-1]);index+=1
                for aa,bb in g['edges']:writer.writerow([index,name,'edge',-1,-1,-1,-1,-1,int(aa),int(bb)]);index+=1
    write(out/'inference_receipt.json',dict(selected=selected,variants=variants,records=records,seconds=time.monotonic()-start,
        cached_final_graph_reads=0,study_feature_cache_reads=0,source_labels_read=0,raw_images_read=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        guard_installed_before_numerical=sitecustomize.INSTALLED_BEFORE_NUMERICAL,guard_scope='Python audit file/socket hooks, not OS isolation'))
    sitecustomize.record()

if __name__=='__main__':main()
