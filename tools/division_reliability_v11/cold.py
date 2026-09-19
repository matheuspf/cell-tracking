"""Renamed complete clips, image-only density strata, independent cold workers."""
from pathlib import Path
import sys,subprocess,shutil,hashlib,csv
from .common import REPO,DATA,WORK,RESULTS,Blocked,read,write,sha,now


def select():
    folder=WORK/'cold/selection';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():return read(folder/'receipt.json')
    split=read(WORK/'source_partitions.json');names=sorted(n for s in split.values() for part in ('fit','calibration') for n in s[part])
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[DATA/'train'/f'{n}.zarr' for n in names],outputs=[folder],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np,zarr
    from .data import normalize
    from .graphs import peaks
    rows=[]
    for name in names:
        image=zarr.open_array(str(DATA/'train'/f'{name}.zarr/0'),mode='r')
        counts=[len(peaks(normalize(np.asarray(image[t]))[0]-.5)[0]) for t in (0,49,99)]
        rows.append(dict(clip=name,embryo=name.split('_')[0],mean_local_maxima=sum(counts)/3,frame_counts=counts))
    chosen={}
    for embryo in ('44b6','6bba'):
        ss=sorted((r for r in rows if r['embryo']==embryo),key=lambda r:(r['mean_local_maxima'],r['clip']))
        maximum=max(r['mean_local_maxima'] for r in ss)
        crowded=min((r for r in ss if r['mean_local_maxima']==maximum),key=lambda r:r['clip'])
        chosen[embryo]=dict(median=ss[(len(ss)-1)//2]['clip'],crowded=crowded['clip'])
    result=dict(status='selected',chosen=chosen,inventory=rows,guard=guard,
        rule='Frames 0,49,99; fixed image-local 1/99 percentile normalization; coarse local maxima >0.5; physical NMS 3um; lower median and largest mean, lexical ties',
        labels_or_model_scores_used=False,finished_utc=now())
    write(folder/'receipt.json',result,immutable=True);return result


def run():
    from .freeze import verify
    frozen=verify();folder=WORK/'cold';folder.mkdir(parents=True,exist_ok=True)
    log=folder/'selection.log'
    with log.open('a') as f:
        subprocess.run([sys.executable,'-m','division_reliability_v11','cold-select'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    selected=read(folder/'selection/receipt.json');results=[]
    for source in ('44b6','6bba'):
        target='6bba' if source=='44b6' else '44b6'
        for seed in (20260918,314159):
            for arm in ('C00','C01','C11'):
                if f'{source}/{seed}/{arm}' not in frozen['models']:continue
                package=WORK/'packages'/source/str(seed)/arm
                for stratum,clip in selected['chosen'][target].items():
                    key=f'{source}/{seed}/{arm}/{clip}';verify(key)
                    renamed='clip_'+hashlib.sha256(('v11-cold:'+clip).encode()).hexdigest()[:12]
                    image=folder/'inputs'/f'{renamed}.zarr'
                    if not image.exists():
                        image.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(DATA/'train'/f'{clip}.zarr',image)
                    out=folder/'predictions'/source/str(seed)/arm/renamed;out.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-m','division_reliability_v11','infer','--package',str(package),
                             '--images',str(image),'--output',str(out),'--cold']
                    with (out/'worker.log').open('a') as f:code=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT).returncode
                    result=dict(source=source,target=target,seed=seed,arm=arm,stratum=stratum,original_clip=clip,renamed_clip=renamed,
                                status='failed',exit_code=code,worker_log_sha256=sha(out/'worker.log'),freeze_identity=frozen['identity'])
                    if code==0:
                        reference=WORK/'predictions'/arm/source/str(seed)/clip
                        old=read(reference/'receipt.json');new=read(out/'receipt.json')
                        with (reference/'submission.csv').open() as a,(out/'submission.csv').open() as b:
                            aa=csv.DictReader(a);bb=csv.DictReader(b);exact=True;rows=0
                            from itertools import zip_longest
                            for x,y in zip_longest(aa,bb):
                                if x is None or y is None:exact=False;break
                                y['dataset']=clip
                                if x!=y:exact=False;break
                                rows+=1
                        exact_graph=old['graph_hash']==new['graph_hash']
                        result.update(status='passed' if exact and exact_graph else 'mismatch',
                            complete_frames=new['frames'],graph_exact=exact_graph,csv_exact_after_dataset_rename=exact,
                            compared_csv_rows=rows,csv_bytes_differ_only_by_dataset_name=exact,
                            raw_image_frames_exact=old['frame_hashes']==new['frame_hashes'],guard=new['guard'],
                            gpu_lease_seconds=new['gpu_lease_seconds'],wall_seconds=new['wall_seconds'],
                            cold_csv_sha256=new['csv_sha256'],cold_graph_sha256=new['graph_sha256'],
                            inference_implementation_sha256=new['inference_implementation_sha256'])
                    write(out/'comparison.json',result);results.append(result)
                    write(folder/'progress.json',dict(completed=len(results),results=results,updated_utc=now()))
    result=dict(status='passed' if results and all(r['status']=='passed' for r in results) else 'failed',
        expected_runs=sum(2 for _ in frozen['models']),completed_runs=len(results),results=results,
        selection_sha256=sha(folder/'selection/receipt.json'),all_four_cells_tested=len({(r['source'],r['seed']) for r in results})==4,
        raw_worker_input_contract='Explicit immutable package and one renamed Zarr only; no baseline cache, GT, network or historical predictions',finished_utc=now())
    write(folder/'receipt.json',result);return result
