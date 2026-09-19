"""Complete source-only pilot image-to-CSV profiling, never target evaluation."""
import fcntl
from pathlib import Path
import sys
import tempfile
import time
from .common import ARCH, DATA, WORK, RESULTS, Blocked, read, write, now, sha


def run(source,seed,clip,data=DATA,overfit=False):
    split=read(WORK/'source_partitions.json')[source]
    if clip not in split['fit']:raise Blocked('Source pilot inference is fit-only')
    folder=WORK/('source_inference-overfit' if overfit else 'source_inference')/source/str(seed)/clip
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    model_path=WORK/('overfit' if overfit else 'pilot')/source/str(seed)/'resume.pt'
    if read(model_path.with_name('receipt.json'))['status'] not in ('real_optimizer_resume_passed','overfit_complete'):
        raise Blocked('Pilot model did not pass optimizer/resume gate')
    # Open the shared lock before source read guarding; hold only during work units.
    lock=Path('/kaggle/working/cell-tracking/detector-screen-20260914.gpu.lock').open('a+')
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[data/'train'/f'{clip}.zarr',model_path],outputs=[folder],
                  code_roots=[Path(__file__).parent,ARCH/'src',Path(sys.prefix),Path(sys.base_prefix),
                              *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    import zarr
    from .data import normalize
    from .upstream import Upstream
    from .graphs import peaks,candidate_union,continuation,export_csv,read_csv,graph_hash
    torch.set_num_threads(4)
    model=Upstream(ARCH)
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)['model']);model.eval()
    image=zarr.open_array(str(data/'train'/f'{clip}.zarr/0'),mode='r')
    started=time.monotonic();charged=0.;windows=[];transitions=[];features={};nodes=[];edges=[];evidence=[]
    # Keep at most three frame feature maps resident on the host. All aligned
    # logits and features from the two temporal windows use arithmetic averaging.
    previous=None;last_points=None;last_feature=None;last_ids=None;confidence=[]
    record=dict(status='running',source=source,seed=seed,clip=clip,model_sha256=sha(model_path),
                pilot_not_retained=True,frames=0,target_scores_opened=False,guard=guard,
                implementation_sha256=sha(Path(__file__)))
    try:
        for t in range(100):
            current=None
            if t<99:
                x=np.stack([normalize(np.asarray(image[j]))[0][None] for j in (t,t+1)])[None]
                fcntl.flock(lock,fcntl.LOCK_EX)
                lease=time.monotonic();free,total=torch.cuda.mem_get_info()
                allowance=min(20*2**30-(total-free),free-2*2**30)
                if allowance<2*2**30:raise Blocked('Insufficient GPU room for inference')
                torch.cuda.set_per_process_memory_fraction(allowance/total)
                try:
                    model.cuda()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,feature=model(torch.from_numpy(x).cuda())
                    current=(logits[0].float().cpu(),feature[0].float().cpu())
                    del logits,feature
                finally:
                    model.cpu();torch.cuda.empty_cache();charged+=time.monotonic()-lease
                    fcntl.flock(lock,fcntl.LOCK_UN)
                windows.append(time.monotonic()-lease)
            if previous is None:logits,feature=current[0][0],current[1][0]
            elif current is None:logits,feature=previous[0][1],previous[1][1]
            else:logits,feature=(previous[0][1]+current[0][0])/2,(previous[1][1]+current[1][0])/2
            points,scores=peaks(logits.numpy());ids=np.arange(len(nodes),len(nodes)+len(points),dtype=np.int64)
            nodes.extend([[int(i),t,*map(int,p)] for i,p in zip(ids,points)])
            confidence.extend((1/(1+np.exp(-scores))).tolist())
            if last_points is not None and len(last_points) and len(points):
                pairs=candidate_union(last_points,points)
                fcntl.flock(lock,fcntl.LOCK_EX);lease=time.monotonic()
                try:
                    model.cuda()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        values=model.association(last_feature.cuda(),feature.cuda(),torch.tensor(last_points,dtype=torch.float32,device='cuda'),
                                                 torch.tensor(points,dtype=torch.float32,device='cuda'),t-1).float().cpu().numpy()
                    chosen=continuation(last_points,points,pairs,values[pairs[:,0],pairs[:,1]])
                    edges.extend([[int(last_ids[a]),int(ids[b])] for a,b in chosen])
                    event_pairs=candidate_union(last_points,points,neighbors=6,reverse_neighbors=2,gate=16.)
                    all_pairs=np.asarray(sorted(set(map(tuple,pairs))|set(map(tuple,event_pairs))),np.int64).reshape(-1,2)
                    evidence.extend([[int(last_ids[a]),int(ids[b]),float(values[a,b])] for a,b in all_pairs])
                finally:
                    model.cpu();torch.cuda.empty_cache();charged+=time.monotonic()-lease;fcntl.flock(lock,fcntl.LOCK_UN)
                transitions.append(time.monotonic()-lease)
                if transitions[-1]>60:raise Blocked('Association work unit exceeded registered 60-second lease; no production run allowed')
            last_points,last_feature,last_ids=points,feature,ids
            previous=current
            record.update(frames=t+1,nodes=len(nodes),edges=len(edges),gpu_lease_seconds=charged,wall_seconds=time.monotonic()-started)
            write(folder/'progress.json',record)
            if t%10==0:print(f'{source} source pilot inference {t+1}/100 frames, {len(nodes)} nodes',flush=True)
        nodes=np.array(nodes,np.int64).reshape(-1,5);edges=np.array(edges,np.int64).reshape(-1,2)
        actual=export_csv(folder/'submission.csv',clip,nodes,edges)
        restored=read_csv(folder/'submission.csv',clip)
        equal=all(np.array_equal(a,b) for a,b in zip(actual,restored))
        if not equal:raise Blocked('Source integer CSV roundtrip mismatch')
        np.savez_compressed(folder/'graph.npz',nodes=actual[0],edges=actual[1],
                            confidence=np.asarray(confidence,np.float32),edge_scores=np.asarray(evidence,np.float64).reshape(-1,3))
        record.update(status='complete_source_pilot',graph_hash=graph_hash(*actual),csv_sha256=sha(folder/'submission.csv'),
                      csv_roundtrip_exact=True,window_seconds=windows,transition_seconds=transitions)
    except Exception as e:
        record.update(status='blocked',failure_type=type(e).__name__,failure=str(e))
        raise
    finally:
        record.update(gpu_lease_seconds=charged,wall_seconds=time.monotonic()-started,finished_utc=now())
        write(folder/'receipt.json',record);lock.close()
    return record
