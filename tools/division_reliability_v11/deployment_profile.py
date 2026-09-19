"""All-frame real crop/embedding cost and fixed-stride complete-group timing."""
from pathlib import Path
import sys,tempfile,time
from .common import REPO,DATA,WORK,Blocked,read,write,sha,now


def run(source,seed,clip):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Source-fit profiling only')
    prediction=WORK/'source_inference-overfit'/source/str(seed)/clip
    fitted=WORK/'mixed_profile'/source/str(seed);folder=WORK/'deployment_profile'/source/str(seed)
    folder.mkdir(parents=True,exist_ok=True);(folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .resources import Resources,ROOT
    resources=Resources('pilot',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,fitted,DATA/'train'/f'{clip}.zarr'],outputs=[folder,ROOT],
                  code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    from .actions import Bank
    from .policy import score_bank
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    model.load_state_dict(torch.load(fitted/'final.pt',map_location='cpu',weights_only=True)['model']);model.eval()
    with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
    bank=Bank(nodes,edges,scores,100);pp=sorted(bank.expanded)
    selected={pp[int(i)] for i in np.linspace(0,len(pp)-1,min(4096,len(pp)))}
    before=(ROOT/'gpu-leases.jsonl').stat().st_size if (ROOT/'gpu-leases.jsonl').exists() else 0
    tic=time.monotonic();count=actions=0
    for group,a,b in score_bank(bank,confidence,DATA/'train'/f'{clip}.zarr',read(fitted/'normalizer.json')['values'],resources,
            model=model,progress=folder/'progress.json',_profile_parents=selected):
        count+=1;actions+=len(b)
    import json,os
    with (ROOT/'gpu-leases.jsonl').open() as f:
        f.seek(before);rows=[json.loads(x) for x in f if x.strip()]
    rows=[r for r in rows if r['pid']==os.getpid()]
    result=dict(status='passed',source=source,seed=seed,clip=clip,guard=guard,all_frames=100,all_nodes=len(nodes),
                profiled_complete_parent_groups=count,profiled_legal_actions=actions,total_deployed_parents=len(pp),
                selection='4096 equally spaced prediction-only parent indices; complete within-parent denominator',
                all_node_embedding_gpu_seconds=sum(r['seconds'] for r in rows if r.get('operation')=='compact_embeddings'),
                sampled_head_gpu_seconds=sum(r['seconds'] for r in rows if r.get('operation')=='compact_heads'),
                total_measured_gpu_seconds=sum(r['seconds'] for r in rows),gpu_leases=rows,
                full_graph_policy_result=False,wall_seconds=time.monotonic()-tic,finished_utc=now())
    write(folder/'receipt.json',result);resources.close();return result
