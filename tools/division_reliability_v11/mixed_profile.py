"""Real 32-group prefix/joint C11 optimizer and exact checkpoint pilot."""
from pathlib import Path
import sys
import tempfile
import time
from .common import REPO,WORK,Blocked,read,write,sha,now


def run(source,seed,clip,*,replay=False):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Source-fit fixture required')
    bank=WORK/'pilot_banks'/source/str(seed)/'fit'
    folder=WORK/'mixed_profile'/source/str(seed);folder.mkdir(parents=True,exist_ok=True)
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .resources import Resources,ROOT
    resources=Resources('pilot',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[bank],outputs=[folder,ROOT],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
        *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    from .event_training import Groups,fit_normalizer,mixed_update
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    from .upstream import learning_rate
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.init()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    normalization=fit_normalizer(bank,[clip]);groups=Groups(bank,[clip],normalization['values'])
    write(folder/'normalizer.json',normalization)
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rows=[];started=time.monotonic()
    if replay:
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['sampler'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    for step in ((402,) if replay else (1,401,402)):
        samples,selection=groups.batch(rng,step,4000)
        with resources.lease(2*2**30) as lease:
            model.cuda()
            for state in opt.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor) and k!='step':state[k]=v.cuda()
            tick=time.monotonic();row=mixed_update(model,opt,samples,learning_rate(step,4000,3e-4,200));torch.cuda.synchronize()
            row['optimizer_seconds']=time.monotonic()-tick
            model.cpu()
            for state in opt.state.values():
                for k,v in state.items():
                    if isinstance(v,torch.Tensor):state[k]=v.cpu()
            torch.cuda.empty_cache()
        rows.append(dict(step=step,**row,selection=selection,gpu_lease_seconds=lease['seconds']))
        if step==401:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=rng.bit_generator.state,
                            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder/'resume.pt')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=rng.bit_generator.state,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder/('replay.pt' if replay else 'final.pt'))
    result=dict(status='passed',source=source,seed=seed,clip=clip,guard=guard,production_fit=False,
                rows=rows,unique_visited_groups=len(groups.visits),complete_source_bank=groups.receipts[clip]['census'],
                real_pixels=True,identity_gradient_tested=True,encoder_recomputed_with_gradients=True,
                wall_seconds=time.monotonic()-started,finished_utc=now(),code_sha256=sha(Path(__file__)))
    write(folder/('replay.json' if replay else 'receipt.json'),result);resources.close();return result
