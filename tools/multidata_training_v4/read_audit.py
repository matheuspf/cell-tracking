"""Actual one-update source-only read probes, separate from production fits."""
import os
from pathlib import Path
import subprocess
import sys

def probe(source):
    opposite='6bba' if source=='44b6' else '44b6';opened=set();denied=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute().resolve();s=str(p)
        if any(x in s for x in ['/cache/real/','/cache/dreal/']):
            if p.name.startswith(opposite):denied.append(s);raise PermissionError('Opposite-embryo training cache denied')
            if p.suffix=='.npz':opened.add(s)
        if ('/evaluation/gt/' in s or '.geff' in s) and opposite in s:raise PermissionError('Opposite-embryo labels denied')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    sys.addaudithook(audit)
    import torch
    from .common import OUT,write,now,SEED
    from .sampling import Bank,paths
    from .train import MODELS,loss
    torch.set_num_threads(2);torch.manual_seed(SEED+1);results=[]
    for component in ['D','G','I']:
        ps=paths(source,component);assert all(p.stem.startswith(source+'_') for p in ps)
        bank=Bank(ps,component);b=bank.batch(0,8);m=MODELS[component]().cuda();optim=torch.optim.AdamW(m.parameters(),lr=.0001)
        optim.zero_grad();value=loss(m,b,component);value.backward();optim.step()
        assert torch.isfinite(value)
        results.append(dict(component=component,actual_updates=1,source_cache_files=len(ps),finite_loss=float(value.detach())))
        del m,optim,b,bank
    try:open(paths(opposite,'G')[0],'rb')
    except PermissionError:pass
    else:raise AssertionError('Opposite source unexpectedly readable')
    write(OUT/f'source_read_audit_{source}.json',dict(created=now(),passed=True,source=source,opposite=opposite,
        source_cache_files_read=len(opened),opposite_cache_reads=0,intentional_denial_probes=len(denied),
        gradient_probes=results,probe_weights_not_retained=True,not_a_full_training_arm=True))

def run():
    from .common import OUT,read,write,now
    for source in ['44b6','6bba']:
        subprocess.run([sys.executable,'-m','multidata_training_v4.read_audit',source],check=True)
    write(OUT/'source_read_audit.json',dict(created=now(),passed=True,actual_extra_sanity_updates=6,
        directions=[read(OUT/f'source_read_audit_{s}.json') for s in ['44b6','6bba']]))

if __name__=='__main__':probe(sys.argv[1])
