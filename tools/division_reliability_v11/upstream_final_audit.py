"""Audit complete C00 trajectories, final checkpoints and all optimizer leases."""
import argparse
from collections import Counter
import hashlib,json,math,sys
from pathlib import Path
from .common import WORK,RESULTS,REPO,Blocked,read,write,sha,now


def run(source,seed):
    lock=read(RESULTS/'execution_lock.json');horizon=lock['upstream_updates']
    if (source,seed) not in [(c['source'],c['seed']) for c in lock['schedule']]:
        raise Blocked('Unregistered source/seed cell')
    fit_names=set(read(WORK/'source_partitions.json')[source]['fit'])
    folder=WORK/'fits'/source/str(seed)/'upstream';resources=WORK/'resources'
    output=WORK/'checks/upstream_final'/source/str(seed)
    output.mkdir(parents=True,exist_ok=True);sys.dont_write_bytecode=True
    from .guard import install
    guard=install(inputs=[folder,resources/'totals.json',resources/'gpu-leases.jsonl'],outputs=[output],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
                    *[Path(p) for p in sys.path if 'site-packages' in p]])
    final=read(folder/'final.json')
    if final['status']!='trained' or final['updates']!=horizon or final['lock_identity']!=lock['identity']:
        raise Blocked('Upstream final does not match the locked horizon')
    for name,value in lock['implementation_sha256'].items():
        if sha(Path(__file__).with_name(name))!=value:raise Blocked('Locked upstream implementation changed: '+name)
    parents={n:sha(folder/n) for n in ('final.json','final.pt','resume.pt','history.jsonl')}
    if parents['final.pt']!=final['weights_sha256']:raise Blocked('Retained upstream weights changed')
    rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,horizon+1)):raise Blocked('Upstream history has missing/repeated updates')
    sampled=Counter()
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('loss','detection','association','gradient_norm','lr')):
            raise Blocked('Nonfinite upstream training measurement')
        if len(r['samples'])!=8 or len(r['groups'])!=8:raise Blocked('Upstream batch differs from the registered size')
        for name,t in r['samples']:
            if name not in fit_names or not 0<=t<99:raise Blocked('Optimizer sample is outside source-fit clip/time support')
            sampled[name]+=1
        for g in r['groups']:
            if r['step']<=horizon//10 and g['proposal']:
                raise Blocked('Proposal supervision entered the GT-jitter prefix')
            if not g['fallback'] and g['supervised']!=g['proposal']:
                raise Blocked('Supported GT/proposal query quotas differ')
    if rows[-1]['lr']!=1e-6:raise Blocked('Final upstream learning rate differs')
    if not final['guard']['installed_before_numerical'] or any(final['guard'][k] for k in ('denied','network_denied','subprocess_denied')):
        raise Blocked('Upstream fitting guard did not pass')
    interrupted=[]
    for p in sorted((folder/'interrupted_updates').glob('*.json')):
        parents[str(p.relative_to(folder))]=sha(p);interrupted.extend(read(p))
    # Read only the journal prefix already closed and included in its atomic cursor.
    size=read(resources/'totals.json')['journal_bytes']
    with (resources/'gpu-leases.jsonl').open('rb') as stream:
        leases=[r for line in stream.read(size).splitlines() if
                (r:=json.loads(line))['stage']=='upstream' and r['source']==source and r['seed']==seed]
    completed=[r for r in leases if r['status']=='complete']
    if len(completed)!=horizon+len(interrupted):raise Blocked('Closed optimizer leases differ from retained plus replayed update attempts')
    discarded_seconds=sum(r['gpu_lease_seconds'] for r in interrupted)
    if not math.isclose(sum(r['seconds'] for r in completed),final['charged_gpu_lease_seconds']+discarded_seconds,rel_tol=0,abs_tol=1e-6):
        raise Blocked('Retained and interrupted optimizer charges do not reconcile with the journal')
    import torch
    torch.set_num_threads(1)
    weights=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    if resume['step']!=horizon or resume['source']!=source or resume['seed']!=seed or resume['lock_identity']!=lock['identity']:
        raise Blocked('Final upstream resume identity differs')
    if not all(k in resume for k in ('optimizer','scheduler','sampler','cpu_rng','cuda_rng','python_rng')):
        raise Blocked('Final optimizer/scheduler/sampler/RNG state is incomplete')
    a,b=weights['model'],resume['model']
    if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) and bool(torch.isfinite(a[k]).all()) for k in a):
        raise Blocked('Retained weights are nonfinite or differ from the final resume state')
    result=dict(status='passed',source=source,seed=seed,updates=horizon,batch_size=8,
        source_fit_clips=len(fit_names),distinct_sampled_fit_clips=len(sampled),
        samples_per_clip_range=[min(sampled.values()),max(sampled.values())],
        all_optimizer_samples_source_fit=True,all_losses_gradients_and_weights_finite=True,
        retained_model_equals_final_resume=True,optimizer_scheduler_sampler_rng_present=True,
        final_learning_rate=rows[-1]['lr'],discarded_update_attempts=len(interrupted),
        discarded_update_gpu_lease_seconds=discarded_seconds,retained_trajectory_gpu_lease_seconds=final['charged_gpu_lease_seconds'],
        all_optimizer_attempt_gpu_lease_seconds=sum(r['seconds'] for r in leases),
        optimizer_lease_status_counts=dict(Counter(r['status'] for r in leases)),
        optimizer_lease_records_sha256=hashlib.sha256(json.dumps(leases,sort_keys=True).encode()).hexdigest(),
        parents=parents,guard=guard,audit_implementation_sha256=sha(Path(__file__)),
        scope='Complete actual C00 training, final artifacts and optimizer charge reconciliation; no accuracy claim',finished_utc=now())
    receipt=output/'receipt.json'
    if receipt.exists():
        prior=read(receipt)
        if any(prior[k]!=result[k] for k in result if k not in ('finished_utc','guard')):
            raise Blocked('Completed upstream audit inputs or result changed')
        return prior
    write(receipt,result,immutable=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,choices=['44b6','6bba'])
    p.add_argument('--seed',required=True,type=int,choices=[20260918,314159]);a=p.parse_args()
    print(json.dumps(run(a.source,a.seed),indent=2))
