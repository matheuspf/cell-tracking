"""Audit a retained compact final against its complete history and resume state."""
import argparse
from collections import Counter
import json,math,sys
from pathlib import Path
from .common import WORK,RESULTS,REPO,Blocked,read,write,sha,now


def run(source,seed):
    lock=read(RESULTS/'execution_lock.json');horizon=lock['event_updates']
    if (source,seed) not in [(c['source'],c['seed']) for c in lock['schedule']]:
        raise Blocked('Unregistered source/seed cell')
    folder=WORK/'fits'/source/str(seed)/'compact'
    output=WORK/'checks/compact_final'/source/str(seed)
    output.mkdir(parents=True,exist_ok=True);sys.dont_write_bytecode=True
    from .guard import install
    guard=install(inputs=[folder],outputs=[output],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
        *[Path(p) for p in sys.path if 'site-packages' in p]])
    final=read(folder/'final.json')
    if final['status']!='trained' or final['updates']!=horizon or final['lock_identity']!=lock['identity']:
        raise Blocked('Compact final does not match the locked horizon')
    parents={n:sha(folder/n) for n in ('final.json','final.pt','resume.pt','history.jsonl','mining/receipt.json')}
    if parents['final.pt']!=final['weights_sha256']:raise Blocked('Retained compact weights changed')
    for name,value in final['implementation_sha256'].items():
        if sha(Path(__file__).with_name(name))!=value:raise Blocked('Compact fitting implementation changed: '+name)
    rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,horizon+1)):raise Blocked('Compact history has missing/repeated updates')
    prefix=horizon//10;denominators=Counter();slots=Counter();fallback=Counter()
    for r in rows:
        early=r['step']<=prefix
        expected=dict(identity=32 if early else 16,occurrence=0 if early else 16,ranking=0 if early else 8,consistency=32)
        expected_slots=dict(identity=32) if early else dict(identity=16,positive=8,negative=4,hard=4)
        if r['denominators']!=expected or dict(Counter(s['slot'] for s in r['selection']))!=expected_slots:
            raise Blocked('Recorded compact groups differ from the registered phase')
        if not all(math.isfinite(v) for v in [*r['losses'].values(),r['gradient_norm'],r['encoder_gradient_norm'],r['lr']]):
            raise Blocked('Nonfinite compact training measurement')
        if early and (r['losses']['occurrence']!=0 or r['losses']['ranking']!=0):
            raise Blocked('Event supervision entered the identity prefix')
        denominators.update(r['denominators']);slots.update(s['slot'] for s in r['selection'])
        phase='before_mining' if r['step']<=horizon//2 else 'after_mining'
        fallback[phase]+=sum(s['hard_fallback'] for s in r['selection'])
    if final['event_supervised_updates']!=horizon-prefix or final['mining_passes']!=1 or rows[-1]['lr']!=1e-6:
        raise Blocked('Final compact schedule metadata differs')
    if fallback['before_mining']!=4*(horizon//2-prefix):raise Blocked('Pre-mining hard slots did not use uniform negatives')
    if not final['guard']['installed_before_numerical'] or any(final['guard'][k] for k in ('denied','network_denied','subprocess_denied')):
        raise Blocked('Compact fitting guard did not pass')
    import torch
    torch.set_num_threads(1)
    weights=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    if resume['step']!=horizon or resume['source']!=source or resume['seed']!=seed or resume['lock_identity']!=lock['identity']:
        raise Blocked('Final resume identity differs')
    if not all(k in resume for k in ('optimizer','scheduler','sampler','cpu_rng','cuda_rng')):
        raise Blocked('Final optimizer/scheduler/sampler/RNG state is incomplete')
    a,b=weights['model'],resume['model']
    if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) and bool(torch.isfinite(a[k]).all()) for k in a):
        raise Blocked('Retained weights are nonfinite or differ from the final resume state')
    result=dict(status='passed',source=source,seed=seed,updates=horizon,identity_supervised_prefix_updates=prefix,
        event_supervised_updates=horizon-prefix,mining_passes=1,denominator_totals=dict(denominators),
        sampled_slot_totals=dict(slots),hard_slot_fallback_totals=dict(fallback),final_learning_rate=rows[-1]['lr'],
        all_losses_gradients_and_weights_finite=True,retained_model_equals_final_resume=True,
        optimizer_scheduler_sampler_rng_present=True,parents=parents,guard=guard,
        audit_implementation_sha256=sha(Path(__file__)),scope='Complete actual source training and retained artifacts; no accuracy claim',finished_utc=now())
    receipt=output/'receipt.json'
    if receipt.exists():
        prior=read(receipt)
        if any(prior[k]!=result[k] for k in result if k not in ('finished_utc','guard')):
            raise Blocked('Completed compact audit inputs or result changed')
        return prior
    write(receipt,result,immutable=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,choices=['44b6','6bba'])
    p.add_argument('--seed',required=True,type=int,choices=[20260918,314159]);a=p.parse_args()
    print(json.dumps(run(a.source,a.seed),indent=2))
