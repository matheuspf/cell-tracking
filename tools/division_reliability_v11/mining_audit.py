"""Check a completed real mining pool and observed post-mining sampler draws."""
import argparse
import json
import math
import sys
from pathlib import Path
from .common import REPO,WORK,RESULTS,read,write,sha,now


def run(source,seed):
    from .readiness import require_production
    require_production('mining-audit')
    lock=read(RESULTS/'execution_lock.json');clips=read(WORK/'source_partitions.json')[source]['fit']
    fit=WORK/'fits'/source/str(seed);mining=fit/'compact/mining';bank=WORK/'banks'/source/str(seed)/'fit'
    output=WORK/'checks/mining_complete'/source/str(seed);output.mkdir(parents=True,exist_ok=True)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[mining,bank,fit/'compact/history.jsonl'],outputs=[output],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    receipt=read(mining/'receipt.json');signature=read(mining/'signature.json')
    assert receipt['passes']==1 and receipt['source_fit_only'] and receipt['signature']==signature
    assert signature['source']==source and signature['seed']==seed and signature['step']==lock['event_updates']//2
    assert sha(mining/'frozen_checkpoint.pt')==signature['checkpoint_sha256']
    assert set(receipt['summary'])==set(signature['source_bank_receipts'])==set(clips)
    selected=0;supported=0;parity=[];parents=0;forks=0
    for clip,details in receipt['summary'].items():
        assert all(details['guard'][k]==0 for k in ('denied','network_denied','subprocess_denied'))
        assert details['unknown_used_as_negative'] is False and len(details['selected_groups'])<=256
        assert sha(bank/clip/'receipt.json')==signature['source_bank_receipts'][clip]
        b=read(bank/clip/'receipt.json');assert sha(bank/clip/'training.npz')==b['training_sha256']
        with np.load(bank/clip/'training.npz') as arrays:
            risks=arrays['risks'];offset=arrays['offset']
            actual=sum(z>a and bool((risks[a:z]==0).all()) for a,z in zip(offset[:-1],offset[1:]))
            assert actual==details['supported_negative_pool']
            for i in details['selected_groups']:
                assert 0<=i<len(offset)-1
                a,z=map(int,offset[i:i+2]);assert z>a and (risks[a:z]==0).all()
        selected+=len(details['selected_groups']);supported+=actual
        p=read(mining/clip/'progress.json');assert p['status']=='complete' and p['embedding_parity']['passed']
        parity.append(p['embedding_parity']['maximum_absolute_difference'])
        parents+=p['total_parents'];forks+=p['census']['complete_group_unique_forks']
    resumed=[]
    with (fit/'compact/history.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):break  # Ignore a concurrently appended unfinished record.
            r=json.loads(line)
            if r['step']<=signature['step']:continue
            slots=[v for v in r['selection'] if v['slot']=='hard'];assert len(slots)==4
            for v in slots:
                assert not v['hard_fallback'] and v['key'] in receipt['pool'][v['clip']]
                expected=1/(len(receipt['pool'])*len(receipt['pool'][v['clip']]))
                assert math.isclose(v['selection_probability'],expected,rel_tol=0,abs_tol=1e-15)
            assert r['denominators']==dict(identity=16,occurrence=16,ranking=8,consistency=32)
            assert all(math.isfinite(x) for x in r['losses'].values())
            resumed.append(r['step'])
    if resumed:assert resumed==list(range(signature['step']+1,max(resumed)+1))
    result=dict(status='passed',utc=now(),source=source,seed=seed,mining_passes=1,complete_source_fit_clips=len(clips),
        frozen_checkpoint_sha256=signature['checkpoint_sha256'],merged_mining_receipt_sha256=sha(mining/'receipt.json'),
        selected_supported_negative_groups=int(selected),complete_supported_negative_pool=int(supported),
        all_selected_groups_verified_supported_negative=True,unknown_used_as_negative=False,
        enumerated_parent_groups=int(parents),complete_unique_forks=int(forks),
        sampled_embedding_parity_max_absolute_difference=max(parity),sampled_embedding_parity_tolerance=1e-5,
        post_mining_resume_observed=bool(resumed),post_mining_updates_observed=len(resumed),
        first_post_mining_update=min(resumed) if resumed else None,last_post_mining_update=max(resumed) if resumed else None,
        observed_hard_slots_verified_against_merged_pool=True,registered_mixed_denominators_exact=True,
        guard=guard,final_model_result=False,target_metrics_opened=False)
    if (output/'receipt.json').exists():
        old=read(output/'receipt.json')
        for key in ('source','seed','frozen_checkpoint_sha256','merged_mining_receipt_sha256',
                    'complete_source_fit_clips','selected_supported_negative_groups',
                    'complete_supported_negative_pool','enumerated_parent_groups','complete_unique_forks'):
            assert old[key]==result[key]
        return old  # Keep the original observed resume window and timestamp.
    write(output/'receipt.json',result,immutable=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['44b6','6bba'],required=True)
    p.add_argument('--seed',type=int,choices=[20260918,314159],required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.seed),indent=2))
