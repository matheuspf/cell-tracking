"""Selective cached-array I/O for the unchanged frozen event decoder.

The algorithm reads event rows only. Classifier features and image crops stay in
the fingerprinted proposal archive and are not materialized again at decode.
"""
from pathlib import Path
import argparse
import json
import resource
import time

import numpy as np

from .common import graph_hash, load_graph, now, read_json, run_pool, save_arrays, sha, write_json
from .context import RunContext
from .event_pipeline import stable_lock, variants


def one(task):
    from .decode import divisions
    ctx,sample,configs,output_root=task
    name=sample['dataset'];start=time.perf_counter();output_root=Path(output_root)
    prediction=read_json(ctx.out/'event_probabilities'/f'{name}.json');pool=prediction['inputs']['pool']
    graph=load_graph(ctx.incumbent(name));proposal=ctx.out/'events'/pool/f'{name}.npz'
    # No feature/crop array is read. The frozen algorithm consumes only events.
    with np.load(proposal,allow_pickle=False) as archive:h={'events':archive['events']}
    native=load_graph(ctx.out/'features'/f'{name}.npz')
    probabilities=load_graph(ctx.out/'event_probabilities'/f'{name}.npz')
    loader_sha=sha(__file__)
    for config in configs:
        variant=config['name'];path=output_root/'candidate_graphs'/variant/f'{name}.npz'
        stamp=dict(incumbent_sha256=sha(ctx.incumbent(name)),event_sha256=prediction['inputs']['event_sha256'],
            probabilities_sha256=prediction['sha256'],config=config,
            decode_code_sha256=sha(Path(__file__).with_name('decode.py')))
        if path.with_suffix('.json').exists():
            receipt=read_json(path.with_suffix('.json'))
            if receipt['inputs']!=stamp or receipt['sha256']!=sha(path):raise ValueError('Selective loader refuses stale graph')
            continue
        edges,report,ledger=divisions(graph['nodes'],graph['edges'],h,probabilities[config['model']],native,
            sample['physical_scale'],threshold=config['threshold'],existing_only=config['existing_only'],replace=config['replace'])
        save_arrays(path,nodes=graph['nodes'],edges=edges)
        receipt=dict(dataset=name,variant=variant,inputs=stamp,sha256=sha(path),graph_hash=graph_hash(graph['nodes'],edges),
            decode=report,seconds=time.perf_counter()-start,
            io_execution=dict(loader_sha256=loader_sha,proposal_arrays_read=['events'],algorithm_unchanged=True))
        write_json(path.with_suffix('.json'),receipt)
        write_json(output_root/'edit_ledgers'/variant/f'{name}.json',dict(dataset=name,actions=ledger))
    return dict(dataset=name,seconds=time.perf_counter()-start,variants=len(configs),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def verify(ctx):
    profile=read_json(ctx.out/'event_dense_decode_profile.json')
    sample=next(s for s in ctx.samples() if s['dataset']==profile['dataset'])
    root=ctx.out/'evaluation/event_io_loader_parity';result=one((ctx,sample,variants(),root));rows=[]
    for config in variants():
        variant=config['name'];relative=Path('candidate_graphs')/variant/f"{sample['dataset']}.npz"
        before=read_json((ctx.out/relative).with_suffix('.json'));after=read_json((root/relative).with_suffix('.json'))
        assert sha(ctx.out/relative)==sha(root/relative)==before['sha256']==after['sha256']
        assert before['inputs']==after['inputs'] and before['decode']==after['decode']
        ledger=Path('edit_ledgers')/variant/f"{sample['dataset']}.json"
        assert read_json(ctx.out/ledger)==read_json(root/ledger)
        rows.append(dict(variant=variant,graph_sha256=after['sha256'],graph_bytes_equal=True,ledger_equal=True,decode_counts_equal=True))
    receipt=dict(created=now(),loader_sha256=sha(__file__),algorithm_sha256=sha(Path(__file__).with_name('decode.py')),
        model_lock_sha256=sha(ctx.out/'event_model_lock.json'),probability_lock_sha256=sha(ctx.out/'event_probability_lock.json'),
        **result,parity=rows,method='Largest fixed candidate pool, all TEN frozen variants; graph bytes, action ledgers and solver counts exactly equal')
    write_json(ctx.out/'event_io_loader_parity.json',receipt);return receipt


def run(ctx,workers=4):
    proof=read_json(ctx.out/'event_io_loader_parity.json')
    if proof['loader_sha256']!=sha(__file__) or len(proof['parity'])!=10:raise ValueError('Selective loader requires current all-arm parity proof')
    if proof['algorithm_sha256']!=sha(Path(__file__).with_name('decode.py')):raise ValueError('Frozen algorithm drift')
    config=read_json(ctx.out/'event_round_config.json');configs=variants();samples=ctx.samples()
    if config['variants']!=configs or config['model_lock_sha256']!=sha(ctx.out/'event_model_lock.json') or config['probability_lock_sha256']!=sha(ctx.out/'event_probability_lock.json'):
        raise ValueError('Frozen round inputs changed')
    stable_lock(ctx.out/'event_io_loader_lock.json',dict(created=now(),loader_sha256=sha(__file__),
        parity_sha256=sha(ctx.out/'event_io_loader_parity.json'),algorithm_sha256=proof['algorithm_sha256'],
        policy_unchanged=True,reason='Selective immutable NPZ materialization only; old exact graphs/receipts resume unchanged'),
        ['loader_sha256','parity_sha256','algorithm_sha256'])
    list(run_pool(one,[(ctx,s,configs,ctx.out) for s in samples],workers))
    predictions={v['name']:{s['dataset']:sha(ctx.out/'candidate_graphs'/v['name']/f"{s['dataset']}.npz") for s in samples} for v in configs}
    stable_lock(ctx.out/'event_round_lock.json',dict(created=now(),variants=configs,predictions=predictions,
        samples=len(samples),both_source_directions_frozen=True,outer_scores_exposed=False),['variants','predictions','samples'])
    import pandas as pd
    records=[]
    for v in configs:
        for sample in samples:
            ledger=read_json(ctx.out/'edit_ledgers'/v['name']/f"{sample['dataset']}.json")
            for action in ledger['actions']:
                records.append(dict(dataset=sample['dataset'],variant=v['name'],candidate=action['candidate'],kind=action['kind'],
                    value=action['value'],event_probability=action['event_probability'],
                    **{k:json.dumps(action[k]) for k in ['removed_edges','added_edges','canonical_nodes','native_support','owner_alternatives']},
                    solver_status=action['solver_status']))
    pd.DataFrame(records,columns=['dataset','variant','candidate','kind','value','event_probability','removed_edges','added_edges',
        'canonical_nodes','native_support','owner_alternatives','solver_status']).to_parquet(ctx.out/'event_edit_ledger.parquet',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['verify','run']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();ctx=RunContext.default().check_outputs()
    if args.stage=='verify':print(verify(ctx),flush=True)
    else:run(ctx,args.workers)


if __name__=='__main__':main()
