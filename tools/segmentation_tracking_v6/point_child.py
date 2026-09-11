"""Image-derived P0 continuation; launched with the early inherited offline guard."""
import argparse
import os
from pathlib import Path

def main():
    import sitecustomize
    assert sitecustomize.INSTALLED_BEFORE_NUMERICAL
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',choices=['44b6','6bba'],required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    args=p.parse_args()
    import time
    import numpy as np
    from strong_tracker_v3.common import read_json,load_graph,save_arrays,write_json,sha,graph_hash
    from strong_tracker_v3.context import RunContext
    from strong_tracker_v3.inference import rebuild_teachers
    from strong_tracker_v3.association_fresh import build_from_inputs
    from strong_tracker_v3.association import decode
    start=time.monotonic()
    sample=read_json(args.output/'inputs.json')[0];name=sample['dataset']
    ctx=RunContext.default(repo=args.package,v1=os.environ['V5_V1'],v2=os.environ['V5_V2'],
                           out=args.output,work=args.output/'point_scratch')
    baseline=load_graph(args.output/'fresh/incumbent'/f'{name}.npz')
    c0=load_graph(args.output/'predictions'/f'{name}.npz')
    raw=load_graph(args.output/'fresh/raw'/f'{name}.npz')
    pre=load_graph(args.output/'fresh/inputs'/f'pre_ilp_{name}.npz')
    old,en,eh,provenance=rebuild_teachers(ctx,sample,raw,pre,source_model=args.source,
        image_dir=Path(sample['image_path']).parent,current_graph=(baseline['nodes'],baseline['edges']))
    f,_=build_from_inputs(ctx,sample,baseline['nodes'],baseline['edges'],raw,pre,old,en,eh,
                          node_features=provenance.pop('_joint_current_features'))
    assert np.array_equal(baseline['nodes'],c0['nodes'])
    spec=read_json(args.model);m=spec['model'];x=f['edge_features']
    scores=x[:,23]+m['beta'][0]+((x[:,spec['columns']]-m['mean'])/m['scale'])@np.asarray(m['beta'][1:])
    edges,ledger=decode(c0['nodes'],c0['edges'],f,scores,margin=3.,max_fraction=.02)
    save_arrays(args.output/'predictions/P0.npz',nodes=c0['nodes'],edges=edges)
    save_arrays(args.output/'point_evidence.npz',pairs=f['pairs'],edge_features=x)
    import csv
    with (args.output/'P0.csv').open('w',newline='') as handle:
        columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();index=0
        for n in c0['nodes']:
            writer.writerow(dict(id=index,dataset=name,row_type='node',**dict(zip(columns[3:8],map(int,n))),source_id=-1,target_id=-1));index+=1
        for a,b in edges:
            writer.writerow(dict(id=index,dataset=name,row_type='edge',node_id=-1,t=-1,z=-1,y=-1,x=-1,source_id=int(a),target_id=int(b)));index+=1
    write_json(args.output/'point_receipt.json',dict(source=args.source,model_sha256=sha(args.model),
        seconds=time.monotonic()-start,graph_hash=graph_hash(c0['nodes'],edges),
        complete_native_recomputed=True,prior_graph_cache_reads=0,ledger=ledger))

if __name__=='__main__': main()
