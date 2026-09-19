"""Separate source-only official graph evaluation and legal action support audit."""
from pathlib import Path
import sys
import tempfile
import time
from .common import DATA, WORK, RESULTS, REPO, Blocked, read, write, now, sha


def run(source,seed,clip,overfit=False,data=DATA):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:
        raise Blocked('Source engineering evaluation is fit-only')
    prediction=WORK/('source_inference-overfit' if overfit else 'source_inference')/source/str(seed)/clip
    folder=WORK/'source_evaluation'/source/str(seed)/clip
    folder.mkdir(parents=True,exist_ok=True);(folder/'tmp').mkdir(exist_ok=True)
    tempfile.tempdir=str((folder/'tmp').resolve())
    official=REPO/'work/annotation-selection-v1/official'
    source_receipt=read(prediction/'receipt.json')
    if source_receipt['status']!='complete_source_pilot':raise Blocked('Complete source prediction required')
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,data/'train'/f'{clip}.geff'],outputs=[folder],
                  code_roots=[REPO/'tools',REPO/'handover',official,Path(sys.prefix),Path(sys.base_prefix),
                              *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from .data import labels
    from .graphs import read_csv,graph_hash
    from annotation_selection.metric_adapter import evaluate_graph
    from .actions import Bank
    from pipeline_error_training.labels import SourceLabels
    started=time.monotonic()
    nodes,edges=read_csv(prediction/'submission.csv',clip)
    gt_nodes,gt_edges=labels(data/'train'/f'{clip}.geff')
    estimate=read(data/'train'/f'{clip}.geff/zarr.json')['attributes']['geff']['extra']['estimated_number_of_nodes']
    score,matches,_=evaluate_graph(clip,nodes,edges,gt_nodes,gt_edges,[1.625,.40625,.40625],estimate)
    record=dict(status='source_pilot_evaluated',source=source,seed=seed,clip=clip,guard=guard,
                score=score,prediction_graph_hash=graph_hash(nodes,edges),csv_sha256=sha(prediction/'submission.csv'),
                source_only=True,retained_model=False,census=dict(positive=0,supported_negative=0,unknown=0,incomplete=0),
                total_forks=0,positive_witness=None,negative_witness=None)
    with np.load(prediction/'graph.npz') as arrays:
        edge_scores=arrays['edge_scores'] if 'edge_scores' in arrays else None
    if edge_scores is not None and len(nodes):
        bank=Bank(nodes,edges,edge_scores,100)
        evaluator=SourceLabels(nodes,edges,gt_nodes,gt_edges,[1.625,.40625,.40625])
        for i,parent in enumerate(sorted(bank.expanded)):
            group=bank.parent(parent)
            if not group['complete']:
                record['census']['incomplete']+=1;continue
            risks=[]
            for action in group['forks']:
                label=evaluator.decision(action);risk=label['metric_fork_target'];risks.append(risk)
                if risk in (0,1):
                    key='positive_witness' if risk else 'negative_witness'
                    if record[key] is None:
                        changed=(set(map(tuple,edges))-set(action.remove))|set(action.add)
                        changed=np.asarray(sorted(changed),np.int64).reshape(-1,2)
                        # IDs in these generated graphs are consecutive native IDs.
                        if not np.array_equal(nodes[:,0],np.arange(len(nodes))):raise ValueError('Explicit index-to-ID conversion required')
                        witness,_,_=evaluate_graph(clip,nodes,changed,gt_nodes,gt_edges,[1.625,.40625,.40625],estimate)
                        record[key]=dict(action_key=str((action.event[0],sorted(action.add),sorted(action.remove))),
                                         risk=risk,official_score=witness,label_guided_not_model=True,
                                         local_legal_oracle_only=True,solver_and_global_edit_cap_applied=False,
                                         changed_edges=len(action.add|action.remove),deployment_edit_cap=int(.02*len(edges)))
            record['total_forks']+=len(risks)
            target='positive' if 1 in risks else 'supported_negative' if risks and all(y==0 for y in risks) else 'unknown'
            record['census'][target]+=1
            if i%1000==0:
                write(folder/'progress.json',dict(parents=i,total_parents=len(bank.expanded),census=record['census']))
        record['parent_census']=len(bank.expanded)
    else:record['bank_blocker']='Empty pilot graph or unavailable complete clean edge evidence'
    record.update(wall_seconds=time.monotonic()-started,finished_utc=now())
    write(folder/'receipt.json',record)
    return record
