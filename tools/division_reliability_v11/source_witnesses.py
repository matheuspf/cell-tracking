"""Label-guided legal-action diagnostics on retained source-fit C00 graphs."""
from pathlib import Path
import sys,tempfile,time
from .common import DATA,REPO,WORK,Blocked,read,write,sha,now


def run(source,seed,*,clips=None,folder=None):
    registered=read(WORK/'source_partitions.json')[source]['fit']
    selected=sorted(registered if clips is None else clips)
    if not selected or not set(selected)<=set(registered):raise Blocked('Witnesses require registered source-fit clips')
    folder=Path(folder or WORK/'source_diagnostics'/source/str(seed)/'witnesses')
    folder.mkdir(parents=True,exist_ok=True);(folder/'tmp').mkdir(exist_ok=True)
    tempfile.tempdir=str((folder/'tmp').resolve());sys.dont_write_bytecode=True
    bank_root=WORK/'banks'/source/str(seed)/'fit';pred_root=WORK/'predictions/C00'/source/str(seed)
    from .guard import install
    guard=install(inputs=[*[bank_root/n for n in selected],*[pred_root/n for n in selected],
        *[DATA/'train'/f'{n}.geff' for n in selected]],outputs=[folder],
        code_roots=[REPO/'tools',REPO/'handover',REPO/'work/annotation-selection-v1/official',
                    Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from .actions import Bank,apply_decisions
    from .data import labels
    from .graphs import read_csv,export_csv
    from .evaluation import score,finite
    from annotation_selection.metric_adapter import aggregate
    from pipeline_error_training.labels import SourceLabels
    started=time.monotonic();choices={};parents={}
    for clip in selected:
        root=bank_root/clip;r=read(root/'receipt.json')
        if r.get('diagnostic') or r['training_sha256']!=sha(root/'training.npz') or r['graph_sha256']!=sha(pred_root/clip/'graph.npz'):
            raise Blocked('Retained source witness ancestry differs from the complete bank')
        label=DATA/'train'/f'{clip}.geff'
        parents[clip]=dict(bank_receipt=sha(root/'receipt.json'),training=r['training_sha256'],graph=r['graph_sha256'],
            label_files={str(p.relative_to(label)):sha(p) for p in label.rglob('*') if p.is_file()})
        if len(choices)==2:continue
        with np.load(root/'training.npz') as arrays:
            risks=arrays['risks'];offset=arrays['offset']
            for parent,a,b in zip(r['group_parents'],offset[:-1],offset[1:]):
                y=risks[a:b]
                for kind,target,eligible in [('positive',1,bool((y==1).any())),('negative',0,bool(len(y) and (y==0).all()))]:
                    if eligible and kind not in choices:
                        choices[kind]=dict(clip=clip,parent=parent,index=int(np.flatnonzero(y==target)[0]),target=target,
                            legal_actions=len(y))
    if (folder/'receipt.json').exists():
        prior=read(folder/'receipt.json')
        if prior['parents']!=parents or prior['implementation_sha256']!=sha(Path(__file__)):
            raise Blocked('Retained source witness parents or implementation changed')
        for kind,r in prior['witnesses'].items():
            if r['status']=='measured' and (sha(folder/f'{kind}.json')!=r['detail_sha256'] or sha(folder/f'{kind}.csv')!=r['csv_sha256']):
                raise Blocked('Retained source witness output changed')
        return prior
    witnesses={}
    for kind in ('positive','negative'):
        if kind not in choices:
            witnesses[kind]=dict(status='unavailable',reason='No supported '+kind+' parent group in the inspected complete banks')
            continue
        choice=choices[kind];clip=choice['clip'];prediction=pred_root/clip
        nodes,edges=read_csv(prediction/'submission.csv',clip)
        with np.load(prediction/'graph.npz') as arrays:
            if not np.array_equal(nodes,arrays['nodes']) or not np.array_equal(edges,arrays['edges']):
                raise Blocked('Source witness baseline CSV and frozen graph differ')
            bank=Bank(nodes,edges,arrays['edge_scores'],100)
        group=bank.parent(choice['parent'])
        if not group['complete'] or len(group['forks'])!=choice['legal_actions']:raise Blocked('Source witness denominator differs')
        action=group['forks'][choice['index']]
        gt,ge=labels(DATA/'train'/f'{clip}.geff');lab=SourceLabels(nodes,edges,gt,ge,[1.625,.40625,.40625])
        actual_risk=lab.decision(action)['metric_fork_target']
        if actual_risk!=choice['target']:raise Blocked('Source witness sparse risk shortcut differs from official local action support')
        baseline,_=score(clip,nodes,edges,DATA/'train'/f'{clip}.geff')
        edited,trace=apply_decisions(nodes,edges,[action],[1.],max_fraction=.02,time_limit=2.)
        csv=folder/f'{kind}.csv';exported=export_csv(csv,clip,nodes,edited);restored=read_csv(csv,clip)
        if not all(np.array_equal(a,b) for a,b in zip(exported,restored)):raise Blocked('Source witness CSV roundtrip failed')
        result,_=score(clip,*restored,DATA/'train'/f'{clip}.geff')
        # Selected witnesses always have supported GT edges. Other clips with
        # undefined single-clip scores remain fully represented in the census.
        before=aggregate([baseline],[clip]);after=aggregate([result],[clip])
        detail=dict(choice=choice,solver=trace,baseline=finite(baseline),edited=finite(result),csv_sha256=sha(csv))
        write(folder/f'{kind}.json',detail,immutable=True)
        witnesses[kind]=dict(status='measured',clip=clip,risk=actual_risk,legal_actions=choice['legal_actions'],
            baseline=finite(before),edited=finite(after),score_delta=after['score']-before['score'],
            accepted_actions=len(trace['edits']),changed_edges=trace['changed_edges'],deployment_edit_cap=int(.02*len(edges)),
            solver_and_global_edit_cap_applied=True,detail_sha256=sha(folder/f'{kind}.json'),csv_sha256=sha(csv),
            label_guided_not_model=True,achievable_policy_bound=False)
    receipt=dict(status='complete',source=source,seed=seed,retained_C00=True,source_only=True,
        fit_clips_inspected=len(selected),full_source_fit_population=selected==sorted(registered),
        witnesses=witnesses,guard=guard,parents=parents,implementation_sha256=sha(Path(__file__)),
        selection='Lexically first clip and first canonical supported positive/fully supported negative parent/action',
        interpretation='Label-guided local controls, not learned predictions or achievable policy scores; registered solver and global edit cap applied.',
        wall_seconds=time.monotonic()-started,finished_utc=now())
    write(folder/'receipt.json',receipt,immutable=True);return receipt
