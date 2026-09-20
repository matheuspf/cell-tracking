"""Exercise final attribution on existing C01 source-calibration decisions only."""
from pathlib import Path
import sys,tempfile,heapq
from .common import WORK,REPO,DATA,read,write,sha,now


def run():
    source='44b6';seed=20260918
    # Fixed diagnostic cases from the retained source safety result. These are
    # branch-coverage controls, never an accuracy estimate or model selection.
    clips=['44b6_2a2eff9f','44b6_81c256f0']
    partition=read(WORK/'source_partitions.json')[source]['calibration']
    assert all(n in partition for n in clips)
    root=WORK/'checks/source_attribution';root.mkdir(parents=True,exist_ok=True)
    (root/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((root/'tmp').resolve())
    calibration=WORK/'fits'/source/str(seed)/'calibration/C01'
    package=WORK/'packages'/source/str(seed)/'C01'
    inputs=[package/'calibration.json']
    for clip in clips:
        old=WORK/'predictions/C00'/source/str(seed)/clip
        inputs.extend([old,calibration/clip,DATA/'train'/f'{clip}.geff'])
        base=root/'predictions/C00'/source/str(seed)/clip;base.mkdir(parents=True,exist_ok=True)
        dest=root/'predictions/C01'/source/str(seed)/clip;dest.mkdir(parents=True,exist_ok=True)
        for link,target in [(base/'graph.npz',old/'graph.npz'),(dest/'policy_logits.npz',calibration/clip/'logits.npz'),
                            (dest/'policy_progress.json',calibration/clip/'progress.json')]:
            if not link.exists():link.symlink_to(target)
    copied_package=root/'packages'/source/str(seed)/'C01';copied_package.mkdir(parents=True,exist_ok=True)
    if not (copied_package/'calibration.json').exists():(copied_package/'calibration.json').symlink_to(package/'calibration.json')
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=inputs,outputs=[root],code_roots=[REPO/'tools',REPO/'handover',REPO/'work/annotation-selection-v1/official',
        Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from . import diagnostics
    from .calibration_cache import cached_utility
    from .graphs import read_csv
    from .evaluation import score,finite
    # The exact reporting routine reads a separate fixture tree containing only
    # links to these source artifacts. The real target prediction tree is absent.
    diagnostics.WORK=root
    results=[]
    for clip in clips:
        dest=root/'predictions/C01'/source/str(seed)/clip;tail=[]
        with np.load(calibration/clip/'logits.npz') as f:
            cache={k:f[k] for k in f.files}
        for p,a,start,end,c in zip(cache['parents'],cache['occurrence'],cache['offset'][:-1],cache['offset'][1:],cache['competitor']):
            conditional=cache['conditional'][start:end]
            raw=cached_utility(float(a),conditional,cache['structural'][start:end],c,1.,0.,0.)
            item=(float(raw.max()),int(p),float(a),len(raw))
            if len(tail)<50:heapq.heappush(tail,item)
            elif item>tail[0]:heapq.heapreplace(tail,item)
        trial=read(calibration/clip/'margin-2.json')
        write(dest/'policy_trace.json',dict(raw_top_groups=sorted(tail,reverse=True),solver=trial['solver']))
        nodes,edges=read_csv(calibration/clip/'margin-2.csv',clip)
        metrics,events=score(clip,nodes,edges,DATA/'train'/f'{clip}.geff')
        for key in ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'):
            assert metrics[key]==trial['metrics'][key],key
        assert abs(metrics['adj_edge_jaccard']-trial['metrics']['adj_edge_jaccard'])<1e-12
        details=diagnostics.target_details(source,seed,'C01',clip,nodes,edges,events)
        selected=details['selected_action_local_risk']
        assert sum(selected.get(k,0) for k in ('supported_compatible','supported_incorrect','unknown'))==len(trial['solver']['edits'])
        write(dest/'details.json',finite(details))
        row=dict(clip=clip,accepted_actions=len(trial['solver']['edits']),official_counts={k:metrics[k] for k in
            ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')},
            stage_counts=details['stage_counts'],selected_action_local_risk=selected,raw_tail_groups=len(details['raw_tail']),
            details_sha256=sha(dest/'details.json'),calibration_graph_metrics_reproduced=True)
        results.append(row);print(row,flush=True)
        del cache,details,nodes,edges
    result=dict(status='passed',source=source,seed=seed,guard=guard,scope='Two retained source calibration diagnostic cases only',
        target_labels_opened=False,model_parameters_updated=False,source_calibration_selection_changed=False,
        results=results,implementation_sha256=sha(Path(__file__)),attribution_code_sha256=sha(Path(__file__).with_name('diagnostics.py')),finished_utc=now())
    write(root/'receipt.json',result);return result


if __name__=='__main__':run()
