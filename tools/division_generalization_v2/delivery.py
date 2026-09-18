"""Verify completed target artifacts and retain portable hashes and error identities."""
from collections import Counter
from pathlib import Path
import numpy as np

from .common import (WORK,RESULTS,inputs,read_json,write_json,sha,now,
                     load_graph,verified_graph,graph_hash)


def run():
    if not (WORK/'queue/complete.json').exists():
        raise ValueError('Delivery verification requires a completed execution queue')
    freeze=read_json(RESULTS/'target_freeze.json')
    barrier=RESULTS/'all_target_predictions_complete.json'
    if read_json(barrier)['freeze_sha256']!=sha(RESULTS/'target_freeze.json'):
        raise ValueError('Target freeze changed after prediction')
    target=read_json(RESULTS/'target_evaluation.json')
    if target['status']!='complete':raise ValueError('Full target evaluation is incomplete')
    records=[];changes=[];raw_changes=[];summaries=[]
    for arm in freeze['qualified_exports']:
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            root=WORK/'target'/arm/str(seed)
            direction_receipt=read_json(root/'both_directions_predicted.json')
            if direction_receipt['clips']!=199 or direction_receipt['freeze_sha256']!=sha(RESULTS/'target_freeze.json'):
                raise ValueError('Incomplete or stale directional export')
            totals=Counter();node_equal=0;edge_equal=0;accepted=0
            for row in inputs():
                dataset=row['dataset'];source='44b6' if row['embryo']=='6bba' else '6bba'
                selected=freeze['selected'][f'{arm}/{source}/{seed}']['selected']
                output=root/'predictions'/dataset
                prediction=output/selected['application']/(dataset+'.npz')
                receipt=prediction.with_suffix('.json')
                score_path=root/'evaluation'/(dataset+'.json')
                error_path=root/'errors'/(dataset+'.json')
                guard_path=output/'guard.json'
                trace_path=output/'trace.json'
                score=read_json(score_path);proof=read_json(receipt);guard=read_json(guard_path)
                if score_path.stat().st_mtime_ns<barrier.stat().st_mtime_ns:
                    raise ValueError('Target score predates the complete-prediction barrier')
                if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
                    raise ValueError('Prediction startup guard failed')
                if proof['source']!=source or proof['partial_fixture'] or not proof['ledger']['exact_outside_owned_actions']:
                    raise ValueError('Prediction routing, completeness or ownership failed')
                if proof['calibration']!=selected['calibration']:
                    raise ValueError('Prediction calibration differs from the frozen source choice')
                actual_sha=sha(prediction)
                if actual_sha!=score['prediction_sha256']:
                    raise ValueError('Scored prediction artifact changed')
                graph=load_graph(prediction);base=verified_graph(row)
                if graph_hash(graph['nodes'],graph['edges'])!=proof['graph_hash']:
                    raise ValueError('Prediction graph content changed')
                np.testing.assert_array_equal(graph['nodes'],base['nodes'])
                if len(graph['nodes'])!=score['num_pred_nodes']:
                    raise ValueError('Scored node count changed')
                same_edges=np.array_equal(graph['edges'],base['edges'])
                node_equal+=1;edge_equal+=int(same_edges)
                if not same_edges:
                    previous=set(map(tuple,base['edges']));current=set(map(tuple,graph['edges']))
                    raw_changes.append(dict(arm=arm,seed=seed,dataset=dataset,
                        prediction_edges_added=sorted(current-previous),
                        prediction_edges_removed=sorted(previous-current)))
                accepted+=proof['ledger']['accepted_actions']
                error=read_json(error_path)
                identities={k:v for k,v in error.items() if isinstance(v,list)}
                totals.update({k:len(v) for k,v in identities.items()})
                if any(identities.values()):
                    changes.append(dict(arm=arm,seed=seed,dataset=dataset,identities=identities))
                paths=dict(prediction=prediction,receipt=receipt,score=score_path,error=error_path,
                           guard=guard_path,trace=trace_path)
                records.append(dict(arm=arm,seed=seed,dataset=dataset,source=source,
                    application=selected['application'],graph_hash=proof['graph_hash'],
                    artifacts={name:dict(path=str(path.relative_to(WORK)),sha256=sha(path))
                               for name,path in paths.items()}))
            summaries.append(dict(arm=arm,seed=seed,clips=node_equal,exact_P0_nodes=node_equal,
                exact_P0_edges=edge_equal,accepted_actions=accepted,transitions=dict(totals)))
    expected=199*sum(1 if arm=='G30' else 2 for arm in freeze['qualified_exports'])
    if len(records)!=expected or len({(r['arm'],r['seed'],r['dataset']) for r in records})!=expected:
        raise ValueError('Missing or duplicate complete target artifacts')
    write_json(RESULTS/'target_artifact_manifest.json',dict(status='verified',work_root='work/division-generalization-v2',
        freeze_sha256=sha(RESULTS/'target_freeze.json'),records=records,
        reconstruction='Run the isolated study queue with input_manifest.json inputs and the frozen source choices; paths are relative to the work root'))
    write_json(RESULTS/'error_identity_changes.json',dict(status='measured',summaries=summaries,changes=changes,
        raw_prediction_graph_changes=raw_changes,
        raw_change_scope='Complete prediction edge identities; scored error transitions are separate and sparse annotations do not establish biological correctness',
        empty_reason=None if changes else 'No recovered/lost supported edges or divisions and no added/removed official false positives',
        categories_overlap=True,full_per_clip_identities='See the hashed error artifacts in target_artifact_manifest.json'))
    runtime=[read_json(p) for p in (WORK/'target_matrix').glob('*/complete.json')]
    seconds=sum(r['seconds'] for r in runtime);anchors=sum(r['counts']['anchors'] for r in runtime)
    projection_path=RESULTS/'target_runtime_projection.json'
    projection=read_json(projection_path) if projection_path.exists() else {}
    projection.update(recorded=now(),status='measured_complete',completed_clips=len(runtime),
        completed_anchors=anchors,full_clip_wall_seconds=seconds,observed_anchors_per_second=anchors/seconds if seconds else None,
        linear_remaining_inference_hours=0.,target_labels_opened=True,
        projection_scope='Completed matrix-worker wall times; excludes per-process startup, scoring and diagnostic review')
    write_json(RESULTS/'target_runtime_projection.json',projection)
    result=dict(status='verified',recorded=now(),expected_target_graphs=expected,verified_target_graphs=len(records),
        summaries=summaries,changed_prediction_clips=len(raw_changes),target_scores_after_all_predictions=True,
        graph_content_and_scored_file_hashes_verified=True,all_startup_guards_verified=True,
        artifact_manifest_sha256=sha(RESULTS/'target_artifact_manifest.json'),
        target_evaluation_sha256=sha(RESULTS/'target_evaluation.json'),
        verification_code_sha256=sha(Path(__file__)))
    write_json(RESULTS/'delivery_validation.json',result)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(run()))
