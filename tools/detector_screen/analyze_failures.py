"""Locate the incumbent's residual errors and measure challenger complementarity."""
import json
import numpy as np
from .evaluate import ROOT


def main():
    panel=json.loads((ROOT/'panel.json').read_text())
    rows={r['key']:r for r in panel['frames'] if r['role']=='assessment'}
    truth=json.loads((ROOT/'evaluation/ground_truth.json').read_text())['frames']
    evaluations={}
    for p in (ROOT/'evaluation').glob('*.json'):
        obj=json.loads(p.read_text())
        if 'method' not in obj or 'per_frame' not in obj:continue
        selected={r['key']:r for r in obj['per_frame'] if r['role']=='assessment'}
        if selected.keys()==rows.keys():evaluations[obj['method']]=selected
    baseline=evaluations['incumbent'];cases=[]
    for key,row in rows.items():
        gt=np.asarray(truth[key]['gt_nodes']).reshape(-1,5)
        misses=[g for g in gt if int(g[0]) not in baseline[key]['matched_gt_at_7']]
        if not misses:continue
        with np.load(ROOT/'predictions/incumbent'/(key+'.npz')) as prediction:
            centers=np.clip(np.rint(prediction['centers_zyx']),0,np.array(row['shape'])-1)
        for node in misses:
            distances=np.linalg.norm((centers-node[2:])*row['spacing_um'],axis=1)
            distance=float(distances.min())
            cases.append(dict(key=key,embryo=row['embryo'],gt_node_id=int(node[0]),
                nearest_incumbent_distance_um=distance,
                cause='one-to-one competition' if distance<=7 else 'no candidate within7um',
                recovered_by=[m for m,fr in evaluations.items() if m!='incumbent' and
                              int(node[0]) in fr[key]['matched_gt_at_7']],
                recovered_at_same_budget_by=[m for m,fr in evaluations.items() if m!='incumbent' and
                              int(node[0]) in fr[key]['budgets']['1.0']['matched_gt_at_7']]))
    result=dict(cases=cases,total_misses=len(cases),
        one_to_one_competition=sum(c['nearest_incumbent_distance_um']<=7 for c in cases),
        distance_7_to_9_um=sum(7<c['nearest_incumbent_distance_um']<=9 for c in cases),
        distance_over_9_um=sum(c['nearest_incumbent_distance_um']>9 for c in cases),
        distance_7_to_7_1_um=sum(7<c['nearest_incumbent_distance_um']<=7.1 for c in cases),
        interpretation='Post-hoc diagnostic of measured detector errors. Recovered_by is evidence of complementarity, not a deployable ensemble or a predicted competition-score gain. Same-budget recovery caps each challenger frame at the incumbent candidate count, using only its own confidence rank. No predictions are altered.')
    (ROOT/'evaluation/failure-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
