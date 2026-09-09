"""Natural-population metrics, after model locks; never used for outer selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score

from .common import OUT,inventory,load_graph,read_json,write_json


def run(args):
    rows=[]
    for embryo in ['44b6','6bba']:
        source='6bba' if embryo=='44b6' else '44b6';inv=[r for r in inventory() if r['embryo']==embryo]
        scores={};labels={};weights={}
        for r in inv:
            name=r['dataset'];p=load_graph(OUT/'scores'/f'{name}.npz');t=load_graph(OUT/'evaluation/training'/f'{name}.npz')
            m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
            if (OUT/'image_scores'/f'{name}.npz').exists():
                im=load_graph(OUT/'image_scores'/f'{name}.npz');p['F_temporal']=im['score']
                for f in ['0_1','0_3']:
                    if 'fraction_'+f in im:p['F_temporal_fraction_'+f]=im['fraction_'+f]
            for mid,s in p.items():
                if mid.startswith('F_'):yy=m['annotation_label'];ss=s;w=np.ones(len(yy))
                elif mid.startswith('E_'):yy=t['edge_y'];ss=s[t['edge_index']];w=1/t['edge_probability']
                else:yy=t['fork_y'];ss=s[t['fork_index']];w=1/t['fork_probability']
                scores.setdefault(mid,[]).append(ss);labels.setdefault(mid,[]).append(yy);weights.setdefault(mid,[]).append(w)
        for mid in scores:
            y=np.concatenate(labels[mid]);p=np.concatenate(scores[mid]);w=np.concatenate(weights[mid])
            rows.append(dict(source=source,outer_embryo=embryo,model=mid,rows=len(y),positives=int(y.sum()),
                scope='natural_full_node_population' if mid.startswith('F_') else 'known_annotated_transitions_inverse_probability_weighted',
                average_precision=average_precision_score(y,p,sample_weight=w),auroc=roc_auc_score(y,p,sample_weight=w),
                log_loss=log_loss(y,p,sample_weight=w),brier=brier_score_loss(y,p,sample_weight=w),
                natural_prevalence=float(np.average(y,weights=w))))
    p=OUT/'native_classifier_metrics.csv';old=pd.read_csv(p);old=old[old.scope.str.startswith('source_fit')]
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(p,index=False)
    write_json(OUT/'classifier_metrics_receipt.json',dict(outer_rows=len(rows),selection_use='descriptive after freeze',
        node_truth='Matched sparse annotation, not biological false-detection truth',
        edge_fork_truth='Known annotated neighborhoods only; unknown transitions masked',
        image_calibration='Source prior odds, no target-label calibration'))
