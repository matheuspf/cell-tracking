"""Post-freeze diagnostic of exact-time candidate censoring on observed GT."""
from collections import defaultdict
import pandas as pd
from .common import *

def run():
    assert (OUT/'prediction_lock.json').exists() and (OUT/'ablation_scores.csv').exists()
    rows=[];primary=['C1short','C1','C2','C3','C4','C5','C6']
    complete=set(pd.read_csv(OUT/'ablation_scores.csv').variant)
    additional=[p.name for p in (OUT/'additional_candidate_evidence').glob('*') if p.is_dir() and p.name in complete]
    for row in inputs():
        name=row['dataset'];g=arrays(V3/'selected_predictions'/f'{name}.npz');gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
        matched=arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches']
        id_to_index={int(v):i for i,v in enumerate(g['nodes'][:,0])}
        observed={int(b):id_to_index[int(a)] for a,b in matched}
        evidence=arrays(OUT/'candidate_evidence'/f'{name}.npz')
        margins=arrays(OUT/'margin_candidates'/f'{name}.npz')
        children=defaultdict(list)
        baseline_children=defaultdict(set)
        for a,b in g['edges']:baseline_children[int(a)].add(int(b))
        for a,b in gt['edges']:children[int(a)].append(int(b))
        for arm in primary+additional:
            if arm in primary:
                anchors=evidence['anchors'];c=evidence['candidates'];gate=evidence['gate_'+arm];events=margins[arm]
            else:
                extra=arrays(OUT/'additional_candidate_evidence'/arm/f'{name}.npz')
                anchors=extra['anchors'];c=extra['candidates'];gate=extra['gate'];events=extra['margin_events']
            anchor_rows={int(a):i for i,a in enumerate(anchors)}
            proposed={int(p):{int(a),int(b)} for p,a,b in events};current_children=defaultdict(set)
            for a,b in arrays(OUT/'predictions'/arm/f'{name}.npz')['edges']:current_children[int(a)].add(int(b))
            counts=dict(observed_gt_forks=0,parent_matched=0,both_daughters_matched=0,exact_frame_evidence=0,within_six_cap=0,after_gate=0,after_margin=0,after_decoder_new_edits=0)
            for parent,ch in children.items():
                if len(ch)!=2:continue
                counts['observed_gt_forks']+=1
                if parent not in observed:continue
                counts['parent_matched']+=1
                if not all(x in observed for x in ch):continue
                counts['both_daughters_matched']+=1
                p=observed[parent];daughters=[observed[x] for x in ch]
                if p not in anchor_rows or not all(g['nodes'][d,1]==g['nodes'][p,1]+1 for d in daughters):continue
                counts['exact_frame_evidence']+=1;i=anchor_rows[p]
                if not set(daughters)<=set(c[i]):continue
                counts['within_six_cap']+=1
                if gate[i]<=.5:continue
                counts['after_gate']+=1
                if proposed.get(p)!=set(daughters):continue
                counts['after_margin']+=1
                pid=int(g['nodes'][p,0]);dids={int(g['nodes'][d,0]) for d in daughters}
                counts['after_decoder_new_edits']+=int(current_children[pid]==dids and baseline_children[pid]!=dids)
            rows.append(dict(dataset=name,embryo=row['embryo'],variant=arm,**counts))
    df=pd.DataFrame(rows);df.to_csv(OUT/'candidate_gt_coverage_rows.csv',index=False)
    grouped=df.groupby(['variant','embryo']).sum(numeric_only=True).reset_index()
    grouped.to_csv(OUT/'candidate_gt_coverage.csv',index=False)
    write(OUT/'candidate_coverage_receipt.json',dict(completed=True,post_outcome_diagnostic=True,used_to_tune=False,
        matching='C0 official node matches, immediate GT daughter IDs; this is not the tolerant official division matching criterion. Decoder coverage counts newly accepted edits, excluding unchanged protected incumbent forks.',
        prediction_lock_sha256=sha(OUT/'prediction_lock.json')))
