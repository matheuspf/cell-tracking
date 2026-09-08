"""Image-content overlap evidence and conservative source/outer locks."""
from __future__ import annotations

from collections import Counter,defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .common import OUT,REPO,SEED,digest,now,read_json,sha,stage,write_json


def overlap_evidence(names):
    lookup=defaultdict(list)
    for name in names:
        for h,*pos in read_json(OUT/'fingerprints'/f'{name}.json'):
            lookup[(name.split('_')[0],h)].append((name,pos))
    votes=defaultdict(Counter)
    for hits in lookup.values():
        for (a,pa),(b,pb) in combinations(hits,2):
            if a==b:continue
            if a>b:a,b,pa,pb=b,a,pb,pa
            votes[(a,b)][tuple(np.array(pa)-np.array(pb))]+=1
    rows=[]
    for (a,b),counts in votes.items():
        offset,n=counts.most_common(1)[0]
        if n>=3:
            rows.append(dict(dataset_a=a,dataset_b=b,offset_tzyx=list(map(int,offset)),exact_patch_matches=n,
                             evidence='Repeated identical 3x5x5 raw image patches at image-predicted local maxima'))
    return rows


def run(args):
    inv=read_json(OUT/'inventory.json');names=[r['dataset'] for r in inv]
    if any(not (OUT/'fingerprints'/f'{n}.json').exists() for n in names):
        raise ValueError('All expected image fingerprints required')
    overlap=overlap_evidence(names)
    write_json(OUT/'overlap_registration.json',overlap)
    embryos=sorted({r['embryo'] for r in inv})
    # The released hashed filenames contain no crop/time origins. Exact matching
    # proves some overlaps; failure to match sampled anchors is not disjointness.
    # All unregistered relationships remain possible, hence one conservative
    # supergroup per embryo. This explicitly disables unsafe inner tuning.
    groups={r['dataset']:r['embryo']+'_unresolved_overlap' for r in inv}
    manifest=dict(schema_version=1,created=now(),embryos=embryos,expected_samples=names,
                  data_hash=read_json(OUT/'inventory_summary.json')['data_hash'],overlap_groups=groups,
                  positive_overlap_pairs=len(overlap),overlap_proof='Exact local image patches with consistent translation; unknown relations conservatively joined',
                  global_transform_census_valid=False,inner_validation='unavailable: one conservative overlap supergroup per source embryo',
                  directions=[dict(source=e,outer=[x for x in embryos if x!=e][0],
                                   source_samples=[r['dataset'] for r in inv if r['embryo']==e],
                                   outer_samples=[r['dataset'] for r in inv if r['embryo']!=e],
                                   inner_folds=[],upstream_supervision='none: fixed image-only detector/linker') for e in embryos],
                  embargo_state='locked_splits_before_fit',outer_evaluation_start=None)
    if len(embryos)!=2:raise ValueError('Study requires explicit adaptation to rediscovered embryo count')
    for d in manifest['directions']:
        if set(groups[n] for n in d['source_samples']) & set(groups[n] for n in d['outer_samples']):
            raise ValueError('Overlap across outer split')
    for d in manifest['directions']:
        write_json(OUT/f'expected_{d["outer"]}.json',d['outer_samples'],immutable=True)
    write_json(OUT/'fold_manifest.json',manifest,immutable=True)
    df=pd.read_csv(OUT/'sample_inventory.csv');df['overlap_group']=df.dataset.map(groups);df.to_csv(OUT/'sample_inventory.csv',index=False)
    original=read_json(REPO/'handover/annotation-selection-v1/experiments.json')
    resolved=dict(original=original,created=now(),split_hash=sha(OUT/'fold_manifest.json'),
                  candidate_config_hash=sha(OUT/'candidate_config.json'),feature_schema_hash=sha(OUT/'feature_schema.json'),
                  tabular_max_fit_candidates=120000,tabular_sampling='uniform without replacement, natural prevalence, source only',
                  class_weighting=False,primary_model='hgb_all_leaf7',primary_policy='membership_tracklets',primary_keep=.9,
                  primary_selection='Fixed before outer outcomes; no valid inner groups, so no source tuning or early stopping',
                  image_epochs=6,image_fit_candidates=30000,image_batch_size=256,image_optimizer='AdamW lr=0.001 weight_decay=0.001',
                  image_architecture='3-channel triplanar Conv2d 16/32/48 channels, global average pool, 32-wide head',
                  image_jitter='random shared x/y flips, random integer roll -1..1 with reflected input borders; applied to both labels',
                  image_checkpoint_choice='fixed epoch 6; no validation-based selection due unresolved source overlap',
                  image_normalizer='source-sample mean/std only',image_seeds=[SEED,314159],
                  sensitivity_models=['hgb_all_leaf7_ambiguous_excluded','hgb_all_leaf7_high_confidence'],
                  controls=['groupwise_tracklet_label_shuffle','independent_tracklet_random_labels','post_lock_oracle'],
                  public_pilot_samples=[names[0],next(n for n in names if n.split('_')[0]!=names[0].split('_')[0])],
                  deviations_before_reveal=['No independent source inner groups can be certified from released metadata; fixed settings replace inner tuning, uncertainty cannot assume independent clips',
                                            'Primary classical linker is one-to-one with no learned division rule; division recall ceiling is explicitly zero',
                                            'Pilot chosen from image quantiles only, spanning both source directions; no outcomes viewed'])
    write_json(OUT/'preregistration.json',resolved,immutable=True)
    stage('S010','complete',samples=len(names),embryos=len(embryos),positive_overlap_pairs=len(overlap),conservative_groups=len(embryos))
    print(f'Locked {len(names)} samples; {len(overlap)} proven overlap pairs; {len(embryos)} conservative supergroups',flush=True)
