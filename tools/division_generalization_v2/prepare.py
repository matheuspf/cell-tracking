"""Source Bernoulli anchor sample plus separately marked positive exposure."""
from collections import Counter
from dataclasses import asdict
import gzip
import json
import time

import numpy as np

from .common import (DATA, RESULTS, WORK, PRIOR_WORK, inputs, read_json, write_json,
                     verified_graph, verified_evidence, save_arrays, sha, digest)
from .contracts import SAMPLE_PROBABILITY, SAMPLING_SEED
from .actions import EventBank, canonical_actions
from .labels import Counterfactual, SourceUtility, COUNT_KEYS
from .features import build


def sampled_anchor(name, persisted_id, probability=SAMPLE_PROBABILITY):
    # Stateless seeded probability sample; component IDs and labels are absent.
    draw = int(digest([SAMPLING_SEED,name,int(persisted_id)])[:16],16)/2**64
    return draw < probability


def serial_decision(d):
    out = asdict(d)
    for key in ('remove','add','resources'):
        out[key] = sorted(out[key])
    return out


def one(row, source, utility):
    from center_comparison.pipeline import read_gt
    from annotation_selection.metric_adapter import evaluate_graph
    if row['embryo'] != source:
        raise PermissionError('Source preparation rejects target labels')
    name = row['dataset']
    folder = WORK/'source'/source/name
    receipt_path = folder/'receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for rel, expected in receipt['files'].items():
            if sha(folder/rel) != expected:
                raise ValueError('Prepared source artifact changed')
        return receipt
    folder.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    graph, native = verified_graph(row), verified_evidence(row)
    gn,ge = read_gt(DATA,name,row['physical_scale'])
    baseline = read_json(PRIOR_WORK/'evaluation/P0'/(name+'.json'))
    labeler = Counterfactual(graph['nodes'],graph['edges'],gn,ge,row['physical_scale'],baseline)
    bank = EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    random = {p for p in bank.expanded if sampled_anchor(name,graph['nodes'][p,0])}
    # Positive oversampling is training-only, never used in inference enumeration
    # or for estimating the supported population prevalence.
    extra = set()
    for roles in labeler.l.roles.values():
        if roles is not None:
            ps,_ = roles
            extra.update(ps)
            extra.update(q for p in ps for q in bank.near[p])
    extra &= bank.expanded
    selected = sorted(random | extra)
    counts = Counter(all_anchors=len(bank.expanded), random_anchors=len(random),
                     positive_stream_anchors=len(extra), sampled_anchors=len(selected))
    groups, files, parities = [], {}, []
    covered = set()
    # Random full graph parity edits are selected without looking at their label.
    parity_remaining = 2
    for parent in selected:
        records,reject = canonical_actions(bank,parent,replace=True)
        if not records:
            continue
        labels = [labeler.label(d) for d,_ in records]
        known = np.array([l['supported'] for l in labels])
        counts['unknown_actions'] += int((~known).sum())
        counts['all_actions'] += len(records)
        if not known.any():
            counts['unknown_only_anchors'] += 1
            continue
        # Preserve unknown alternatives in the archive; masks exclude them from
        # both loss numerator and denominator.
        for l in labels:
            l['utility'] = utility.value(l['delta'], baseline)
            covered.update(l['compatible_events'])
        positives = any(l['biological']==1 for l in labels)
        group_id = labeler.l.groups.get(parent)
        if group_id is None:
            candidates = {labeler.l.groups[n] for d,_ in records for e in d.remove | d.add
                          for n in e if n in labeler.l.groups}
            group_id = min(candidates) if candidates else f'anchor-{parent}'
        rel = f'{parent}.npz'
        arrays = build(bank,parent,records)
        arrays.update(utility=np.asarray([l['utility'] for l in labels],np.float64),
                      supported=known, biological=np.asarray([l['biological'] for l in labels],np.int8),
                      identity=np.asarray([l['identity'] for l in labels],np.int8),
                      delta=np.asarray([l['delta'] for l in labels],np.int16))
        save_arrays(folder/rel,**arrays)
        files[rel] = sha(folder/rel)
        metadata = dict(anchor=parent,node_id=int(graph['nodes'][parent,0]),
            time=int(graph['nodes'][parent,1]),position=graph['nodes'][parent,2:].tolist(),
            group=f'{name}:{group_id}',dataset=name,positive=positives,
            random_included=parent in random,random_probability=SAMPLE_PROBABILITY,
            positive_stream_included=parent in extra, arrays=rel, actions=len(records),
            supported_actions=int(known.sum()),
            decisions=[serial_decision(d) for d,_ in records],labels=labels)
        groups.append(metadata)
        counts['supported_anchors'] += 1
        counts['positive_anchors' if positives else 'negative_only_anchors'] += 1
        # One arbitrary edit and a nontrivial utility edit, across all 199 clips,
        # give independent complete scorer checks of edge and fork assignment.
        if parity_remaining and parent in random:
            candidates = [i for i,(d,_) in enumerate(records) if d.add or d.remove]
            if candidates:
                i = candidates[int(digest([name,parent])[:8],16)%len(candidates)]
                d,_ = records[i]
                ix = {int(n[0]):i for i,n in enumerate(graph['nodes'])}
                before = {(ix[int(a)],ix[int(b)]) for a,b in graph['edges']}
                edited = (before-set(d.remove)) | set(d.add)
                edges = np.asarray([(graph['nodes'][a,0],graph['nodes'][b,0]) for a,b in sorted(edited)],np.int64).reshape(-1,2)
                score,_,_ = evaluate_graph(name,graph['nodes'],edges,gn,ge,row['physical_scale'],row['estimated_total'])
                actual = [score[k]-baseline[k] for k in COUNT_KEYS]
                if actual != labels[i]['delta']:
                    write_json(folder/'invalid_label_parity.json',dict(parent=parent,decision=serial_decision(d),
                               expected=actual,fast=labels[i]['delta']))
                    raise ValueError('Complete action count parity failed')
                parities.append(dict(parent=parent,key=d.key,delta=actual,full_graph_replay=True))
                parity_remaining -= 1
    meta = folder/'anchors.json.gz'
    with gzip.open(meta,'wt') as f:
        json.dump(groups,f,separators=(',',':'),allow_nan=False)
    files[meta.name] = sha(meta)
    receipt = dict(dataset=name,source=source,counts=dict(counts),files=files,
        bank_sha256=bank.hash,covered_events=len(covered),annotated_events=len(labeler.l.roles),
        groups=len({g['group'] for g in groups}),counterfactual_parities=parities,
        seconds=time.monotonic()-start,prior_graph_sha256=row['baselines']['P0']['sha256'],
        feature_labels_separate=True,unknown_as_negative_count=0)
    write_json(receipt_path,receipt)
    return receipt


def run(args):
    source = args.source
    rows = inputs(source)
    utility = SourceUtility([read_json(PRIOR_WORK/'evaluation/P0'/(r['dataset']+'.json')) for r in rows])
    selected = [r for r in rows if not args.clip or r['dataset']==args.clip]
    result = []
    for i,row in enumerate(selected,1):
        r = one(row,source,utility)
        result.append(r)
        print(f'Prepared {source} {i}/{len(selected)} {row["dataset"]}: {r["counts"]} ({r["seconds"]:.1f}s)',flush=True)
    if args.clip:
        return result
    write_json(WORK/'source'/source/'manifest.json',dict(source=source,clips=result,
        source_only=True,random_sampling_probability=SAMPLE_PROBABILITY,
        utility_denominators=vars(utility),unknown_as_negative_count=0))
    return result
