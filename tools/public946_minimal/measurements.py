"""Compact, descriptive evidence for the completed registered experiment matrix.

This module runs only in the reporting/evaluation process. It never supplies
diagnostics, historical graphs, or matching results to prediction workers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .common import ARMS, REPO, digest, load_arrays, now, read_json, sha, write_json


def graph_edits(before, after):
    """Exact spatiotemporal multisets: detector variants may renumber node IDs."""
    def keys(graph):
        coordinates = {int(r[0]): tuple(map(int, r[1:])) for r in graph['nodes']}
        nodes = Counter(coordinates.values())
        edges = Counter((coordinates[int(a)], coordinates[int(b)]) for a, b in graph['edges'])
        return nodes, edges
    a, ae = keys(before)
    b, be = keys(after)
    return dict(nodes_added_exact=sum((b-a).values()), nodes_removed_exact=sum((a-b).values()),
                edges_added_exact=sum((be-ae).values()), edges_removed_exact=sum((ae-be).values()))


def telemetry_summary(out):
    path = out/'telemetry.jsonl'
    if not path.exists():
        return dict(status='unavailable', cuda_event_active_seconds=None)
    totals = defaultdict(Counter)
    previous = {}
    with path.open() as handle:
        for line in handle:
            try:
                value = __import__('json').loads(line)
            except ValueError:
                continue  # A concurrently written final line can be incomplete.
            job = Path(value['job'])
            if 'predictions' not in job.parts:
                continue
            i = job.parts.index('predictions')
            # Packaged workers have predictions/<stem>/job.json, unlike study
            # workers' predictions/<scope>/<arm>/<stem>/job.json.
            key = '/'.join(job.parts[i+1:i+3]) if len(job.parts)-i >= 5 else 'package/'+job.parts[i-1]
            rec = totals[key]
            identity = (value['pid'], str(job))
            t = value['unix_time']
            dt = t-previous.get(identity,t-1)
            previous[identity] = t
            rec['samples'] += 1
            rec['max_framebuffer_mib'] = max(rec['max_framebuffer_mib'],value.get('framebuffer_mib') or 0)
            if value.get('sm_percent') is None or not 0 < dt <= 5:
                rec['unavailable_or_gapped_samples'] += 1
                continue
            rec['observed_seconds'] += dt
            rec['positive_sm_sample_seconds'] += dt * (value['sm_percent'] > 0)
            rec['sm_utilization_integral_seconds'] += dt * value['sm_percent']/100
    return dict(status='sampled', cuda_event_active_seconds=None, groups=dict(totals),
        definition='One-second nvidia-smi pmon per-process samples; positive-SM sample duration and utilization integral are estimates, not CUDA-event kernel time. Missing samples remain unavailable.',
        telemetry_sha256=sha(path))


def collect(args, destination):
    import numpy as np
    from .study import used_seconds
    rows = read_json(args.out/'image_inventory.json')
    by_name = {r['dataset']:r for r in rows}
    records, ledger = {}, {}
    names = [r['dataset'] for r in rows]
    for arm in ARMS:
        root = args.out/'predictions/full'/arm
        receipt_paths = sorted(root.glob('*/complete.json'))
        if not receipt_paths:
            continue
        parent = 'B1' if arm.startswith('X') else 'B0'
        totals = Counter()
        stages = defaultdict(Counter)
        mechanisms = Counter()
        displacement = []
        clip_ledger = {}
        worst = []
        for index,p in enumerate(receipt_paths,1):
            r = read_json(p)
            name = r['dataset']
            job = read_json(p.parent/'job.json')
            graph = load_arrays(p.parent/'final.npz')
            if sha(p.parent/'final.npz') != r['final_sha256']:
                raise ValueError('Prediction changed before reporting: '+str(p))
            totals['worker_seconds'] += r['seconds']
            totals['fresh_tracking_model_clips'] += int(r['neural_fresh'])
            totals['replayed_tracking_model_clips'] += int(not r['neural_fresh'])
            totals['annotation_guard_passed_clips'] += int(r['annotation_access_denied'])
            totals['gpu_allocated_peak_bytes'] = max(totals['gpu_allocated_peak_bytes'],r['gpu_peak_bytes'])
            totals['rss_peak_bytes'] = max(totals['rss_peak_bytes'],r['rss_peak_bytes'])
            totals['prediction_output_bytes'] += sum(x.stat().st_size for x in p.parent.iterdir() if x.is_file())
            totals['raw_rounded_outside_nodes'] += r['original_validation']['outside']
            totals['final_nodes'] += r['validation']['nodes']
            totals['final_edges'] += r['validation']['edges']
            totals['final_forks'] += r['validation']['forks']
            totals['deepcenter_maps_generated'] += r['heatmaps']['generated']
            totals['deepcenter_cache_reads'] += r['heatmaps']['reads']
            for stage in r['stages']:
                stages[stage['stage']].update({k:stage[k] for k in ('nodes','edges')})
            for k,v in r['stats'].items():
                if isinstance(v,(int,float)):
                    mechanisms[k] += v
            for component in r['mechanisms'].get('E01',[]):
                mechanisms['E01_components'] += 1
                for flag in ('accepted','eligible','degree_preserved','no_fork','complete_evidence'):
                    mechanisms['E01_'+flag] += int(component[flag])
            if 'E02' in r['mechanisms']:
                e02 = r['mechanisms']['E02']
                for k in ('eligible_originals','shifted_originals'):
                    mechanisms['E02_'+k] += e02[k]
                d = np.asarray(e02['displacement_voxels'],dtype=float).reshape(-1,3)
                displacement.append(np.linalg.norm(d*np.asarray(by_name[name]['scale']),axis=1))
            neural = read_json(Path(r['neural_evidence']).parent/'neural.json')
            summaries = neural['summaries']
            compaction_path=p.parent/'evidence_compaction.json'
            compaction=read_json(compaction_path) if compaction_path.exists() else None
            if compaction:
                totals['dense_evidence_bytes_released'] += compaction['original_bytes']-compaction['compact_bytes']
            totals['candidate_probability_columns'] += sum(x['shape'][1] for x in summaries['association_transitions'])
            totals['candidate_probability_entries'] += sum(x['shape'][0]*x['shape'][1] for x in summaries['association_transitions'])
            totals['probability_column_max_error'] = max(totals['probability_column_max_error'],max(
                (x['column_sum_max_error'] for x in summaries['association_transitions']),default=0))
            base_path = args.out/'predictions/full'/parent/name/'final.npz'
            edits = graph_edits(load_arrays(base_path),graph) if base_path.exists() else None
            if edits:
                totals.update(edits)
                totals['exact_graph_identity_clips'] += int(not any(edits.values()))
            score_path = args.out/'scores/full'/arm/(name+'.json')
            score = read_json(score_path) if score_path.exists() else None
            baseline_path = args.out/'scores/full'/parent/(name+'.json')
            if score:
                totals['original_export_lower_clamped_nodes'] += score.get('original_export_lower_clamped_nodes',0)
                totals['original_export_outside_nodes'] += score.get('original_export_outside_nodes',0)
                totals['localization_sum_um'] += score['localization_sum_um']
                totals['localization_count'] += score['localization_count']
            if score and baseline_path.exists():
                base = read_json(baseline_path)
                a, b = set(map(tuple,base['tp_gt_edges'])),set(map(tuple,score['tp_gt_edges']))
                worst.append(dict(dataset=name,embryo=by_name[name]['embryo'],
                    edge_tp_delta=score['edge_tp']-base['edge_tp'],edge_fp_delta=score['edge_fp']-base['edge_fp'],
                    original_tp_lost=len(a-b),new_tp=len(b-a)))
            clip_ledger[name] = dict(job_sha256=r['job_sha256'],job_fingerprint=r['fingerprint'],
                neural_fingerprint=job['neural_fingerprint'],input_sha256=by_name[name]['image_sha256'],
                final_sha256=r['final_sha256'],graph_hash=r['graph_hash'],stages=r['stages'],
                actual_modules=r['modules'],concurrent_worker_slots=job.get('concurrent_worker_slots',1),
                annotation_access_denied=r['annotation_access_denied'],neural_fresh=r['neural_fresh'],
                neural_evidence_sha256=neural['evidence_sha256'],neural_provenance=neural['provenance'],
                original_dense_evidence_sha256=neural.get('original_dense_evidence_sha256',neural['evidence_sha256']),
                evidence_format=neural.get('evidence_format','full_native_probabilities'),
                original_export_outside_nodes=score.get('original_export_outside_nodes') if score else None,
                validation=r['validation'],edits=edits,metric_fingerprint=score['fingerprint'] if score else None)
            if index % 50 == 0:
                print(f'Report evidence {arm}: {index}/{len(receipt_paths)} clips',flush=True)
        if displacement:
            d = np.concatenate(displacement)
            displacement_summary = dict(count=len(d),mean_um=float(d.mean()) if len(d) else None,
                p50_um=float(np.percentile(d,50)) if len(d) else None,
                p95_um=float(np.percentile(d,95)) if len(d) else None,max_um=float(d.max()) if len(d) else None)
        else:
            displacement_summary = None
        records[arm] = dict(parent=parent,expected=names,completed=sorted(clip_ledger),
            missing=sorted(set(names)-set(clip_ledger)),totals=dict(totals),stages=dict(stages),
            mechanism_counts=dict(mechanisms),subvoxel_displacement=displacement_summary,
            matched_localization_mean_um=totals['localization_sum_um']/totals['localization_count'] if totals['localization_count'] else None,
            worst_tp_survival_clips=sorted(worst,key=lambda r:(-(r['original_tp_lost']-r['new_tp']),r['dataset']))[:5],
            clip_ledger_sha256=digest(clip_ledger),
            edit_definition='Exact multisets of (time,z,y,x) nodes and their endpoint-coordinate edges; no geometric tolerance, no cross-detector ID assumption. Coordinate refinements count as remove+add in these diagnostic edit counts.')
        ledger[arm] = clip_ledger
    write_json(destination/'measurements.json',records)
    write_json(destination/'prediction_receipts.json',ledger)
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row['image_sha256']].append(row['dataset'])
    write_json(destination/'input_fingerprints.json',dict(images=[{k:r[k] for k in (
        'dataset','embryo','image_shape','image_sha256','scale','contrast_q99_q10')} for r in rows],
        exact_duplicate_content_groups=[v for v in duplicates.values() if len(v)>1],
        grouping='Embryo prefix is the known biological dependency group. No global crop origin or pairwise crop-overlap map is present in the inspected Zarr metadata; other overlaps remain unknown.'))
    telemetry = telemetry_summary(args.out)
    write_json(destination/'gpu_telemetry_summary.json',telemetry)
    package_seconds = sum(read_json(p)['seconds'] for p in (args.out/'package_validation').glob('*/package_test_receipt.json'))
    scratch = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file() and not p.is_symlink())
    preflight = read_json(args.out/'preflight.json')
    cost = dict(conservative_budget_charged_hours=used_seconds(args.out)/3600,
        full_matrix_worker_hours=sum(r['totals']['worker_seconds'] for r in records.values())/3600,
        successful_packaged_notebook_wall_hours=package_seconds/3600,new_scratch_bytes=scratch,
        elapsed_since_preflight_hours=(datetime.fromisoformat(now())-datetime.fromisoformat(preflight['created'])).total_seconds()/3600,
        accounting='Charged hours include completed study worker intervals, recorded failures, unscored smoke workers and actual package test wall time. Worker intervals include CPU graph work but exclude interpreter startup and initial hash checks. Agent/preflight/reporting time is not CUDA kernel time.',
        envelope=preflight['limits'])
    write_json(destination/'cost_summary.json',cost)
    return records,cost


def append_report(args, destination, measurements, cost):
    """Explain measured behavior without turning exploratory subgroups into gates."""
    paragraphs = ['\n## Measured graph changes and cost\n',
        'Exact edit counts below compare spatiotemporal node/edge multisets against each arm’s parent. Changed coordinates count as remove+add; IDs from different detector runs are not assumed to identify the same cell. Matched GT edge survival uses fresh official per-arm rematching.\n']
    for arm,record in measurements.items():
        t=record['totals']
        score_path=args.out/'scores/full'/arm/'summary.json'
        s=read_json(score_path) if score_path.exists() else {}
        paragraphs.append(f"- {arm}: {len(record['completed'])}/199 clips; {t['final_nodes']:,} nodes, {t['final_edges']:,} edges, {t['final_forks']:,} forks. "
            f"Exact node edits +{t.get('nodes_added_exact',0):,}/−{t.get('nodes_removed_exact',0):,}; edge edits +{t.get('edges_added_exact',0):,}/−{t.get('edges_removed_exact',0):,}. "
            f"{t['fresh_tracking_model_clips']} fresh tracking-model clips, {t['replayed_tracking_model_clips']} fingerprinted neural replays; "
            f"{t['worker_seconds']/3600:.3f} summed worker hours, peak per-process allocated GPU {t['gpu_allocated_peak_bytes']/2**30:.3f} GiB and RSS {t['rss_peak_bytes']/2**30:.3f} GiB. "
            +(f"Matched parent TP edges retained/lost/new: {s['original_tp_survived']:,}/{s['original_tp_lost']:,}/{s['recovered_tp']:,}." if 'original_tp_survived' in s else '')+'\n')
        if arm=='E01':
            m=record['mechanism_counts']
            paragraphs.append(f"  E01 proposed {m.get('E01_components',0):,} connected swap components: {m.get('E01_eligible',0):,} eligible and {m.get('E01_accepted',0):,} accepted. Every other proposal retained the pre-relink edge set.\n")
        if record['subvoxel_displacement']:
            d=record['subvoxel_displacement']
            paragraphs.append(f"  E02 refined {d['count']:,} surviving originals; displacement mean {d['mean_um']:.4f} µm, median {d['p50_um']:.4f} µm, 95th percentile {d['p95_um']:.4f} µm, maximum {d['max_um']:.4f} µm before smoothing.\n")
        if 'E07_anchors' in record['mechanism_counts']:
            m=record['mechanism_counts']
            paragraphs.append(f"  E07 protected {m['E07_anchors']:,} anchors around {m['E07_forks']:,} predicted forks.\n")
    paragraphs.append(f"\nConservatively charged device time: {cost['conservative_budget_charged_hours']:.3f} hours, including actual packaged-notebook test wall time {cost['successful_packaged_notebook_wall_hours']:.3f} hours; full-matrix worker time {cost['full_matrix_worker_hours']:.3f} hours; current new scratch: {cost['new_scratch_bytes']/2**30:.3f} GiB. "
        'Worker wall time includes CPU work and is charged as device time. See cost_summary.json and gpu_telemetry_summary.json for coverage and limitations; sampled utilization is not exact CUDA-event active time.\n')
    paragraphs.append('Full B0 native probability matrices and every pilot matrix are retained. Other completed neural arms retain exact candidate values, offsets, node probabilities, source/target universes, full-matrix hashes and summaries, while dropping redundant dense matrices after inference. No retained value is quantized. Fresh-finalist DeepCenter maps are clip-local with frame hashes retained. This recorded retention policy keeps the multi-arm study inside its scratch allocation without changing predictions.\n')
    paragraphs.append('\n## Source-proven no-ops\n')
    for arm in ('E03','E06'):
        p=args.out/'decisions'/(arm+'.json')
        if p.exists():
            proof=read_json(p).get('proof',{})
            if arm=='E03':
                paragraphs.append(f"- E03: {proof.get('sampled_features',0):,} baseline feature samples inspected across {proof.get('actual_baseline_clips_inspected',0)} clips; "
                    f"{proof.get('fractional_coordinates',0)} fractional coordinates and maximum feature difference {proof.get('max_abs_feature_difference')}. At integer grid coordinates trilinear weights select exactly the original gathered feature.\n")
            else:
                paragraphs.append(f"- E06: actual context length {proof.get('window_size')}; {proof.get('transitions',0):,} adjacent transitions, exactly one valid native context each and zero additional contexts. "
                    f"Checkpoint window length was verified in {proof.get('actual_baseline_clips_inspected',0)} baseline neural receipts.\n")
    paragraphs += ['\n## Historical control reproduction and serialization\n']
    for arm,expected in [('B0',.911774),('B1',.934206)]:
        p=args.out/'scores/full'/arm/'summary.json'
        if not p.exists():
            continue
        s=read_json(p);t=measurements[arm]['totals']
        paragraphs.append(f"- {arm}: public-style original export {s['original_export']['score']:.12f}; shared bounds-sanitized export {s['pooled']['score']:.12f}; "
            f"historical rounded score {expected:.6f}, original-export delta {s['original_export']['score']-expected:+.12f}. "
            f"The actual public writer lower-clamps {t['original_export_lower_clamped_nodes']:,} rounded negative nodes; {t['original_export_outside_nodes']:,} nodes exceed upper image bounds before common sanitation.\n")
    paragraphs.append('The first evaluator implementation mistakenly treated pre-clamp rounded coordinates as the original CSV. The source writer already applies `max(0, round(value))`. That diagnostic was corrected before full-cohort scoring, earlier score receipts were retained, and all affected pilots were freshly rematched. The deployed graphs, frozen inference code and scientific recipes did not change. Two earlier unscored setup failures (annotation guard and missing GAP2 declarations) are also retained.\n')
    strata=args.out/'diagnostics/strata.json'
    if strata.exists():
        data=read_json(strata)
        write_json(destination/'strata.json',data)
        paragraphs+=['\n## Descriptive groups\n',
            'Contrast and raw-image signal-occupancy quartiles were fixed without labels or scores. Temporal/depth bins describe evaluator GT edge recovery only. These correlated groups are not extra validation folds, and no group-specific model routing is used. Exact duplicate-image groups and the absence of a global crop-overlap map are recorded in input_fingerprints.json.\n']
        for arm,value in data['arms'].items():
            if arm=='B0':
                continue
            parent='B1' if arm.startswith('X') else 'B0'
            if parent not in data['arms']:
                continue
            baseline=data['arms'][parent]['clip_groups']
            diffs=[(v['score']-baseline[k]['score'],k) for k,v in value['clip_groups'].items() if k in baseline]
            if diffs:
                delta,group=min(diffs)
                paragraphs.append(f'- {arm}: worst predefined clip group versus {parent}: `{group}`, score delta {delta:+.8f}.\n')
        paragraphs.append('Detailed first/last-two-frame and depth-quartile GT edge recovery counts, group scores and raw denominators are in strata.json; worst matched-TP-survival clips are in measurements.json. These are descriptions of failure locations, not causal proof.\n')
    paragraphs+=['\n## Source, runtime and portability evidence\n',
        'The actual checkpoint context is two frames; detector grid stride is (1,4,4) with zero-origin strided sampling. Effective detector pooling is the public CLI default 3.0 µm; checkpoint metadata contains 5.0 µm but that value is not passed to PredictConfig. The primary association features already use eight inverse-aligned D4 views; the secondary stream uses identity features. Public harmonic fusion combines forward/reverse association evidence; the secondary model uses the original calibrated low-margin logit mix. All numeric thresholds, fusion weights, models and repair settings remain fixed.\n',
        'The pinned metric revision is [`075fc5f5a52d11077f9dc2b074644618f26939e2`](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot), verified against the official repository. Fresh throttled competition pages are retained in the ignored current_reference directory. The archive metadata specifies NvidiaTeslaT4; local tests use one RTX4090 and the pinned existing Python environment. Hidden Kaggle population/runtime feasibility is not established by local timing.\n',
        'The [public notebook page](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) declares Apache-2.0; all three public dataset metadata records declare CC0-1.0, as recorded in [license_receipt.json](license_receipt.json). Standard license text, original attribution and the precise license-verification scope accompany the packages. No source/artifact relicense is claimed.\n']
    manifest=args.out/'packages/manifest.json'
    if manifest.exists():
        paragraphs.append('\n## Actual notebook and CSV validation\n')
        for package in read_json(manifest):
            check=package.get('actual_csv_validation')
            full=package.get('full_cohort_fresh_image_to_csv')
            paragraphs.append(f"- {package['arm']}: actual notebook return code {package.get('notebook_returncode')}; "
                f"{len(package.get('tested_pilots',[]))} full pilots; "
                +(f"{check['rows']:,} CSV rows exactly match the validated node/edge graphs. " if check else 'CSV validation receipt is pending. ')
                +(f"Fresh full-cohort workers plus the actual packaged exporter produced {full['rows']:,} verified CSV rows for {len(full['datasets'])} clips. " if full else '')
                +f"Ready for manual test: {package.get('ready_for_manual_test')}; notebook SHA256 `{package['notebook_sha256']}`.\n")
        paragraphs.append('A complete notebook invocation is tested on the four full pilots. A novel packaging finalist also receives a fresh all-model pass over 199 clips followed by its actual packaged CSV exporter. That composed full-cohort execution is explicitly distinguished from invoking the whole notebook over all 199 clips. See packaging_receipts.json for exact scope and artifact paths.\n')
    p=destination/'REPORT.md'
    p.write_text(p.read_text()+'\n'.join(paragraphs))
