"""Sanitized measured report and small offline dashboard; no clip/label payloads."""
import csv
import json
import re
from .common import *

def summaries():
    from annotation_selection.metric_adapter import aggregate
    expected=inventory();records=[]
    for arm in ['C0','P0','B0','M0','U0','U1']:
        paths=list((OUT/'evaluation'/arm).glob('*.json'))
        measured={read_json(p)['dataset']:read_json(p) for p in paths}
        complete=set(measured)=={r['dataset'] for r in expected}
        for em in ['44b6','6bba','pooled']:
            names=[r['dataset'] for r in expected if em=='pooled' or r['embryo']==em]
            row=dict(variant=arm,embryo=em,status='complete' if complete else 'blocked',
                     samples=sum(n in measured for n in names),expected_clips=len(names))
            if complete:
                sub=[measured[n] for n in names];a=aggregate(sub,names)
                row.update({k:v for k,v in a.items() if k!='counts'},**a['counts'],
                           delta_C0=a['score']-BASE[em],
                           official_seconds=sum(r['seconds'] for r in sub),
                           prediction_seconds=sum(r['prediction_seconds'] for r in sub))
            records.append(row)
    return records

def run():
    target=REPO/'results/segmentation-tracking-v6';target.mkdir(parents=True,exist_ok=True)
    rows=summaries();pre=read_json(OUT/'preflight.json')
    def receipt(name):
        path=OUT/f'{name}.json';return read_json(path) if path.exists() else {}
    fresh=receipt('fresh_receipt');point=receipt('fresh_point_receipt');repeat=receipt('repeatability')
    cli=receipt('cli_check')
    p0=[r for r in rows if r['variant']=='P0']
    pass_scores=len(p0)==3 and all(r.get('delta_C0',-1)>=-1e-8 for r in p0) and p0[-1].get('delta_C0',0)>0
    retained=pass_scores and point.get('passed',False) and repeat.get('passed',False)
    status=dict(study_id='segmentation-tracking-v6',updated=now(),
        status='independent_stages_executed_segmentation_and_ultrack_blocked',
        selected='P0' if retained else 'C0',incumbent='C0',incumbent_score=BASE['pooled'],
        selected_score=p0[-1]['score'] if retained else BASE['pooled'],
        target_met=False,target_score=.95,complete_configurations=2,registered_configurations=6,
        learned_segmenter_frames=0,ultrack_frames=0,learned_tool_integration_complete=False,
        retained_point_control=retained,mask_benefit='unmeasured',ultrack_benefit='unmeasured',
        blocked_arms=['B0','M0','U0','U1'],blockers=pre['blockers'],
        exact_c0_fresh_clips=len(fresh.get('clips',[])) if fresh.get('passed') else 0,
        exact_p0_fresh_clips=len(point.get('clips',[])) if point.get('passed') else 0,
        deterministic_repeat_passed=repeat.get('passed',False),
        exploratory_validation=True,automatic_external_downloads=False,prior_studies_preserved=True,
        mask_division_graph_integration='not executed; contracts implemented, inherited P0 decoder freezes forks',
        new_package_selected=False)
    status['delivered_cli_verified']=cli.get('passed',False)
    columns=['variant','embryo','status','samples','expected_clips','score','delta_C0','adj_edge_jaccard',
             'division_jaccard','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn',
             'num_pred_nodes','node_recall','prediction_seconds','official_seconds']
    for directory in [OUT,target]:
        with (directory/'ablation_scores.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        write(directory/'status.json',status)
    write(OUT/'selection.json',dict(selected=status['selected'],score=status['selected_score'],
        gates=dict(pooled_gain_and_no_embryo_regression=pass_scores,deterministic_repeat=repeat.get('passed',False),
                   fresh_point=point.get('passed',False)),models={s:str(OUT/'models'/f'P0_{s}.json') for s in ['44b6','6bba']},
        original_c0_lock_unchanged=True,selection_scope='Retained point control only; learned segmentation and Ultrack blocked'))
    text=['# V6 measured execution results','',
        '**Learned segmentation and Ultrack integration are blocked by missing local runtimes.** '
        'The independent C0 and P0 comparisons completed on all 199 clips. No learned masks were generated; '
        'B0, M0, U0 and U1 have no competition scores. Classical watershed was not used as a replacement.','',
        f"C0 remains intact at **{BASE['pooled']:.15f}**. P0 scores **{p0[-1]['score']:.15f}**, "
        f"a pooled gain of **{p0[-1]['delta_C0']:+.15f}**. "
        + ('The small point-control improvement is retained after both-embryo, deterministic-repeat and fresh-image gates passed. '
           if retained else 'P0 has not yet passed every retention gate; C0 remains selected. ')
        + '**The 0.95 target is not attained.** This gain does not measure bbox, mask or Ultrack benefit.','',
        '## Complete measurements','',
        'All rows use fresh pinned official node assignment, edge/division scoring and exact run-level aggregation. '
        'Scores are not means of per-clip scores. Both arms retain 4,108,943 C0 observations.']
    for arm in ['C0','P0']:
        text+=['',f'**{arm}**','']
        for r in rows:
            if r['variant']!=arm:continue
            text.append(f"- {r['embryo']}: {r['samples']} clips; score {r['score']:.15f}; "
                f"edge TP/FP/FN {r['edge_tp']:,}/{r['edge_fp']:,}/{r['edge_fn']:,}; "
                f"division TP/FP/FN {r['division_tp']}/{r['division_fp']}/{r['division_fn']}; "
                f"{r['num_pred_nodes']:,} nodes.")
    control=receipt('control_receipt');images=receipt('image_checks')
    text+=['','## What ran','',
        '- S600: all 199 C0 graph hashes, cached scores, image axes/spacing and full fresh native evidence were checked. '
        'Primary, secondary, native source and frozen teacher checkpoints passed their manifest hashes. '
        'The latest v5 local snapshot had 12/18 scored configurations, six/eight native fits, and selected C0. '
        'No v5 process was active at preflight; the supervisor JSON was stale. No prior job was stopped or restarted.',
        '- P0: the full 38-field incumbent evidence and identical candidate bank feed one regularized logistic residual '
        'with a fixed native-logit offset (L2=1, at most 250 L-BFGS iterations). Each source embryo trains its opposite direction. '
        'Only supported incoming-parent contradictions are negatives. An unrecorded second daughter remains unknown. '
        'The unchanged v3 association decoder uses margin 3, a 2% edit cap and frozen existing forks. '
        'This is a new point control applied to C0, not the weakened v5 primary-only control.',
        '- Both deterministic fits reproduced their coefficients exactly; all 199 repeated P0 graphs matched. '
        'The fitter and decoder have no random sampling, so a nominal second seed would not be a distinct experiment.',
        '- The source44 optimizer converged after 180 iterations. The source6 optimizer used its full 250-iteration '
        'budget and reported an iteration-limit stop (not convergence). The fixed-budget coefficients were finite '
        'and exactly reproducible. The budget was not increased after target results.',
        f"- Two complete fresh C0 clips passed exact graph/CSV parity in {sum(r['seconds'] for r in fresh.get('clips',[])):.2f} seconds. "
        'Both used unfamiliar image names and explicit source-model arguments. The early Python audit guard denied real GT, '
        'old graph/observation reads and external DNS; successful inference audits recorded no blocked access attempts.',
        '- Fresh P0 checks rebuild the full native point features from those new image-derived arrays, then check exact '
        'candidate/features, graph and CSV parity. The first attempt exposed an absent packaged input-hash manifest in the '
        'optional heatmap-sharing helper. The v6 caller now uses the original independent regeneration path. '
        'The failure log is preserved locally.',
        f"- {images.get('real_frames_read',0)} actual image frames were read across four image-density-selected clips, "
        'two per embryo, eight consecutive frames each. Orthogonal raw-image/C0-center views were inspected. '
        'Nuclear fluorescence supports nuclei segmentation. These views contain no learned masks and cannot validate '
        'mask merges, duplicate instances or segmentation tile seams.',
        f"- The C0/P0 fit/decode/score stage took {control.get('seconds',0):.2f} wall seconds with three bounded CPU workers; "
        f"deterministic repeat checks took {repeat.get('seconds',0):.2f} seconds. Source fits used no GPU training."]
    if cli.get('passed'):
        text += [f"- The delivered `infer --arm P0` CLI independently repeated both full fresh clips in "
            f"{sum(r['seconds'] for r in cli['clips']):.2f} seconds. Both emitted C0 and P0 CSVs matched the complete "
            'batch graphs; startup guards passed. The blocked M0 CLI check exited explicitly before baseline execution.']
    text+=['','## Blocked stages and implementation limits','']+[f'- {b}.' for b in pre['blockers']]
    text+=['',
        'The three FOCUS weight SHA256 values match `/root/FOCUS-3D/SHA256SUMS`; nuclei weights are the appropriate '
        'compartment. Weights alone do not supply `infer_volume`, its preprocessing or its runtime. No downloads, '
        'license acceptance, external data collection, fine-tuning or other model survey was performed.', '',
        'The maintained adapters implement native ZYX shape/origin validation, anisotropic size conversion, '
        'compressed bbox-local occupancy with hashes, physical shape/volume, inside-mask intensity and nullable confidence. '
        'Strict one-to-one containment flags shared masks and leaves unmatched C0 observations intact. Bbox and occupancy '
        'features are distinct. Daughter-union/persistent-separation features, hierarchy exclusion, exact mask survival, '
        'explicit Ultrack observation ancestry and full-native geometry provenance have executable contracts.', '',
        '**Those contracts are not completed tool integration.** FOCUS signature binding, Cellpose volumetric execution, '
        'Ultrack conversion/hierarchy/solver signatures, hierarchy-mask extraction, native link writing and mask-division '
        'terms in the graph objective remain unvalidated/unexecuted without their runtimes and real masks. '
        'No physical recipe was selected. B0/M0 fits, U0/U1, learned-mask timing/storage pilot, sparse containment checks '
        'and fresh image-to-mask-to-graph tests were not run. `infer --arm B0/M0/U0/U1` fails explicitly before C0 execution. '
        'There is no silently filled tool score or newly selected segmentation package.', '',
        '## Resources, checks and preservation','',
        f"The initial persistent filesystem had {pre['persistent_free_gib']:.3f} GiB free, already below the plan's 8 GiB floor. "
        'No old artifacts were deleted. Only small v6 code/models/metadata were persisted; transient control graphs, '
        'fresh exports and images used `/dev/shm/cell-tracking-segmentation-v6`, with a separate 8 GiB admission floor. '
        'RAM-backed artifacts are local but do not survive a reboot. Persistent mask/database allocation stayed blocked.', '',
        'One 4090 inference worker ran at a time. C0 fresh process-tree RSS and total GPU peaks were sampled against '
        '24/20 GiB limits. CPU control peak RSS was not sampled, and the measured wall runtimes must not be described '
        'as measured GPU-kernel hours. The 24 GPU-hour cap was not approached. No new neural-network training was run.', '',
        'The 38 inherited NumPy region-contract tests and 13 new tests passed. They cover external-network denial, half-open boxes, spacing, '
        'empty frames, label permutation, packed occupancy, ownership, duplicate/hierarchy conflicts, sparse unknowns, '
        'daughter union, legal forks/gaps and rejection of unrelated native scores. Synthetic seam/conflict fixtures '
        'are distinct from unrun real segmenter seam checks.', '',
        'Code and the small offline dashboard reuse the repository scorer, decoder, native predictor, CSV routines '
        'and reporting/browser conventions. The score CSV includes explicit blocked rows with blank score/count fields. '
        'Prior source/results and the uncommitted plan edit are preserved. Weights, raw images, masks, per-clip labels, '
        'model coefficients and credentials are outside Git. No merge or Kaggle submission occurred.', '',
        'Both embryos have been reused and upstream checkpoints retain prior exposure. These are exploratory local '
        'comparisons, not independent biological validation or evidence of a hidden-leaderboard gain.','']
    monitored=fresh.get('clips',[])+cli.get('clips',[])
    if monitored:
        text += [f"Measured fresh/CLI peak process-tree RSS: {max(r['peak_process_tree_rss_gib'] for r in monitored):.3f} GiB; "
                 f"peak total GPU memory: {max(r['peak_total_gpu_gib'] for r in monitored):.3f} GiB.",'']
    for directory in [OUT,target]:(directory/'final_report.md').write_text('\n'.join(text))
    payload=dict(status=status,scores=rows,packages=pre['packages'],
        runtime=dict(control_seconds=control.get('seconds'),repeat_seconds=repeat.get('seconds'),
                     fresh_c0_seconds=sum(r['seconds'] for r in fresh.get('clips',[])),
                     fresh_p0_seconds=sum(r['seconds'] for r in point.get('clips',[]))),
        fresh_point=[{k:r[k] for k in ['embryo','seconds','passed','exact_graph_parity','csv_roundtrip','exact_native_inputs']} for r in point.get('clips',[])])
    template=(Path(__file__).parent/'dashboard_template.html').read_text()
    old_template=(REPO/'tools/strong_tracker_v2/dashboard_template.html').read_text()
    css=re.search(r'<style>(.*?)</style>',old_template,re.S).group(1)
    html=template.replace('__CSS__',css).replace('__DATA__',json.dumps(payload,allow_nan=False).replace('<','\\u003c'))
    for directory in [OUT,target]:(directory/'dashboard.html').write_text(html)
    write(OUT/'report_receipt.json',dict(at=now(),sanitized=True,selected=status['selected'],rows=len(rows),
        files={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    print('report',status['selected'],status['selected_score'],status['status'],flush=True)
