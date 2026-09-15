"""Export compact, reproducible evidence after the required panels are complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .evaluate import ROOT, REPO
from annotation_selection.common import METRIC_REV

OUT = REPO / 'results/detector-screen-20260914'
NAMES = {
    'incumbent': 'Temporal U-Net ensemble (incumbent)',
    'cellect': 'CELLECT',
    'organoid': 'OrganoidTracker 2.0',
    'nucverse': 'NucVerse3D generalized',
    'spotiflow': 'Spotiflow synth_3d',
    'spotiflow_loose': 'Spotiflow synth_3d, threshold 0.05',
    'xenopus': 'Xenopus StarDist3D origins',
    'xenopus_centroid': 'Xenopus StarDist3D mask centroids',
    'xenopus_loose': 'Xenopus StarDist3D origins, threshold 0.1',
    'pacmap': 'PAC-MAP spheroids',
    'anystar': 'AnyStar-mix origins',
    'anystar_centroid': 'AnyStar-mix mask centroids',
    'cellpose_cpdino_vitb': 'Cellpose cpdino-vitb',
    'focus_reference': 'FOCUS-3D (previous pilot reference)',
}
REPOSITORIES = ['CELLECT','OrganoidTracker','NucVerse3D','KapoorLabs-VollSeg',
                'stardist','spotiflow','PAC-MAP','AnyStar','cellpose','dinov3']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def compute_seconds(receipt):
    if 'timing_seconds' in receipt:
        return receipt['timing_seconds'].get('total')
    return receipt.get('seconds', receipt.get('inference_seconds'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--required', nargs='+', default=[
        'incumbent','cellect','organoid','nucverse','spotiflow','xenopus','xenopus_centroid'])
    args = parser.parse_args()
    panel = json.loads((ROOT/'panel.json').read_text())
    expected = {role: {r['key'] for r in panel['frames'] if r['role']==role}
                for role in ('pilot','assessment')}
    data = {}
    for path in (ROOT/'evaluation').glob('*.json'):
        obj = json.loads(path.read_text())
        if 'method' in obj and 'per_frame' in obj:
            data[obj['method']] = obj
    for method in args.required:
        assert method in data, method
        actual = {r['key'] for r in data[method]['per_frame'] if r['role']=='assessment'}
        assert actual == expected['assessment'], (method,len(actual),len(expected['assessment']))

    models=[]
    for method, obj in sorted(data.items()):
        # Incomplete panels cannot enter a comparison table.
        complete = [role for role in expected if
                    {r['key'] for r in obj['per_frame'] if r['role']==role} == expected[role]]
        summary = [r for r in obj['summary'] if r['role'] in complete]
        if not summary:
            continue
        rows=[]
        for r in obj['per_frame']:
            if r['role'] not in complete:
                continue
            item={k:r[k] for k in ['key','dataset','embryo','time','role','gt','predicted',
                'matches','duplicates_after_rounding','clipped_centers',
                'baseline_candidates','gt_multiple_candidates_at_7','nearest_in_5_to_7_um']}
            item['compute_seconds']=compute_seconds(r.get('runtime_receipt',{}))
            item['budgets']={b:{k:v for k,v in vals.items() if k!='matched_gt_at_7'}
                             for b,vals in r['budgets'].items()}
            rows.append(item)
        runtime=[]
        for role in complete:
            times=[r['compute_seconds'] for r in rows if r['role']==role and r['compute_seconds'] is not None]
            if times:
                runtime.append(dict(role=role,frames=len(times),mean_seconds=float(np.mean(times)),
                                    median_seconds=float(np.median(times)),max_seconds=float(np.max(times))))
        model=dict(id=method,name=NAMES.get(method,method),complete_panels=complete,
                   summary=summary,runtime=runtime,confidence_ranking_available=method!='focus_reference',
                   progress={role:sum(r['role']==role for r in obj['per_frame']) for role in expected})
        models.append(model)
        write(OUT/'per-frame'/(method+'.json'),rows)

    result=dict(date='2026-09-14',panel_definition_sha256=panel['definition_sha256'],
        selection=panel['selection'],frames={k:len(v) for k,v in expected.items()},
        primary_metric=panel['primary_metric'],models=models,
        metric='Installed tracksdata DistanceMatching optimal=True, scale=(1.625,0.40625,0.40625), radius7µm, after native integer rounding and clipping.',
        tracksdata_version=version('tracksdata'),official_metric_revision=METRIC_REV,
        scope='Pretrained inference and detector evaluation only; no training, temporal association, submission, or competition score.',
        budget_definition='Per-frame upper caps of 0.5/0.75/1/1.25/1.5 times incumbent candidate count, retaining each method\'s highest confidence candidates.',
        limitations=[panel['limitations'],
            'Sparse annotations do not identify all real cells; unmatched candidates are not automatically false positives.',
            'Both-endpoint coverage is a detector diagnostic under official per-frame matches, not edge Jaccard or division score.',
            'Candidate counts are before any temporal pruning and do not directly determine the final node-count penalty.',
            'Adapter-recorded compute excludes GPU queue waits; I/O coverage differs and the incumbent was reused rather than retimed.',
            'Cellpose CPU affinity changed during assessment; timings are descriptive measurements, not a controlled speed comparison.',
            'FOCUS reference confidence values are placeholders; its confidence-budget curves are not interpretable.'])
    failure_path=ROOT/'evaluation/failure-analysis.json'
    if failure_path.exists():
        result['incumbent_failure_analysis']=json.loads(failure_path.read_text())
    write(OUT/'summary.json',result)
    write(OUT/'panel.json',dict(definition_sha256=panel['definition_sha256'],selection=panel['selection'],
        frames=[{k:r[k] for k in ('key','dataset','embryo','time','role','shape','spacing_um')} for r in panel['frames']]))

    repos=[]
    for name in REPOSITORIES:
        path=Path('/home/mpf/code/kaggle')/name
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        url=subprocess.check_output(['git','-C',str(path),'remote','get-url','origin'],text=True).strip()
        repos.append(dict(name=name,path=str(path),commit=commit,source=url))
    receipts={}
    for relative in [
        'incumbent/provenance.json',
        'cellect/provenance.json','cellect/source_parity.json','cellect/source_parity_merged.json',
        'organoid/models/verification.json','organoid/geometry-verification.json','organoid/source-equivalence.json',
        'organoid/config-pilot.json','organoid/config-assessment.json','organoid/metric-audit-scan.json',
        'organoid/submission-coordinate-contract.json','organoid/panel-summary.json',
        'organoid/execution-provenance.json',
        'nucverse/checkpoint.json','nucverse/verification/model_load.json',
        'nucverse/verification/session-equivalence-44b6_81c256f0-t025.json',
        'nucverse/assessment-summary.json','nucverse/coverage-summary.json',
        'assets/xenopus/receipt.json','xenopus/config.json','xenopus/execution-decision.json',
        'xenopus/default-nms-tile-verification.json','assets/spotiflow/receipt.json','spotiflow/config.json',
        'evaluation/geometry-verification.json','evaluation/output-validation.json',
        'evaluation/runtimes.json','cellect/validation_summary.json',
        'cellpose/verification/profile/parity.json','cellpose/verification/compile/parity.json',
        'cellpose/cpu-limit-receipt.json','organoid/complementarity-budget-summary.json']:
        path=ROOT/relative
        if path.exists():receipts[relative]=json.loads(path.read_text())
    # Supplemental adapters supply their own compact provenance/config receipts.
    for folder in ['pacmap','anystar','cellpose']:
        for name in ['provenance.json','assets.json','config.json','config-pilot.json','checkpoint.json',
                     'saved-checkpoint-metadata.json','preset.json','geometry-verification.json',
                     'source_parity.json','coordinate_validation.json','environment.json',
                     'batch-equivalence.json','performance-receipt.json','execution-provenance.json',
                     'pilot-summary.json','pilot_summary.json','panel-summary.json','validation_summary.json',
                     'assessment-summary.json','coverage-summary.json','completion-receipt.json',
                     'adapter_export_notes.json']:
            path=ROOT/folder/name
            if path.exists():receipts[str(path.relative_to(ROOT))]=json.loads(path.read_text())
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO/'tools/detector_screen').glob('*.py'))}
    write(OUT/'provenance.json',dict(repositories=repos,receipts=receipts,adapter_sha256=hashes))
    print(json.dumps({'models':len(models),'complete_assessment':[m['id'] for m in models if 'assessment' in m['complete_panels']],
                      'output':str(OUT)},indent=2))


if __name__=='__main__':
    main()
