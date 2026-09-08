"""Freeze reproduced predictions before any outer outcome is opened."""
from .common import OUT,REPO,now,read_json,sha,stage,write_json


def run(args):
    original=read_json(OUT/'predictions/inference_receipt.json')
    reproduced=read_json(OUT/'predictions_gt_unavailable/inference_receipt.json')
    if original['prediction_hashes']!=reproduced['prediction_hashes']:
        raise ValueError('GT-unavailable predictions differ')
    expected=read_json(OUT/'fold_manifest.json')['expected_samples']
    if len(original['prediction_hashes'])!=len(expected):raise ValueError('Incomplete inference')
    if any(x['annotation_reads']!=0 or x['limit'] for x in [original,reproduced]):
        raise ValueError('Invalid GT-free verification receipt')
    result=dict(created=now(),prediction_hashes=original['prediction_hashes'],
                gt_unavailable_reproduction_passed=True,expected_sample_count=len(expected),
                model_lock_sha256=sha(OUT/'model_lock.json'),fold_manifest_sha256=sha(OUT/'fold_manifest.json'),
                inference_receipt_sha256=sha(OUT/'predictions/inference_receipt.json'),
                reproduction_receipt_sha256=sha(OUT/'predictions_gt_unavailable/inference_receipt.json'),
                code_hashes={str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'tools/annotation_selection').glob('*.py'))},
                policies='Fixed preregistration; no target outcome selection',outer_outcomes_revealed=False)
    write_json(OUT/'prediction_lock.json',result,immutable=True)
    stage('S070','complete',policies='all preregistered budgets',identity_test=True)
    stage('S080','complete_fixed_setting_fallback',gt_unavailable_reproduction=True,samples=len(expected),source_inner_tuning_available=False)
    print(f'Both directions locked; {len(expected)} GT-unavailable prediction hashes identical',flush=True)
