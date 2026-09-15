"""Final correctness evidence, separate from model training and selection."""
from pathlib import Path
import subprocess
import sys
import time

from .common import RESULTS, WORK, inputs, read_json, sha, write_json
from .report import optional


def tests():
    root=WORK/'validation';root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    command=[sys.executable,'-m','pytest','tests/pipeline_error_training','-q']
    with (root/'tests.log').open('w') as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(status='measured' if process.returncode==0 else 'failed',returncode=process.returncode,
        seconds=time.monotonic()-started,command=command,log_sha256=sha(root/'tests.log'),
        summary=(root/'tests.log').read_text().strip().splitlines()[-1],
        tested_source_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        test_files_sha256={p.name:sha(p) for p in sorted(Path('tests/pipeline_error_training').glob('test_*.py'))})
    write_json(RESULTS/'test_validation.json',receipt)
    return receipt


def run(run_tests=False):
    tested=tests() if run_tests else optional(RESULTS/'test_validation.json')
    baseline=read_json(RESULTS/'baseline_validation.json')
    zero=read_json(RESULTS/'D00.json')
    frozen=optional(RESULTS/'target_freeze.json') or {}
    actual_baseline_hashes=True
    for row in inputs():
        for arm in ['P0','C4_m6','C0']:
            spec=row['baselines'][arm]
            if sha(spec['path'])!=spec['sha256']:
                actual_baseline_hashes=False
                raise ValueError('Historical baseline bytes changed')
    arms={}
    for arm in frozen.get('experiments',[]):
        if arm=='D00':continue
        complete=optional(WORK/'full_evaluation'/arm/'summary.json')
        if not complete:
            arms[arm]=dict(status='not run',reason='Complete serialized and scored all-199 arm unavailable')
            continue
        manifests=[]
        for row in inputs():
            path=WORK/'predictions'/arm/f'{row["dataset"]}.npz'
            receipt=read_json(path.with_suffix('.json'))
            guard=read_json(path.with_suffix('.guard.json'))
            if sha(path)!=receipt['prediction_sha256'] or not path.with_suffix('.csv').exists():
                raise ValueError('Scored complete graph or its verified CSV is unavailable')
            if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
                raise RuntimeError('Candidate access guard failed')
            source='6bba' if row['embryo']=='44b6' else '44b6'
            if receipt['source']!=source:
                raise ValueError('Candidate model source was not explicit and opposite-directional')
            manifests.append(dict(dataset=row['dataset'],prediction_sha256=sha(path),csv_sha256=sha(path.with_suffix('.csv')),
                guard_sha256=sha(path.with_suffix('.guard.json'))))
        write_json(WORK/'validation'/f'{arm}-serialized_manifest.json',dict(records=manifests),immutable=True)
        arms[arm]=dict(status='measured',complete_clips=len(manifests),guard_before_numerical=True,
            csv_roundtrip_verified_in_prediction_process=True,metric_aggregation_verified_in_separate_scorer=True,
            manifest_sha256=sha(WORK/'validation'/f'{arm}-serialized_manifest.json'))
    fresh=optional(RESULTS/'fresh_image_validation.json') or {}
    native=optional(RESULTS/'native_refresh_validation.json') or {}
    required=['real_event_fixture.json','inference_optimization.json','staged_inference_parity.json',
              'staged_crowded_parity.json','multi_model_parity.json','replacement_bound_parity.json',
              'resume_validation.json','C4_trace_validation.json']
    contracts={name:(optional(RESULTS/name) or {}).get('status','not run') for name in required}
    hard=bool(frozen and tested and tested['status']=='measured' and actual_baseline_hashes and baseline['status']=='measured'
        and zero['status']=='measured' and fresh.get('status')=='measured' and all(v=='measured' for v in contracts.values()))
    if any(a.startswith('O') and r['status']=='measured' for a,r in arms.items()):
        hard=hard and native.get('status')=='measured'
    result=dict(status='measured' if hard else 'failed',hard_correctness_pass=hard,
        reason=None if hard else 'One or more required executable correctness or fresh-image contracts are incomplete or failed',
        tests=tested,baseline_hashes_preserved=actual_baseline_hashes,zero_head=zero,
        contracts=contracts,arms=arms,fresh_image_status=fresh.get('status','not run'),
        actual_changed_coordinate_status=native.get('status','not run'),target_freeze_exists=bool(frozen),
        new_target_fitting=False,source_only_head_inherited_upstream_exposure=True,
        clean_end_to_end_transfer=dict(status='blocked',reason='Completed clean upstream fits are unavailable'),
        independent_source_groups_certified=False,production_default_changed=False,
        Kaggle_12_hour_runtime_measured=False)
    write_json(RESULTS/'validation.json',result)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    raise SystemExit(0 if run('--tests' in sys.argv)['status']=='measured' else 1)
