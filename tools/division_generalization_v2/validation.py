"""Result evidence aggregation, with missing gates explicitly incomplete."""
import gzip
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .common import RESULTS,WORK,inputs,read_json,write_json,sha,verified_graph,verified_evidence


def run(args=None):
    from .model import ActionModel
    from .infer import predict
    from pipeline_error_training.serialization import export_csv,export_geff
    started=time.monotonic()
    testlog=WORK/'validation/pytest.log';testlog.parent.mkdir(parents=True,exist_ok=True)
    with testlog.open('w') as f:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests/division_generalization_v2',
            'tests/pipeline_error_training/test_contracts.py',
            'tests/pipeline_error_training/test_scientific_contracts.py',
            'tests/pipeline_error_training/test_guards_and_observations.py'],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Correctness tests failed; see '+str(testlog))
    fixtures=[]
    for source in ('44b6','6bba'):
        row=min(inputs(source),key=lambda r:r['baselines']['P0']['nodes'])
        graph,native=verified_graph(row),verified_evidence(row)
        model=ActionModel(image=False)
        # This performs real candidate generation, model scoring, complete
        # alternatives, bounded solving and export; no return-P0 shortcut.
        out,trace=predict(row,graph,native,model,dict(image=False,source=source),literal_zero=True)
        for app,g in out.items():
            np.testing.assert_array_equal(g['nodes'],graph['nodes'])
            np.testing.assert_array_equal(g['edges'],graph['edges'])
        path=WORK/'validation'/source
        path.mkdir(parents=True,exist_ok=True)
        csv=export_csv(path/'zero.csv',row['dataset'],out['protected']['nodes'],out['protected']['edges'])
        geff_path=path/'zero.geff'
        geff=export_geff(geff_path,out['protected']['nodes'],out['protected']['edges']) if not geff_path.exists() else dict(previously_exported=True)
        disabled,_=predict(row,graph,native,model,dict(image=False,source=source),disable=True)
        np.testing.assert_array_equal(disabled['protected']['edges'],graph['edges'])
        fixtures.append(dict(dataset=row['dataset'],trace=trace['counts'],literal_zero_exact=True,
            disable_exact=True,csv=csv,geff=geff,full_clip=True))
    parities=[]
    for source in ('44b6','6bba'):
        path=WORK/'source'/source/'manifest.json'
        if path.exists():
            for row in read_json(path)['clips']:
                parities.extend(dict(source=source,dataset=row['dataset'],**p) for p in row['counterfactual_parities'])
    all_prepared=all((WORK/'source'/s/'manifest.json').exists() for s in ('44b6','6bba'))
    source_counts={s:sum(p['source']==s for p in parities) for s in ('44b6','6bba')}
    parity_ready=all_prepared and all(n>=32 for n in source_counts.values())
    write_json(RESULTS/'action_label_parity.json',dict(status='measured' if parity_ready else 'incomplete',
        full_graph_replays=parities,pinned_metric_unchanged=True,per_source_replays=source_counts,
        sampling='Up to two seeded edits per source clip with supported sampled actions; no fabricated cases in clips without support',
        all_source_preparation_complete=all_prepared))
    result=dict(status='measured',unit_tests_passed=True,pytest_log_sha256=sha(testlog),
        unit_tests_count=int(re.search(r'(?m)^(\d+) passed',testlog.read_text()).group(1)),
        actual_zero_scorer_fixtures=fixtures,literal_zero_path_test=True,
        complete_counterfactual_replays=len(parities),seconds=time.monotonic()-started,
        fresh_image_proof='pending',target_comparisons='pending')
    for name,field in [('corrected_image_validation.json','corrected_image_validation_sha256'),
                       ('matrix_parity.json','matrix_parity_sha256'),
                       ('fast_matrix_parity.json','optimized_matrix_parity_sha256'),
                       ('vectorized_feature_parity.json','vectorized_feature_parity_sha256')]:
        if (RESULTS/name).exists():result[field]=sha(RESULTS/name)
    write_json(RESULTS/'validation.json',result)
    return result
