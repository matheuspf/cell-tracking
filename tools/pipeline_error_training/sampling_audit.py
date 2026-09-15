"""Describe the executed deterministic calibration expansion without refitting."""
from pathlib import Path


def run():
    import numpy as np
    from .common import RESULTS,WORK,read_json,sha,write_json
    rows = []
    for source in ['44b6','6bba']:
        package = WORK/'training/D20_temporal'/source/'20260915'
        fit = read_json(package/'package.json')
        calibrated = read_json(package/'calibration.json')
        counts = dict(supported_rows=0,positive_rows=0,weighted_supported_rows=0.,weighted_positive_rows=0.)
        caches = []
        weights_seen = set()
        for name in calibrated['clips']:
            path = WORK/'calibration/D20_temporal'/source/'20260915'/f'{name}.npz'
            receipt = read_json(path.with_suffix('.json'))
            if sha(path)!=receipt['sha256'] or receipt['model_sha256']!=fit['weights_sha256']:
                raise ValueError('Executed source calibration cache changed')
            with np.load(path,allow_pickle=False) as data:
                weights = data['inverse_anchor_inclusion'][data['known']]
                good = data['good'][data['known']]
                counts['supported_rows'] += len(weights)
                counts['positive_rows'] += int(good.sum())
                counts['weighted_supported_rows'] += float(weights.sum())
                counts['weighted_positive_rows'] += float(weights[good].sum())
                weights_seen.update(map(float,np.unique(weights)))
            caches.append(dict(dataset=name,sha256=receipt['sha256']))
        fraction = counts['weighted_positive_rows']/counts['weighted_supported_rows']
        np.testing.assert_allclose(fraction,calibrated['calibration']['effective_unbalanced_positive_fraction'],rtol=0,atol=1e-15)
        if weights_seen!={1.,9.}:
            raise ValueError('Recorded deterministic calibration expansion changed')
        rows.append(dict(source=source,**counts,anchor_expansion_weights=sorted(weights_seen),
            raw_supported_positive_fraction=counts['positive_rows']/counts['supported_rows'],
            expanded_supported_positive_fraction=fraction,
            positive_event_groups_in_training_pool=fit['source_event_groups'],
            checkpoint_sha256=fit['weights_sha256'],calibration_sha256=sha(package/'calibration.json'),
            cache_records=caches))
    result = dict(status='measured',directions=rows,
        ordinary_anchor_rule='frame % 9 == source GT component index % 9; all supported event-compatible anchors retained',
        field_interpretation='The existing sampling_probability=1/9 field denotes a deterministic temporal sampling fraction; inverse_anchor_inclusion=9 is a fixed expansion weight, not a proven randomized row propensity.',
        calibration_scope='Expanded supported source candidate sample; not an exact census of every source-field hypothesis',
        limitations=['Deterministic temporal phase can correlate with lineage indexing or changing candidate difficulty.',
            'Unknown labels remain censored; the annotated supported population need not represent unannotated cells.',
            'Inner source independence is not certified; calibration remains exploratory.'],
        group_preservation='Event-compatible anchors and all competing alternatives remain grouped; this audit does not alter the prepared bank.',
        recipes_changed=False,models_refitted=False,new_target_metrics_read=False,
        inspected_source_sha256={n:sha(Path(__file__).with_name(n+'.py')) for n in ['prepare','dataset','calibration']})
    write_json(RESULTS/'calibration_sampling_audit.json',result,immutable=True)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    print(run(),flush=True)
