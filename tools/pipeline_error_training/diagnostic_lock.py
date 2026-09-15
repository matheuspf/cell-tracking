"""Image-only strata and one fixed module-stress recipe, before target scoring."""
import numpy as np

from .common import RESULTS, inputs, read_json, sha, verified_evidence, write_json


def run():
    split = read_json(RESULTS / 'split_manifest.json')['directions']
    thresholds = {}
    for source in ['44b6', '6bba']:
        sample = []
        for row in inputs():
            if row['embryo'] == source and split[source][row['dataset']]['partition'] == 'fit':
                x = verified_evidence(row)['node_features']
                sample.append(x[::16][:, [6, 7, 8, 9]])
        values = np.concatenate(sample)
        thresholds[source] = dict(intensity=np.quantile(values[:, 0], [.25, .75]).tolist(),
            local_contrast=np.quantile(values[:, 1], [.25, .75]).tolist(),
            density10=np.quantile(values[:, 2], [1/3, 2/3]).tolist(),
            normalized_depth=[1/3, 2/3], spatial_boundary_um=7.,
            temporal_boundary_frames=[2, 4], close_center_distance_um=7., source_image_only_rows=len(values))
    screen = read_json(RESULTS / 'source_screen_lock.json')
    result = dict(status='source_only_frozen', thresholds=thresholds,
        stress=dict(brightness=.9, contrast=.9, gamma=1.1, read_noise_std=.01,
            blur_kernel=[.25, .5, .25], sampling_scale=1.02, center_shift_um_zyx=[.4, .2, .2],
            missing_relative_frame=1, seed=20260915,
            scope='One combined fixed module perturbation at P0 observations; upstream P0 outputs and native evidence remain the same. This is not a new independent embryo or a detector stress test.'),
        representative_clips=screen['selected'],
        event_support='Post-freeze official local-window observation/topology categories; never a model input.',
        targets_used_to_fit_thresholds=False, no_repeated_stress_selection=True,
        split_manifest_sha256=sha(RESULTS / 'split_manifest.json'))
    write_json(RESULTS / 'diagnostic_lock.json', result, immutable=True)
    write_json(RESULTS / 'protocol_addenda.json', dict(before_target_comparisons=True,
        context_dropout='All representations drop relative frame -1 or +1. This preserves the existing Organoid and compact behavior and aligns the seven-frame representation before its primary training starts.',
        observation_provenance='Original raw graph hashes additionally verified against the already pinned v3 fresh-evidence receipts. Pre-verification preparation is preserved under the new invalid-attempt root; no O model used it.'), immutable=True)
