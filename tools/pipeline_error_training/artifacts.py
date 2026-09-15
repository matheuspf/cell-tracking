"""Resolve historical weights through their pinned receipts, without copying them."""
from .common import RESULTS, ROOT, read_json, sha, write_json

P0_MODELS = ROOT / 'segmentation-tracking-v6-local/prior-v6/study/models'


def run():
    artifacts = {}
    p0 = read_json(P0_MODELS.parent / 'model_lock.json')
    for source, expected in p0['models'].items():
        path = P0_MODELS / f'P0_{source}.json'
        if sha(path) != expected:
            raise ValueError('Historical P0 model hash mismatch')
        artifacts[f'P0_{source}'] = dict(path=str(path), sha256=expected)
    package = ROOT / 'strong-tracker-v3/inference_package'
    manifest = read_json(package / 'manifest.json')
    for key, path in manifest['external_checkpoint_paths'].items():
        expected = manifest[f'{key}_weights_sha256']
        if sha(path) != expected:
            raise ValueError(f'Inherited {key} checkpoint hash mismatch')
        artifacts[key] = dict(path=path, sha256=expected)
    for source, path in manifest['external_teacher_paths'].items():
        expected = manifest['legacy_teacher_model_hashes'][source]
        if sha(path) != expected:
            raise ValueError('Inherited teacher hash mismatch')
        artifacts[f'E_hgb_{source}'] = dict(path=path, sha256=expected)
    v4 = ROOT / 'multidata-training-v4'
    manifest = read_json(v4 / 'checkpoint_manifest.json')
    for source in ['44b6', '6bba']:
        for component in ['G', 'I']:
            key = f'{component}_C4_{source}'
            path = v4 / 'models' / f'{key}.pt'
            expected = manifest[key]['sha256']
            if sha(path) != expected:
                raise ValueError(f'Historical C4 model hash mismatch: {key}')
            artifacts[key] = dict(path=str(path), sha256=expected, exposure=manifest[key]['inherited_exposure'])
    write_json(RESULTS / 'model_input_manifest.json', artifacts, immutable=True)
    return artifacts
