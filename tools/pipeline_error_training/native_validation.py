"""Prepare a label-free original-coordinate and changed-coordinate native proof."""
from pathlib import Path
import subprocess
import sys

from .common import RESULTS, ROOT, WORK, inputs, read_json, sha, write_json


def run():
    name = '44b6_d754aa59'
    row = next(r for r in inputs() if r['dataset'] == name)
    safe = {k: row[k] for k in ['dataset', 'image_path', 'image_shape', 'physical_scale', 'metadata_sha256', 'raw', 'evidence']}
    safe['baselines'] = {'P0': row['baselines']['P0']}
    manifest = read_json(ROOT/'image-native-tracking-v5/inference_package_validation/base/manifest.json')
    dependencies = [dict(path=manifest['external_checkpoint_paths'][k], sha256=manifest[k+'_weights_sha256']) for k in ['primary', 'secondary']]
    original = ROOT/'strong-tracker-v2/raw'/f'{name}.npz'
    receipt = Path(row['evidence']['path']).with_suffix('.json')
    dependencies.extend([dict(path=str(original), sha256=read_json(receipt)['inputs']['raw_sha256']),
        dict(path=str(receipt), sha256=row['evidence']['receipt_sha256'])])
    for field in ['raw', 'evidence']:
        dependencies.append(dict(path=row[field]['path'], sha256=row[field]['sha256']))
    dependencies.append(dict(path=row['baselines']['P0']['path'], sha256=row['baselines']['P0']['sha256']))
    root = WORK/'native_validation'
    write_json(root/'job.json', dict(row=safe, root=str(root/'output'), dependencies=dependencies), immutable=True)
    with (root/'worker.log').open('a') as log:
        child = subprocess.run([sys.executable, '-m', 'pipeline_error_training.native_validation_entry', str(root/'job.json')],
                               stdout=log, stderr=subprocess.STDOUT)
    if child.returncode:
        write_json(RESULTS/'native_refresh_validation.json', dict(status='failed', returncode=child.returncode,
            log_sha256=sha(root/'worker.log'), scientific_model_selection_unchanged=True))
        raise RuntimeError('Native coordinate proof failed; original artifact and log preserved')
    result = read_json(root/'output/receipt.json')
    guard = read_json(root/'output/guard.json')
    if guard['blocked_reads'] or guard['blocked_network'] or not guard['installed_before_numerical']:
        write_json(RESULTS/'native_refresh_validation.json',dict(status='failed',stage='access_guard',
            guard=guard,original_coordinate_parity=result.get('unchanged_raw_query_coordinate_parity'),
            scientific_model_selection_unchanged=True,log_sha256=sha(root/'worker.log')))
        raise RuntimeError('Native coordinate proof guard failed')
    write_json(RESULTS/'native_refresh_validation.json', dict(result, guard=guard))


if __name__ == '__main__':
    run()
