"""Concrete portable inference candidate and isolated fresh-image validation."""
import shutil
import subprocess
import sys
import time
from .common import *

def build():
    import torch
    package=OUT/'inference_package';package.mkdir(parents=True,exist_ok=True)
    if not (package/'base').exists():shutil.copytree(V3/'inference_package',package/'base')
    target=package/'tools/multidata_training_v4';target.mkdir(parents=True,exist_ok=True)
    for name in ['__init__.py','common.py','models.py','proposals.py','adapters.py','decode.py','infer.py','package_predict.py']:
        shutil.copyfile(Path(__file__).parent/name,target/name)
    # Prediction imports only pure/guard utilities from the base code, without the metric.
    for module in ['strong_tracker_v3','annotation_selection','strong_tracker_v2']:
        src=package/'base/tools'/module
        if src.exists():shutil.copytree(src,package/'tools'/module,dirs_exist_ok=True)
    for root in [package/'tools',package/'base/tools']:
        shutil.copyfile(Path(__file__).parent/'offline_guard.py',root/'sitecustomize.py')
    checkpoint={}
    for p in sorted((OUT/'models').glob('*.pt')):
        if not any(tag in p.stem for tag in ['_C','_adapt_']):continue
        dest=package/'models'/p.name;dest.parent.mkdir(exist_ok=True)
        state=torch.load(p,map_location='cpu',weights_only=False)
        torch.save(dict(model=state['model']),dest)
        checkpoint[p.stem]=dict(training_checkpoint_sha256=sha(p),inference_checkpoint_sha256=sha(dest))
    shutil.copyfile(OUT/'winning_config.json',package/'winning_config.json')
    calibration=read(OUT/'calibration.json')
    if (OUT/'calibration_secondary.json').exists():calibration['models'].update(read(OUT/'calibration_secondary.json')['models'])
    if (OUT/'calibration_rendering.json').exists():calibration['models'].update(read(OUT/'calibration_rendering.json')['models'])
    write(package/'calibration.json',calibration)
    (package/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
v4_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
v4_python=${MULTIDATA_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v4_python=$2; shift 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$v4_package/tools"
exec "$v4_python" "$v4_package/tools/multidata_training_v4/package_predict.py" "$@"
''');(package/'run.sh').chmod(0o755)
    (package/'README.md').write_text('''# Local v4 inference package

Run `./run.sh --python /path/to/CUDA/python --images /path/to/zarr_directory --output /path/to/NEW_output --v1 /path/to/annotation-selection-v1 --v2 /path/to/strong-tracker-v2 --source-model 44b6` (or `6bba`). The source fit is explicit; unknown dataset names never select an embryo policy.

The default is the measured selection in winning_config.json. `--disable-new-heads` preserves v3 identity. `--variant C4` runs the prespecified external candidate even if the selected default is v3. New weights are bundled. No external training images, Zoo arrays, labels, match tables, estimates or scorer inputs are needed.

The preserved base package verifies its upstream primary/secondary neural weights, DeepCenter weights, source files and historical teacher dependencies. Supply the explicit v1/v2 artifact roots and the original DeepCenter path documented in base/README.md. These large pre-existing dependencies are not duplicated here. The runtime is the tested local CUDA environment; Kaggle portability/runtime is not certified.

The existing DeepCenter checkpoint must be available at `/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt`. The v1 root supplies `public_harmonic_full/harmonic_isolated.py`, the pinned `tracking_repo`, primary and secondary weights. The v2 root supplies the frozen E teacher models. Exact dependency hashes are in `base/manifest.json`; library versions are in the study's runtime_versions.json.

Outputs are integer, in-bounds graphs and submission.csv. This package does not submit, upload or publish anything. All results remain operational exploratory; public teacher and 44b6 simulator exposure persist. Manifest hashes identify code and exported checkpoints.
''')
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    manifest=dict(package_files=files,checkpoints=checkpoint,base_manifest_sha256=sha(package/'base/manifest.json'))
    previous=read(package/'manifest.json') if (package/'manifest.json').exists() else None
    if previous is None or {k:v for k,v in previous.items() if k!='created'}!=manifest:
        write(package/'manifest.json',dict(created=now(),**manifest))
    write(OUT/'inference_package_manifest.json',dict(root=str(package),manifest_sha256=sha(package/'manifest.json'),checkpoints=checkpoint))
    return package

def pilots(package,names=None,receipt_name='inference_receipt.json'):
    from strong_tracker_v3.common import graph_hash
    selected=read(OUT/'winning_config.json')['variant']
    names=names or [r['dataset'] for r in read(V3/'fresh_pilot_lock.json')['selected']]
    receipts=[]
    for name in names:
        source='6bba' if name.startswith('44b6') else '44b6';view=OUT/'fresh_views'/name;view.mkdir(parents=True,exist_ok=True)
        link=view/f'{name}.zarr'
        if not link.exists():link.symlink_to(DATA/'train'/f'{name}.zarr',target_is_directory=True)
        for variant in list(dict.fromkeys([selected,'C4','C0'])):
            output=OUT/'fresh_pilots'/variant/name;receipt=output/'inference_receipt.json'
            if not receipt.exists():
                if output.exists():raise RuntimeError('Partial fresh output needs separate recovery namespace: '+str(output))
                output.parent.mkdir(parents=True,exist_ok=True)
                log=OUT/'logs'/f'fresh_{variant}_{name}.log'
                with log.open('w') as f:
                    command=[str(package/'run.sh'),'--python',sys.executable,'--images',str(view),'--output',str(output),
                        '--v1',str(V1),'--v2',str(STUDIES/'strong-tracker-v2'),'--source-model',source]
                    command += ['--disable-new-heads'] if variant=='C0' else ['--variant',variant]
                    subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,cwd='/tmp')
            actual=arrays(output/'predictions'/f'{name}.npz');expected=arrays((V3/'selected_predictions' if variant=='C0' else OUT/'predictions'/variant)/f'{name}.npz')
            for key in ['nodes','edges']:assert np.array_equal(actual[key],expected[key]),(name,variant,key)
            import pandas as pd
            csv=pd.read_csv(output/'submission.csv');nn=csv[csv.row_type=='node'][['node_id','t','z','y','x']].to_numpy(np.int64)
            ee=csv[csv.row_type=='edge'][['source_id','target_id']].to_numpy(np.int64)
            assert np.array_equal(nn,actual['nodes']) and np.array_equal(ee,actual['edges'])
            r=read(receipt);assert r['package_manifest_sha256']==sha(package/'manifest.json'),'Fresh receipt is from another package version'
            receipts.append(dict(dataset=name,variant=variant,seconds=r['seconds'],graph_hash=graph_hash(nn,ee),
                complete_graph_parity=True,csv_roundtrip=True,annotation_reads=r['annotation_reads'],external_dataset_reads=r['external_dataset_reads'],
                weights_loaded=r['graphs'][0]['weights_loaded'],offline_network_guard=r['offline_network_guard'],
                new_head_peak_gpu_gib=r['new_head_peak_gpu_gib'],parent_peak_rss_gib=r['parent_peak_rss_gib'],child_peak_rss_gib=r['child_peak_rss_gib'],
                package_manifest_sha256=sha(package/'manifest.json')))
    write(OUT/receipt_name,dict(completed=True,fresh_image_pilots=receipts,offline_local_dependencies=True,
        no_external_training_datasets_required=True,identity_fallback=True,kaggle_runtime_verified=False))

def run():
    package=build();pilots(package)
    archive=shutil.make_archive(str(OUT/'selected_inference_package'),'zip',package)
    write(OUT/'inference_archive.json',dict(path=archive,sha256=sha(archive),bytes=Path(archive).stat().st_size))
    status('W480',state='fresh package verified and archived')
