"""Additive inference bundle; base code and dependencies are referenced and pinned."""
import shutil,zipfile
from .common import *

MODULES=['__init__','common','native_adapter','observations','candidates','cache','banks','fresh_observations',
    'score_models','hoct_adapter','calibrate','event_paths','event_index','serialization','inference_fingerprints','temporal_decode','predict_batch','deepcenter','package_predict']

def build(selected='C0',name='inference_package'):
    package=OUT/name
    if package.exists():raise ValueError('Do not overwrite a tested package; choose a new package name')
    package.mkdir(parents=True);shutil.copytree(V4/'inference_package/base',package/'base')
    for folder in [package/'tools',package/'base/tools']:
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).parent/'offline_guard.py',folder/'sitecustomize.py')
    for module in MODULES:
        target=package/'tools/image_native_tracking_v5'/f'{module}.py';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__).parent/f'{module}.py',target)
    for pattern in ['models/native/*.pt','models/hoct/*.pt','calibration/*.json']:
        for source in OUT.glob(pattern):
            if '.resume' in source.name or '_tiny' in source.name:continue
            if source.suffix=='.pt' and source.name not in ['general_v1.pt','ctc_v0.pt']:
                if not source.with_suffix('.json').exists():continue
                assert sha(source)==read(source.with_suffix('.json'))['sha256']
            target=package/source.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for file in ['motion_config.json','execution_protocol.json']:
        shutil.copyfile(OUT/file,package/file)
    dest=package/'handover/image-native-tracking-v5/config.json';dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REPO/'handover/image-native-tracking-v5/config.json',dest)
    shutil.copytree(WORK/'python',package/'python',ignore=shutil.ignore_patterns('__pycache__','*.pyc','bin'))
    shutil.copyfile(WORK/'hoct/LICENSE',package/'HOCT_LICENSE.txt')
    parity=read(OUT/'fresh_primary_comparison.json') if (OUT/'fresh_primary_comparison.json').exists() else {}
    write(package/'winning_config.json',dict(variant=selected,baseline=BASE['pooled'],target=.95,
        source='complete both-embryo replicated and fresh-parity gate; C0 until a measured candidate passes',
        validation_state='primary comparisons complete' if parity else 'validation pending',
        fresh_verified_variants=parity.get('verified_variants',[]),
        fresh_parity_failed_variants=parity.get('failed_parity_variants',[]),
        experimental_HOCT_scope='Registered physical-unit morphology and coordinates; upstream default voxel-unit full-clip pipeline was not reproduced. See the study unit-contract audit before interpreting experimental H exports.',
        HOCT_unit_audit_sha256=sha(OUT/'HOCT_unit_contract_audit.json') if (OUT/'HOCT_unit_contract_audit.json').exists() else None))
    runner='''#!/usr/bin/env bash
set -euo pipefail
v5_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
v5_python=${V5_PYTHON:-/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python}
if [[ ${1:-} == --python ]]; then v5_python=$2; shift 2; fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export V5_PACKAGE_ROOT="$v5_package"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$v5_package/tools:$v5_package/python:$v5_package/base/tools"
exec "$v5_python" -m image_native_tracking_v5.package_predict "$@"
'''
    (package/'run.sh').write_text(runner);(package/'run.sh').chmod(0o755)
    (package/'README.md').write_text('''# Image-native tracking v5 inference

Run `./run.sh --python /path/to/CUDA/python --images IMAGE_DIRECTORY --output NEW_DIRECTORY --v1 V1_ROOT --v2 V2_ROOT --source-model 44b6`.
Source-model identifies the training source; scored transfer uses the opposite source from the input embryo.
Use `--disable-new-heads` for C0. `--variant P_DC_N_J --also-variant H_probe_native_J` executes the learned native encoder, full-frame proposals, DeepCenter optical evidence, HOCT backbone/probe and the joint solver on fresh images. The primary requested variant writes submission.csv; additional variants have separate CSVs.

The selected default is the gated export. Explicit `--variant` options expose research comparisons, including any pipeline that failed fresh graph/count parity. `winning_config.json` records verified and failed-parity variant names, without annotation arrays or scoring counts. Check that metadata before interpreting an experimental export as a reproduced result.

The bundle contains new weights, pinned HOCT/pooch source and the copied C0 code/repair models. External base dependencies are listed with hashes in base/manifest.json: the patched tracking source, primary/secondary native checkpoints and DeepCenter checkpoint remain required. This is not a self-contained CUDA environment. Use the preserved tested Python runtime; runtime versions accompany the study report.

No annotations, training datasets, count estimates or cached study graphs/features are needed. Python file/socket audit hooks deny those reads and nonlocal network access; they do not provide Linux namespace isolation. The additional v5 observer validates 100x64x256x256 images; use C0 for another shape. Local RTX4090 tests do not certify Kaggle runtime. No submission is performed.
''')
    files={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
    base_manifest=read(package/'base/manifest.json')
    external_configs={}
    for key in ['primary','secondary']:
        config_path=Path(base_manifest['external_checkpoint_paths'][key]).parent/'config.json'
        if config_path.exists():external_configs[str(config_path)]=sha(config_path)
    # The inherited inference reader verifies E_hgb against this hash-only model
    # lock. It contains checkpoint hashes/configuration provenance, no GT arrays.
    external_configs[str(V2/'model_lock.json')]=sha(V2/'model_lock.json')
    write(package/'manifest.json',dict(created=now(),files=files,selected=selected,
        base_dependencies=base_manifest,external_model_configs=external_configs,annotation_inputs=False,cached_graphs=False,new_models_bundled=True))
    archive=OUT/(name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write(OUT/(name+'_receipt.json'),dict(package=str(package),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(files),selected=selected,zip_crc_passed=True))
    return package

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--selected',default='C0');p.add_argument('--name',default='inference_package');a=p.parse_args();print(build(a.selected,a.name))
