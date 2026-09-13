"""Build a self-contained, offline notebook with the actual public source embedded."""
from __future__ import annotations

import ast
import base64
import difflib
import json
import zlib
from pathlib import Path

from .common import ARMS, REPO, read_json, sha, write_json


def build(args, arm, modules, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Keep standard license text and evidence beside the notebook. The executable
    # cells are unchanged, so their independently tested hashes remain valid.
    import shutil
    license_source = REPO/'tools/public946_minimal/licenses'
    if license_source.exists():
        shutil.copytree(license_source, destination/'licenses', dirs_exist_ok=True)
    original = (args.archive/'biohub-harmonic-fusion.py').read_text()
    source = original[:original.index('TRAIN_DIR = COMP_DIR / "train"')]
    files = {'public946_minimal/__init__.py': ''}
    for name in ('common.py', 'source.py', 'modules.py', 'neural.py', 'worker.py', 'isolation.py'):
        text = (REPO/'tools/public946_minimal'/name).read_text()
        if name == 'common.py':
            start = text.index('REPO = ')
            end = text.index('\n\ndef now():', start)
            text = text[:start] + 'REPO = Path.cwd()\nHANDOVER = REPO\nARMS = ' + repr(ARMS) + '\n' + text[end:]
        files['public946_minimal/'+name] = text
    files['guard/sitecustomize.py'] = (REPO/'tools/public946_minimal/guard/sitecustomize.py').read_text()
    files['worker_entry.py'] = '''import os,sys
from pathlib import Path
from types import SimpleNamespace
from public946_minimal.isolation import install
install()
from public946_minimal.worker import run
run(SimpleNamespace(job=Path(sys.argv[1])))
'''
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    source_payload = base64.b64encode(zlib.compress(source.encode())).decode()
    prefix_payload = base64.b64encode(zlib.compress(original[:original.index('test_stems = list_test_stems()')].encode())).decode()
    driver = '''# Public Harmonic Fusion v29, flexonafft; expanded public946 recipe ARM_NAME.
# Public support artifacts by pilkwang. No training, labels, downloads, or automatic submissions.
import base64, csv, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path

ARM = ARM_VALUE
MODULES = MODULES_VALUE
ROOT = Path(os.environ.get('PUBLIC946_OUTPUT', '/kaggle/working/public946-manual/' + ARM))
ROOT.mkdir(parents=True, exist_ok=True)
data_default = next((p for p in [Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/test'), Path('/kaggle/input/biohub-cell-tracking-during-development/test')] if p.exists()), None)
DATA = Path(os.environ['PUBLIC946_TEST_DIR']) if 'PUBLIC946_TEST_DIR' in os.environ else data_default
if DATA is None or not DATA.is_dir():
    raise FileNotFoundError('Attach the Biohub competition input or set PUBLIC946_TEST_DIR')
SUBMISSION = Path(os.environ.get('PUBLIC946_SUBMISSION', '/kaggle/working/submission.csv'))
software = ROOT/'software'
files = json.loads(zlib.decompress(base64.b64decode(FILES_PAYLOAD)))
for name, contents in files.items():
    path = software/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
archive = ROOT/'public_v29_without_validator.py'
archive.write_text(zlib.decompress(base64.b64decode(SOURCE_PAYLOAD)).decode())
sys.path.insert(0, str(software))
from public946_minimal.common import sha, digest, tree_hash, write_json, read_json
from public946_minimal.source import notebook_settings

slugs = ['biohub-tracking-support-pack-50ep-v1', 'biohub-temporal-unet3d-seed314159-v1', 'biohub-deepcenter-unet3d-center-prior-v1']
artifacts = {}
for slug in slugs:
    options = [Path('/kaggle/input/datasets/pilkwang')/slug, Path('/kaggle/input')/slug]
    if 'PUBLIC946_ARTIFACTS' in os.environ:
        options.insert(0, Path(os.environ['PUBLIC946_ARTIFACTS'])/slug)
    artifacts[slug] = next((p for p in options if p.is_dir()), None)
    if artifacts[slug] is None:
        raise FileNotFoundError('Attach public input pilkwang/' + slug)
public_root = ROOT/'public_source'
receipt = public_root/'portable_materialization.json'
settings = notebook_settings(archive.read_text())
if not receipt.exists():
    public_root.mkdir(parents=True, exist_ok=True)
    prefix = zlib.decompress(base64.b64decode(PREFIX_PAYLOAD)).decode()
    prefix = prefix.replace('/kaggle/working', str(public_root))
    for slug, path in artifacts.items():
        prefix = prefix.replace('/kaggle/input/' + slug, str(path))
    preparation = public_root/'prepare.py'
    preparation.write_text(prefix)
    env = {k:v for k,v in os.environ.items() if not k.startswith('BIOHUB_')}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', BIOHUB_ALLOW_PIP_INSTALL='0')
    with (public_root/'prepare.log').open('w') as log:
        subprocess.run([sys.executable, str(preparation)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(receipt, dict(source_sha256=sha(archive), preparation_sha256=sha(preparation)))
else:
    if read_json(receipt)['source_sha256'] != sha(archive):
        raise RuntimeError('Existing package output belongs to a different source')
primary = public_root/'tracking_repo/weights/unet_transformer/split_0/edge_predictor_best.pth'
secondary = public_root/'secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth'
deepcenter = artifacts['biohub-deepcenter-unet3d-center-prior-v1']/'weights/full_frame_center/best.pt'
expected = ['12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771', '9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f', '8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0']
model_hashes = {str(p):sha(p) for p in (primary,secondary,deepcenter)}
if list(model_hashes.values()) != expected:
    raise RuntimeError('Pinned public model identity mismatch')
settings['BIOHUB_SECONDARY_WEIGHTS'] = str(secondary)
settings['BIOHUB_DEEPCENTER_CHECKPOINT'] = str(deepcenter)
rows = []
for path in sorted(DATA.glob('*.zarr')):
    attrs = read_json(path/'zarr.json')['attributes']
    meta = read_json(path/'0/zarr.json')
    if [a['name'].upper() for a in attrs['multiscales'][0]['axes']] != ['T','Z','Y','X']:
        raise ValueError('Unexpected image axes')
    record = tree_hash(path)
    write_json(ROOT/'input_hashes'/(path.stem+'.json'), record, immutable=True)
    row = dict(dataset=path.stem,path=str(path),image_shape=meta['shape'],image_sha256=record['sha256'])
    rows.append(row)
if not rows:
    raise ValueError('No complete test image inputs')
env = {k:v for k,v in os.environ.items() if not k.startswith('BIOHUB_')}
env.update(PUBLIC946_OUT=str(ROOT),PUBLIC946_INFERENCE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(software/'guard')+':'+str(software),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for row in rows:
    directory=ROOT/'predictions'/row['dataset']
    job=dict(arm=ARM,modules=MODULES,row=row,out=str(ROOT),directory=str(directory),
        source_root=str(public_root/'tracking_repo'),archive_source=str(archive),original_source_sha256=sha(archive),
        primary_weights=str(primary),secondary_weights=str(secondary),deepcenter_weights=str(deepcenter),
        model_hashes=model_hashes,settings=settings,transform=None,reuse_neural=None,
        limits=dict(gpu_device_hours_max=12,gpu_allocated_gib_max=float(os.environ.get('PUBLIC946_GPU_GIB','14')),
                    new_scratch_gib_max=48,filesystem_reserve_gib_min=10),
        neural_fingerprint=digest(dict(image=row['image_sha256'],models=model_hashes,modules=MODULES,source=sha(archive),software=files)))
    job['fingerprint']=digest(job)
    job_path=directory/'job.json'
    write_json(job_path,job,immutable=True)
    done=directory/'complete.json'
    if done.exists():
        previous=read_json(done)
        if previous['fingerprint'] != job['fingerprint'] or previous['final_sha256'] != sha(directory/'final.npz'):
            raise RuntimeError('Completed package output fingerprint mismatch')
        continue
    with (directory/'worker.log').open('w') as log:
        subprocess.run([sys.executable,str(software/'worker_entry.py'),str(job_path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed',row['dataset'],flush=True)
import numpy as np
columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
SUBMISSION.parent.mkdir(parents=True,exist_ok=True)
with SUBMISSION.open('w',newline='') as handle:
    writer=csv.writer(handle)
    writer.writerow(columns)
    index=0
    for row in rows:
        with np.load(ROOT/'predictions'/row['dataset']/'final.npz',allow_pickle=False) as graph:
            for node in graph['nodes']:
                writer.writerow([index,row['dataset'],'node',*map(int,node),-1,-1]);index+=1
            for a,b in graph['edges']:
                writer.writerow([index,row['dataset'],'edge',-1,-1,-1,-1,-1,int(a),int(b)]);index+=1
write_json(ROOT/'submission_receipt.json',dict(arm=ARM,modules=MODULES,source_sha256=sha(archive),
    model_hashes=model_hashes,datasets=[r['dataset'] for r in rows],submission_sha256=sha(SUBMISSION),rows=index,
    validation='Actual fresh image-to-CSV execution',leaderboard_score=None))
print('Wrote',SUBMISSION,'with',index,'rows',flush=True)
'''
    for key, value in [('ARM_NAME', arm), ('ARM_VALUE', repr(arm)), ('MODULES_VALUE', repr(modules)),
                       ('FILES_PAYLOAD', repr(payload)), ('SOURCE_PAYLOAD', repr(source_payload)), ('PREFIX_PAYLOAD', repr(prefix_payload))]:
        driver = driver.replace(key, value)
    compile(driver, f'{arm}-portable.py', 'exec')
    name = {'B0': 'baseline_public946', 'B1': 'control_no_motion'}.get(arm, 'candidate_' + arm.lower())
    py = destination/(name+'.py')
    py.write_text(driver)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[dict(cell_type='markdown', metadata={}, source=[
            '# Harmonic Fusion v29 — '+arm+'\n',
            'Original notebook: flexonafft/biohub-harmonic-fusion v29; public artifacts by pilkwang.\n',
            'Registered modules: '+repr(modules)+'. Local reused-embryo evidence does not establish unseen-embryo generalization. No new leaderboard score is claimed.\n',
            'Attach the competition and the three public datasets listed in MANUAL_KAGGLE.md. Internet off; GPU required.\n']),
            dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=driver.splitlines(True))])
    ipynb = destination/(name+'.ipynb')
    write_json(ipynb, notebook)
    execution=read_json(args.out/'execution_lock.json')
    artifact_receipt=read_json(args.out/'artifact_hashes.json')
    pinned_artifacts={slug:{name:value for name,value in record['files'].items()
                     if name.endswith(('.pth','.pt','config.json'))}
                     for slug,record in artifact_receipt.items()}
    return dict(arm=arm, modules=modules, python=str(py), python_sha256=sha(py), notebook=str(ipynb), notebook_sha256=sha(ipynb),
        pinned_public_artifact_hashes=pinned_artifacts,metric_identity=execution['metric'],
        execution_lock_sha256=sha(args.out/'execution_lock.json'),
        public_materialized_source_hashes=execution['source']['source_hashes'],
        fixed_public_settings=execution['source']['settings'],
        effective_motion_relink='no_motion' not in modules,
        local_runtime=read_json(args.out/'preflight.json'),
        original_source_sha256=sha(args.archive/'biohub-harmonic-fusion.py'), sanitized_source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
        software_hashes={p: __import__('hashlib').sha256(s.encode()).hexdigest() for p,s in files.items()},
        public_inputs=['competition:biohub-cell-tracking-during-development', *['pilkwang/'+s for s in (
            'biohub-tracking-support-pack-50ep-v1','biohub-temporal-unet3d-seed314159-v1','biohub-deepcenter-unet3d-center-prior-v1')]],
        attribution='flexonafft public notebook; pilkwang support/model datasets; preserve bundled LICENSE files and Kaggle source attribution',
        license_status='Notebook public page: Apache-2.0; three public dataset metadata records: CC0-1.0. See licenses/provenance.json for exact verification scope.',
        shared_runtime_changes=['Explicit offline paths', 'GT-reading validator workflow excluded', 'Image-only worker and descendant guard',
            'Per-clip streaming/evidence fingerprints', 'Natural integer serialization followed by common bounds clipping'],
        scientific_changes=modules, leaderboard_score=None)
