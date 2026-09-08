"""Run an isolated copy of Harmonic Fusion; never edit notebook originals."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .common import DATA,OUT,WORK,now,read_json,sha,write_json


def run(args):
    source=Path('/kaggle/working/biohub-harmonic-fusion.py')
    text=source.read_text();source_hash=sha(source)
    root=OUT/('public_harmonic_pilot' if args.limit else 'public_harmonic_full');root.mkdir(parents=True,exist_ok=True)
    inv=read_json(OUT/'inventory.json')
    names=[inv[0]['dataset'],next(r['dataset'] for r in inv if r['embryo']!=inv[0]['embryo'])][:args.limit] if args.limit else [r['dataset'] for r in inv]
    receipt=root/'adapter_manifest.json'
    if receipt.exists():
        previous=read_json(receipt)
        if previous['status']=='inference_complete':
            if previous['source_sha256']!=source_hash or previous['expected_samples']!=names:
                raise ValueError('Completed public run inputs changed')
            if sha(root/'harmonic_isolated.py')!=previous['adapter_sha256'] or not (root/'submission.csv').exists():
                raise ValueError('Completed public run artifact drift')
            print('Verified completed isolated public inference',flush=True)
            return
    images=root/'inputs/test';images.mkdir(parents=True,exist_ok=True)
    for n in names:
        p=images/f'{n}.zarr'
        if not p.exists():p.symlink_to(DATA/'train'/f'{n}.zarr',target_is_directory=True)
    # Reviewable mechanical path changes; algorithms/settings are unchanged.
    text=text.replace('/kaggle/working',str(root))
    old='TEST_DIR = COMP_DIR / "test"'
    if text.count(old)!=1:raise ValueError('Public notebook TEST_DIR anchor drift')
    text=text.replace(old,f'TEST_DIR = Path({str(images)!r})')
    # Stop before the notebook's validator (which reads labels and chooses examples).
    marker='TRAIN_DIR = COMP_DIR / "train"'
    if text.count(marker)!=1:raise ValueError('Public validator boundary drift')
    text=text[:text.index(marker)]
    # Export the exact pre-ILP candidate coordinates and edge probabilities.
    # This inserts only an output hook after all notebook patches, before inference.
    anchor='test_stems = list_test_stems()'
    hook='''
_study_predict_path = REPO_DIR / "scripts/predict_unet_transformer.py"
_study_predict_source = _study_predict_path.read_text()
_study_init = "    coord_lists: list[np.ndarray] = []"
_study_append = "                coord_lists.append(arr)"
if _study_predict_source.count(_study_init) != 1 or _study_predict_source.count(_study_append) != 1:
    raise RuntimeError("Study detector probability export anchor drift")
_study_predict_source = _study_predict_source.replace(_study_init, _study_init + "\\n    _study_node_probabilities = []")
_study_probability = "\\n                _study_prob = torch.sigmoid(det_logits[f_idx][0, 0]).detach().cpu().numpy()\\n                _study_node_probabilities.append(_study_prob[tuple(arr[:, 1:].astype(int).T)])"
_study_predict_source = _study_predict_source.replace(_study_append, _study_append + _study_probability)
_study_return = "    return coords, all_edges"
if _study_predict_source.count(_study_return) != 1:
    raise RuntimeError("Study pre-ILP export anchor drift")
_study_export = "    np.savez_compressed(ds_path.parent.parent / ('pre_ilp_' + ds_path.stem + '.npz'), coords=coords, node_probabilities=np.concatenate(_study_node_probabilities), edge_scores=np.asarray(all_edges, dtype=np.float64))\\n"
_study_predict_path.write_text(_study_predict_source.replace(_study_return, _study_export + _study_return))
'''
    if text.count(anchor)!=1:raise ValueError('Public inference anchor drift')
    if not args.limit:text=text.replace(anchor,hook+'\n'+anchor)
    adapter=root/'harmonic_isolated.py'
    compile(text,str(adapter),'exec');adapter.write_text(text)
    decision=dict(created=now(),source_path=str(source),source_sha256=source_hash,adapter_sha256=sha(adapter),
                  lane='diagnostic_contaminated',expected_samples=names,
                  deviations=['Output root relocated','Image-only training input view','Unused GT-reading validation cell excluded']+
                             (['Pre-ILP coordinate/edge-probability export hook'] if not args.limit else []),
                  settings='Original inference defaults; no shell install enabled',status='running')
    write_json(root/'adapter_manifest.json',decision)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',BIOHUB_ALLOW_PIP_INSTALL='0',BIOHUB_VALIDATOR_ENABLE='0',
                                    BIOHUB_DIAGNOSTIC_ARM='annotation_selection_pilot',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2')
    start=time.perf_counter()
    try:
        with (root/'run.log').open('w') as log:
            proc=subprocess.run(['/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python',str(adapter)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=12*3600)
    except subprocess.TimeoutExpired:
        decision.update(status='failed',reason='12-hour isolated notebook timeout',seconds=time.perf_counter()-start)
        write_json(root/'adapter_manifest.json',decision)
        raise
    decision.update(status='inference_complete' if proc.returncode==0 else 'failed',returncode=proc.returncode,seconds=time.perf_counter()-start,
                    original_unchanged=sha(source)==source_hash)
    write_json(root/'adapter_manifest.json',decision)
    print(decision,flush=True)
