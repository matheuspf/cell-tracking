"""Read-only incumbent, assets, image grid, and live resource audit."""
import importlib.metadata
import importlib.util
import shutil
import subprocess
import time
import numpy as np
from .common import *

def run():
    start = time.monotonic()
    packages = {}
    for name in ['torch', 'cellpose', 'ultrack', 'detectron2', 'pyscipopt', 'mip', 'scikit-learn', 'tracksdata']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    weights = []
    focus = Path('/root/FOCUS-3D')
    sums = dict((line.split()[1], line.split()[0]) for line in (focus/'SHA256SUMS').read_text().splitlines()) if (focus/'SHA256SUMS').exists() else {}
    for name, expected in sums.items():
        p = focus / name
        actual = sha(p) if p.is_file() else None
        weights.append(dict(path=str(p), bytes=p.stat().st_size if p.exists() else 0,
                            sha256=actual, expected_sha256=expected, verified=actual == expected))
    source_files = list(focus.rglob('*.py')) if focus.exists() else []
    blockers = []
    if not source_files: blockers.append('FOCUS: /root/FOCUS-3D has checkpoints but no Python source or headless infer_volume backend')
    if packages['detectron2'] is None: blockers.append('FOCUS runtime: detectron2 package absent in the annotation runtime')
    if packages['cellpose'] is None: blockers.append('Cellpose fallback: cellpose package and local volumetric checkpoint absent')
    if packages['ultrack'] is None: blockers.append('Ultrack: ultrack package/source absent; installed pyscipopt alone is not Ultrack')
    disk_free = shutil.disk_usage(ROOT).free / 2**30
    if disk_free < 8: blockers.append(f'Persistent disk has {disk_free:.3f} GiB free, below 8 GiB floor; no persistent mask/database allocation')
    manifest = read_json(V4/'inference_package/base/manifest.json')
    protected = {}
    for key, p in manifest['external_checkpoint_paths'].items():
        assert sha(p) == manifest[key+'_weights_sha256'], p
        protected[p] = sha(p)
    native = Path(manifest['upstream_tracking_source_root'])
    for rel, expected in manifest['tracking_source_files'].items():
        assert sha(native/rel) == expected, rel
        protected[str(native/rel)] = expected
    for name, p in manifest['external_teacher_paths'].items():
        assert sha(p) == manifest['legacy_teacher_model_hashes'][name]
        protected[p] = sha(p)
    for rel, expected in manifest['package_files'].items():
        assert sha(V4/'inference_package/base'/rel) == expected, rel
    locked = {r['dataset']: r for r in read_json(V3/'selected_prediction_lock.json')['graphs']}
    rows = inventory(); clips = []; density = []
    for row in rows:
        name = row['dataset']; g = c0(name)
        h = graph_hash(g['nodes'], g['edges'])
        assert h == locked[name]['graph_hash']
        met = image_metadata(DATA/'train'/f'{name}.zarr')
        assert met['shape'] == row['image_shape']
        ev = read_json(V5/'evaluation/C0'/f'{name}.json')
        assert ev['graph_hash'] == h
        evidence = V3/'fresh_evidence'/f'{name}.npz'
        rec = read_json(evidence.with_suffix('.json'))
        assert rec['sha256'] == sha(evidence)
        f = load_graph(evidence)
        assert np.array_equal(f['incumbent_nodes'], g['nodes'])
        pairs = set(map(tuple, g['nodes'][f['pairs'], 0]))
        assert set(map(tuple, g['edges'])) <= pairs
        clips.append(dict(dataset=name, graph_hash=h, evidence_sha256=sha(evidence),
            image=met, cached_evaluation_sha256=sha(V5/'evaluation/C0'/f'{name}.json'),
            c0_sha256=sha(V3/'selected_predictions'/f'{name}.npz')))
        density.append(dict(dataset=name, embryo=row['embryo'], density=len(g['nodes'])/np.prod(met['shape'])))
    selected = []
    for embryo in ['44b6', '6bba']:
        subset = sorted((r for r in density if r['embryo']==embryo), key=lambda r:(r['density'],r['dataset']))
        for fraction in [1/3, 2/3]:
            selected.append(dict(**subset[int((len(subset)-1)*fraction)], frames=list(range(46,54)),
                                 source=embryo, source_selection_only=True))
    parity = read_json(V3/'fresh_teacher_native_parity_summary.json')
    assert parity['samples'] == 199 and parity['all_current_native_feature_fields_exact']
    from annotation_selection.metric_adapter import aggregate
    cached = [read_json(V5/'evaluation/C0'/f"{r['dataset']}.json") for r in rows]
    scores = {em: aggregate([r for r in cached if em=='pooled' or r['embryo']==em],
                           [r['dataset'] for r in rows if em=='pooled' or r['embryo']==em])
              for em in BASE}
    for em in BASE: assert abs(scores[em]['score']-BASE[em]) < 1e-12
    preserve_dirs = ['results/strong-tracker-v3','results/image-native-tracking-v5',
                     'tools/image_native_tracking_v5','tools/strong_tracker_v3']
    for directory in preserve_dirs:
        for p in (REPO/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p): protected[str(p)] = sha(p)
    for p in [REPO/'handover/segmentation-tracking-v6/EXPERIMENTS.md', V5/'status.json', V5/'selection.json']:
        protected[str(p)] = sha(p)
    result = dict(created=now(), status='blocked_segmenter_and_ultrack_independent_controls_runnable',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        packages=packages, weights=weights, focus_source_files=len(source_files), blockers=blockers,
        persistent_free_gib=disk_free, scratch_free_gib=shutil.disk_usage(SCRATCH.parent).free/2**30,
        resources=subprocess.check_output(['nvidia-smi'],text=True),
        active_processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,%cpu,rss,args'],text=True),
        prior_v5_status=read_json(V5/'status.json'), prior_v5_selection=read_json(V5/'selection.json')['selected'],
        complete_c0_clips=len(clips), full_native_evidence=True, native_parity=parity,
        cached_c0_scores=scores, pilots=selected, seconds=time.monotonic()-start,
        disk_policy='Only small code/metadata on inherited under-floor disk; transient controls use /dev/shm with its own 8 GiB floor')
    write(OUT/'preflight.json',result); write(OUT/'inputs.json',clips); write(OUT/'preservation.json',protected)
    print(json.dumps(dict(blockers=blockers, c0_clips=len(clips), seconds=result['seconds']),indent=2),flush=True)
    return result
