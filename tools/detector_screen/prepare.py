"""Freeze the detector panel and export image-only inputs and baseline centers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'work/detector-screen-20260914'
WORK = Path('/kaggle/working/cell-tracking')
DATA = Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
SPACING = [1.625, .40625, .40625]
TIMES = [9, 10, 29, 30, 49, 50, 69, 70, 89, 90]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    old = json.loads((REPO/'work/segmentation-center-review/manifest.json').read_text())
    pilot_names = {r['dataset'] for r in old['frames']}
    selections = []
    for embryo in ['44b6', '6bba']:
        candidates = []
        for p in (WORK/'annotation-selection-v1/evaluation/gt').glob(embryo+'*.npz'):
            if p.stem not in pilot_names:
                n = np.load(p)['nodes']
                candidates.append((len(n), p.stem))
        candidates.sort()
        ix = np.unique(np.rint(np.linspace(0, len(candidates)-1, 20)).astype(int))
        selections.extend((candidates[i][1], t, 'assessment') for i in ix for t in TIMES)
    selections = [(r['dataset'],r['time'],'pilot') for r in old['frames']] + selections
    frames = []; evaluation = {}; clips = {}
    for name in sorted({n for n,t,role in selections}):
        a = zarr.open_group(str(DATA/(name+'.zarr')),mode='r')['0']
        graph = zarr.open_group(str(DATA/(name+'.geff')),mode='r')
        nodes = np.column_stack([graph['nodes/ids'][:].astype(np.int64),
            *[graph[f'nodes/props/{axis}/values'][:] for axis in 'tzyx']]).astype(np.int64)
        edges = graph['edges/ids'][:].astype(np.int64)
        meta = json.loads((DATA/(name+'.zarr')/'zarr.json').read_text())
        actual_scale = meta['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1:]
        assert actual_scale == SPACING
        clips[name] = {'nodes':nodes.tolist(),'edges':edges.tolist(), 'shape':list(a.shape)}
        ns = [(n,t,r) for n,t,r in selections if n==name]
        required = sorted({max(0,min(a.shape[0]-1,t+d)) for n,t,r in ns for d in [-1,0,1]})
        image_dir = ROOT/'images'/name; image_dir.mkdir(parents=True,exist_ok=True)
        for t in required:
            path = image_dir/f't{t:03}.npy'
            if not path.exists(): np.save(path,np.asarray(a[t]))
        baseline = np.load(WORK/'annotation-selection-v1/public_harmonic_full/inputs'/f'pre_ilp_{name}.npz')
        for _,t,role in ns:
            key = f'{name}-t{t:03}'; chosen = baseline['coords'][:,0]==t
            centers = baseline['coords'][chosen,1:].astype(np.float64)
            scores = baseline['node_probabilities'][chosen].astype(np.float64)
            dest = ROOT/'predictions/incumbent'; dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/(key+'.npz'), centers_zyx=centers,scores=scores)
            row = dict(key=key,dataset=name,embryo=name.split('_')[0],time=t,role=role,
                image_path=str(image_dir/f't{t:03}.npy'),
                previous_path=str(image_dir/f't{max(0,t-1):03}.npy'),
                next_path=str(image_dir/f't{min(a.shape[0]-1,t+1):03}.npy'),
                spacing_um=SPACING,shape=list(a.shape[1:]))
            frames.append(row)
            evaluation[key] = {'gt_nodes':nodes[nodes[:,1]==t].tolist(), 'baseline_candidates':len(centers)}
            if role=='pilot':
                f = REPO/'work/segmentation-center-review/frames'/key/'focus.npz'
                if f.exists():
                    prev=np.load(f); dest=ROOT/'predictions/focus_reference';dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/(key+'.npz'),centers_zyx=prev['centroid'],scores=np.ones(len(prev['centroid'])))
        print('prepared',name,len(ns),flush=True)
    frames.sort(key=lambda r:(r['role']!='pilot',r['key']))
    panel = dict(schema=1,created='2026-09-14',frames=frames,
        selection='12 existing pilot frames plus 20 clips per embryo at evenly spaced ranks of annotated-node count, excluding pilot clips; five fixed adjacent-frame pairs per assessment clip. Frozen before challenger predictions.',
        primary_metric='One-to-one annotated-node recall at 7 micrometers after integer submission rounding, before temporal linking.',
        secondary_metrics=['recall at 5 and 6 micrometers','GT edges with both endpoints detected at 7 micrometers','candidate count and fixed incumbent-budget recall','7 micrometer near-neighbor ambiguity'],
        limitations='Annotation-count stratified exploratory panel. Public incumbent has inherited exposure to both embryos. No independent biological validation, temporal association tuning, or competition-score claim.')
    encoded=json.dumps(panel,sort_keys=True).encode();panel['definition_sha256']=hashlib.sha256(encoded).hexdigest()
    write(ROOT/'evaluation/ground_truth.json',dict(frames=evaluation,clips=clips))
    write(ROOT/'panel.json',panel)
    print(json.dumps({'frames':len(frames),'assessment_frames':sum(r['role']=='assessment' for r in frames),
        'assessment_nodes':sum(len(evaluation[r['key']]['gt_nodes']) for r in frames if r['role']=='assessment'),
        'panel':str(ROOT/'panel.json')},indent=2),flush=True)

if __name__=='__main__': main()
