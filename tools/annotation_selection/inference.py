"""GT-unavailable deployment process, intentionally independent of train/labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np

from .common import OUT,digest,load_graph,now,read_json,sha,write_json
from .features import FEATURES,assert_features
from .filter_graph import filtered,keep_mask


def deny_annotations():
    reads=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if '.geff' in path or '/evaluation/' in path or 'membership_labels' in path:
                raise PermissionError(f'Annotation access forbidden in inference: {path}')
            reads.append(path)
    sys.addaudithook(audit)
    return reads


def infer(args):
    reads=deny_annotations()
    import torch
    from .image_model import predict_image
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock=read_json(OUT/'model_lock.json')
    for path,h in lock['models'].items():
        if sha(OUT/path)!=h:raise ValueError('Frozen model changed')
    manifest=read_json(OUT/'fold_manifest.json')
    dest=Path(args.output) if args.output else OUT/'predictions'
    records={}
    for d in manifest['directions']:
        source=d['source']
        if args.source and source!=args.source:continue
        root=OUT/'models'/source
        tabular={p.stem:joblib.load(p) for p in sorted(root.glob('*.joblib'))}
        images={p.stem:torch.load(p,map_location='cpu',weights_only=True) for p in sorted(root.glob('*.pt'))}
        names=d['outer_samples'][:args.limit] if args.limit else d['outer_samples']
        for j,name in enumerate(names,1):
            path=dest/source/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                records[str(path.relative_to(dest))]=sha(path);continue
            base=load_graph(OUT/'baseline/clean'/f'{name}.npz');features=base['features']
            scores={'quality':features[:,5].copy()}
            for model_id,a in tabular.items():
                assert_features(a['columns']);ix=[FEATURES.index(c) for c in a['columns']]
                scores[model_id]=a['model'].predict_proba(features[:,ix])[:,1].astype(np.float32)
            patch=np.load(OUT/'patches'/f'{name}.npy',mmap_mode='r')
            for model_id,c in images.items():scores[model_id]=predict_image(c,patch,features)
            np.savez_compressed(path,**scores)
            records[str(path.relative_to(dest))]=sha(path)
            print(f'GT-free inference {source} -> {name} {j}/{len(names)}',flush=True)
    write_json(dest/'inference_receipt.json',dict(created=now(),model_lock_sha256=sha(OUT/'model_lock.json'),
                prediction_hashes=records,annotation_read_guard='Python audit hook blocks .geff and evaluation directory opens',
                annotation_reads=0,imports_labels=False,read_path_hash=digest(sorted(set(reads))),limit=args.limit))
