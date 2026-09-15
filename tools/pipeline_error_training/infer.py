"""Explicit frozen package inference on P0 points; this module never reads labels."""
from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import torch

from .actions import Alternatives, apply_decisions
from .bank import EventBank
from .common import WORK, digest, graph_hash, read_json, save_arrays, save_graph, sha, write_json
from .crops import Images, compact_view, prediction_tracklet
from .compact_inference_crops import sample as sample_compact
from .fast_crops import sample_native
from .frame_crops import sample_gpu
from .scoring import load_model, score


def embeddings(model, row, graph, bank, path, model_hash, indices=None):
    """Cache node embeddings rather than the combinatorial alternative table."""
    path = Path(path)
    requested = np.arange(len(graph['nodes'])) if indices is None else np.asarray(sorted(set(indices)), np.int64)
    stamp = dict(image_metadata_sha256=row['metadata_sha256'], graph_hash=graph_hash(graph['nodes'], graph['edges']),
                 family=model.family, model_sha256=model_hash,
                 encoder_code_sha256={name: sha(Path(__file__).with_name(name+'.py')) for name in ['models', 'organoid_adapter', 'crops']},
                 embedding_function_sha256=digest(__import__('inspect').getsource(embeddings)),
                 requested_node_indices_sha256=digest(requested),
                 sampler_code_sha256={name:sha(Path(__file__).with_name(name+'.py')) for name in
                     {'organoid':['organoid_inference_crops'],
                      'compact':['frame_crops','compact_inference_crops'],
                      'temporal':['frame_crops']}[model.family]},
                 source_image_sha256=sha(Path(row['image_path']) / 'zarr.json'))
    if path.exists():
        receipt = read_json(path.with_suffix('.json'))
        if receipt['inputs'] != stamp or sha(path) != receipt['sha256']:
            raise ValueError('Inference embedding cache mismatch')
        for t, expected in receipt['image_frame_hashes'].items():
            if sha(Path(row['image_path']) / f'0/c/{t}/0/0/0') != expected:
                raise ValueError('Inference image content changed')
        with np.load(path, allow_pickle=False) as f:
            return f['embedding'], f['valid']
    images = Images(row['image_path'])
    nodes = graph['nodes']
    result = np.full((len(nodes), 128), np.nan, np.float32)
    valid_result = np.full((len(nodes), 3 if model.family == 'compact' else 7, 4), np.nan, np.float32)
    # Frame-coherent batches reuse native images and the released Organoid quantiles.
    with torch.inference_mode():
        for t in range(row['image_shape'][0]):
            ids = requested[nodes[requested, 1] == t]
            for start in range(0, len(ids), 32):
                ii = ids[start:start+32]
                if model.family == 'organoid':
                    from .organoid_inference_crops import patches
                    p = patches(images, nodes, ii)
                    masks = []
                    for i in ii:
                        _, tracked = prediction_tracklet(nodes, bank.pred, bank.succ, int(i))
                        masks.append([[1., float(tracked[dt]), 1., 1.] if 0 <= t+dt < images.shape[0] else [0., 0., 0., 0.]
                                      for dt in range(-2, 5)])
                    v = np.asarray(masks, np.float32)
                    x = torch.from_numpy(p).cuda()
                elif model.family == 'compact':
                    p, v = sample_compact(images,nodes,bank.pred,bank.succ,ii)
                    x = torch.from_numpy(p).cuda().float()/255
                else:
                    p, vv = sample_gpu(images, nodes, bank.pred, bank.succ, ii)
                    v = vv.cpu().numpy()
                    x = p.float()/255
                result[ii] = model.encoder(x, torch.from_numpy(v).cuda()).cpu().numpy()
                valid_result[ii] = v
                del x
            if t % 20 == 0:
                print(f'Image-derived {model.family} node embeddings: frame {t}/{row["image_shape"][0]}', flush=True)
    save_arrays(path, embedding=result, valid=valid_result)
    write_json(path.with_suffix('.json'), dict(inputs=stamp, sha256=sha(path),
        image_frame_hashes=images.read_hashes, annotation_reads=0, model_specific_path=True))
    return result, valid_result


def predict(row, graph, native, package, destination, *, replacement=False, zero=False, cache_root=None):
    destination = Path(destination)
    if zero:
        save_graph(destination, graph['nodes'], graph['edges'])
        return dict(exact_zero_head=True, graph_hash=graph_hash(graph['nodes'], graph['edges']))
    model, spec = load_model(package)
    arm = spec['recipe']['arm']
    model_hash = spec['weights_sha256']
    begin = time.monotonic()
    bank = EventBank(graph['nodes'], graph['edges'], native, row['physical_scale'])
    # Caller acquires the common lease for this complete clip. Only one GPU worker runs.
    model.cuda()
    z, valid = embeddings(model, row, graph, bank,
        Path(cache_root or WORK / 'inference_embeddings') / model_hash / f'{row["dataset"]}.npz', model_hash)
    if arm == 'A10':
        from .association import decode
        features = native['edge_features']
        scores = []
        with torch.inference_mode():
            zz = torch.from_numpy(z).cuda()
            for start in range(0, len(native['pairs']), 4096):
                pairs = torch.from_numpy(native['pairs'][start:start+4096]).cuda()
                x = features[start:start+4096]
                normalized = np.clip((x-spec['mean'])/spec['scale'], -10, 10).astype(np.float32)
                s = model.link_scores(zz, pairs, torch.from_numpy(normalized).cuda()).cpu().numpy()+x[:, 23]
                s = s/spec['calibration']['temperature']+spec['calibration']['intercept']
                scores.extend(s)
        edges, ledger = decode(graph['nodes'], graph['edges'], native, np.asarray(scores))
    else:
        alternatives = Alternatives(bank, replace=replacement)
        # Materialize only improving actions; BoundedActionComponents provides
        # transitive spill/abstention closure during application.
        def scored():
            for parent in sorted(bank.expanded):
                decisions, features, seen = [], [], set()
                for event, f, _ in bank.iter_parent(parent):
                    for d in alternatives.event(event):
                        if d.key not in seen:
                            seen.add(d.key)
                            decisions.append(d)
                            features.append(f)
                if not decisions:
                    continue
                values, _ = score(model, spec, graph['nodes'], native, z, valid, decisions, features, row['physical_scale'])
                calibration = spec['calibration']
                for d, value in zip(decisions, values):
                    value = value/calibration['temperature'] + (calibration['intercept'] if d.kind != 'keep' else 0.)
                    if value > 1e-9 and d.kind != 'keep':
                        yield d, float(value)
        # tee buffers at most one item because zip consumes both streams in lockstep.
        from itertools import tee
        first, second = tee(scored())
        edges, ledger = apply_decisions(graph['nodes'], graph['edges'],
            (d for d, _ in first), (v for _, v in second))
        ledger['bank_structural_rejections'] = dict(alternatives.rejections)
    model.cpu()
    torch.cuda.empty_cache()
    save_graph(destination, graph['nodes'], edges)
    receipt = dict(arm=arm, source=spec['recipe']['source'], model_sha256=model_hash,
        prediction_sha256=sha(destination), graph_hash=graph_hash(graph['nodes'], edges),
        node_hash=digest(graph['nodes']), unchanged_nodes=True, bank_sha256=bank.hash,
        replacement=replacement, seconds=time.monotonic()-begin, ledger=ledger,
        annotation_reads=0, explicit_model_package=str(package))
    write_json(destination.with_suffix('.json'), receipt)
    return receipt
