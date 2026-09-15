"""Read each full-clip crop once for distinct frozen encoders of one family.

The original embedding function still owns crop sampling, batch order, image
hash checks and cache validation. A temporary forward hook passes each unchanged
input batch through the other encoders. No embeddings are averaged or shared
between weights.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, save_arrays, sha, write_json
from .infer import embeddings


def embeddings_many(models, row, graph, bank, paths, model_hashes):
    if set(models) != set(paths) or set(models) != set(model_hashes):
        raise ValueError('Shared crop execution requires explicit matching model/cache/hash keys')
    if len({str(Path(p).resolve()) for p in paths.values()}) != len(paths):
        raise ValueError('Distinct model outputs require distinct embedding cache paths')
    if len({id(m.encoder) for m in models.values()}) != len(models):
        raise ValueError('Shared crop execution requires distinct encoder objects')
    if any(m.training or m.encoder.training for m in models.values()):
        raise ValueError('Shared inference crops require frozen evaluation mode')
    groups, results = defaultdict(list), {}
    for name, model in models.items():
        if Path(paths[name]).exists():
            results[name] = embeddings(model, row, graph, bank, paths[name], model_hashes[name])
        else:
            groups[model.family].append(name)
    nodes = graph['nodes']
    for family, names in groups.items():
        base, *others = names
        if not others:
            results[base] = embeddings(models[base], row, graph, bank, paths[base], model_hashes[base])
            continue
        batches = []
        for frame in range(row['image_shape'][0]):
            indices = np.flatnonzero(nodes[:, 1] == frame)
            batches.extend(indices[start:start+32] for start in range(0, len(indices), 32))
        cursor, calls = iter(batches), 0
        arrays = {name: np.full((len(nodes), 128), np.nan, np.float32) for name in others}

        def encode_others(module, arguments, output):
            nonlocal calls
            indices = next(cursor, None)
            if indices is None or len(indices) != len(arguments[0]):
                raise ValueError('Original frame/batch order changed during shared crop execution')
            calls += 1
            for name in others:
                arrays[name][indices] = models[name].encoder(*arguments).cpu().numpy()

        handle = models[base].encoder.register_forward_hook(encode_others)
        try:
            results[base] = embeddings(models[base], row, graph, bank, paths[base], model_hashes[base])
        finally:
            handle.remove()
        if calls != len(batches):
            raise ValueError('Shared crop execution did not visit every original batch')
        reference = read_json(Path(paths[base]).with_suffix('.json'))
        if sha(paths[base]) != reference['sha256']:
            raise ValueError('Base embedding changed before writing distinct encoder outputs')
        for name in others:
            valid = results[base][1]
            path = Path(paths[name])
            save_arrays(path, embedding=arrays[name], valid=valid)
            receipt = dict(reference,
                inputs=dict(reference['inputs'], model_sha256=model_hashes[name]), sha256=sha(path),
                shared_crop_execution=dict(implementation_sha256=sha(Path(__file__)),
                    original_embedding_sha256=reference['sha256'], family=family,
                    separate_frozen_encoder=True, unchanged_batches_of_32=True,
                    model_count=len(names), batches=calls, embedding_averaging=False))
            write_json(path.with_suffix('.json'), receipt)
            # Use the ordinary loader to enforce exactly the same provenance
            # checks as independently computed node embeddings.
            results[name] = embeddings(models[name], row, graph, bank, path, model_hashes[name])
    return results
