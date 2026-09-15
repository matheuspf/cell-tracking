"""Label-blind streaming cache of genuine HOCT edge features."""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .common import CONFIG, ROOT, arrays, load_model, pilot, read, save, setup, sha, write
from .cellpose import feature_graph, guard
from .runtime import CooperativeModel


def run():
    setup()
    guard()
    from hoct._api import _MEAN, _STD
    from hoct.data import TiledRoiDataset
    from hoct.data._batching import DataKeys as K
    from hoct.data._transforms import Standardize
    from hoct.features import REGIONPROPS
    from tracksdata.functional import TilingScheme
    logging.getLogger("hoct").setLevel(logging.ERROR)
    recipe = read(CONFIG)["cellpose"]
    model = CooperativeModel(load_model("general_v1"))
    for clip in pilot():
        name = clip["dataset"]
        path = ROOT / "cellpose/features" / f"{name}.npz"
        dest = ROOT / "embeddings" / f"{name}.npz"
        inputs = dict(features=sha(path), model=read(CONFIG)["models"]["general_v1"], code=sha(__file__))
        if dest.with_suffix(".json").exists():
            old = read(dest.with_suffix(".json"))
            assert old["inputs"] == inputs and old["sha256"] == sha(dest)
            continue
        data = arrays(path)
        graph, mapping = feature_graph(data, clip["shape"])
        reverse = {int(gid): int(nid) for nid, gid in mapping.items()}
        pair_index = {tuple(map(int, pair)): i for i, pair in enumerate(data["pairs"])}
        edge_reverse = {int(e): pair_index[(reverse[int(a)], reverse[int(b)])] for e, a, b in
                        graph.edge_attrs(attr_keys=["edge_id", "source_id", "target_id"]).select("edge_id", "source_id", "target_id").iter_rows()}
        ds = TiledRoiDataset(graph, REGIONPROPS,
             TilingScheme(tuple(recipe["tile_shape_tzyx"]), tuple(recipe["tile_overlap_tzyx"])),
             dict_transforms=[Standardize(_MEAN, _STD)])
        sums = np.zeros((len(data["pairs"]), 288), np.float64)
        counts = np.zeros(len(sums), np.int32)
        started = time.monotonic()
        windows = 0
        try:
            with torch.inference_mode():
                for batch in ds:
                    if batch is None or len(batch[K.EDGE_ID]) <= 1:
                        continue
                    x, e = batch[K.NODE_FEATS][None], batch[K.EDGE_BATCH_ID][None]
                    _, _, features, _ = model(x, batch[K.NODE_POS][None], batch[K.EDGE_POS][None], e,
                         torch.ones(x.shape[:2], dtype=torch.bool), torch.ones(e.shape[:2], dtype=torch.bool))
                    ii = np.array([edge_reverse[int(v)] for v in batch[K.EDGE_ID]])
                    assert len(np.unique(ii)) == len(ii)
                    values = features[0].numpy()
                    assert values.shape == (len(ii), 288) and np.isfinite(values).all()
                    sums[ii] += values
                    counts[ii] += 1
                    windows += 1
        finally:
            model.release()
        if np.any(counts == 0):
            raise ValueError("Candidate lacks genuine contextual embedding")
        values = (sums / counts[:, None]).astype(np.float32)
        save(dest, features=values, contexts=counts)
        write(dest.with_suffix(".json"), dict(dataset=name, inputs=inputs, sha256=sha(dest), windows=windows,
              edges=len(values), channels=288, seconds=time.monotonic()-started, runtime_cumulative=model.receipt(),
              annotation_reads_blocked=True, all_candidates_cached=True, aggregation="mean over upstream contexts"))
        print("EMBEDDINGS", name, len(values), round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__":
    run()
