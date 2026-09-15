"""CPU/CUDA agreement on real upstream batches before using the shared GPU queue."""
import itertools

import numpy as np
import torch

from .common import CONFIG, ROOT, RESULTS, arrays, load_model, pilot, read, setup, sha, write
from .runtime import CooperativeModel
from .cellpose import feature_graph


def run():
    setup()
    from hoct.data import TiledRoiDataset
    from hoct.data._batching import DataKeys as K
    from hoct.data._transforms import Standardize
    from hoct.features import REGIONPROPS
    from hoct._api import _MEAN, _STD
    from tracksdata.functional import TilingScheme

    results = []
    config = read(CONFIG)["cellpose"]
    for name in ("44b6_81c256f0", "44b6_8f5ab931"):
        data = arrays(ROOT / "cellpose/features" / f"{name}.npz")
        clip = next(c for c in pilot() if c["dataset"] == name)
        graph, _ = feature_graph(data, clip["shape"])
        dataset = TiledRoiDataset(graph, REGIONPROPS,
            TilingScheme(tuple(config["tile_shape_tzyx"]), tuple(config["tile_overlap_tzyx"])),
            dict_transforms=[Standardize(_MEAN, _STD)])
        batches = list(itertools.islice(iter(dataset), 4))
        for model_name in ("general_v1", "ctc_v0"):
            cpu = load_model(model_name)
            gpu = CooperativeModel(load_model(model_name), batches_per_lease=4)
            differences = []
            try:
                with torch.inference_mode():
                    for batch in batches:
                        x = batch[K.NODE_FEATS][None]
                        edges = batch[K.EDGE_BATCH_ID][None]
                        args = (x, batch[K.NODE_POS][None], batch[K.EDGE_POS][None], edges,
                                torch.ones(x.shape[:2], dtype=torch.bool), torch.ones(edges.shape[:2], dtype=torch.bool))
                        a = cpu(*args)
                        b = gpu(*args)
                        for index in (0, 3):
                            error = float(torch.max(torch.abs(a[index]-b[index])))
                            differences.append(error)
                            np.testing.assert_allclose(a[index].numpy(), b[index].numpy(), atol=2e-4, rtol=2e-4)
            finally:
                gpu.release()
            results.append(dict(dataset=name, model=model_name, batches=len(batches),
                                maximum_logit_error=max(differences), runtime=gpu.receipt()))
    result = dict(results=results, annotation_reads=False, unchanged_model_features_and_decoder=True,
                  config_sha256=sha(CONFIG), runtime_sha256=sha(__import__(CooperativeModel.__module__, fromlist=["x"]).__file__),
                  reason="Dense CPU runtime exceeded first-clip estimate; FP32 cooperative GPU execution preserves the model recipe.")
    write(ROOT / "runtime-parity.json", result)
    write(RESULTS / "runtime-parity.json", result)
    print("CPU/CUDA PARITY PASSED", max(r["maximum_logit_error"] for r in results), flush=True)


if __name__ == "__main__":
    run()
