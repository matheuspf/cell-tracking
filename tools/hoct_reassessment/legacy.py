"""Paired CPU FP32 unit intervention with frozen legacy observations and context."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

from .common import CONFIG, OLD, ROOT, arrays, load_model, read, save, setup, sha, write


def converted_features(nodes, properties, mode):
    """Convert only model inputs; never mutate metric nodes or context coordinates."""
    positions = np.asarray(nodes[:, 2:] * [1.625, .40625, .40625], np.float32)
    props = properties.astype(np.float32, copy=True)
    if mode == "voxel":
        positions /= 1.625
        props[:, 0] /= 1.625
        props[:, 5:14] /= 1.625**2
    elif mode != "physical":
        raise ValueError(mode)
    return positions, props


def iter_batches(nodes, pairs, properties, valid):
    """Exactly the legacy core ownership/context policy; source batching stays upstream."""
    setup()
    from tools.image_native_tracking_v5.hoct_adapter import feature_graph
    from hoct.data._batching import DataKeys, item_from_filter
    from hoct.features import REGIONPROPS

    graph, mapping, _, edge_reverse = feature_graph(nodes, pairs, properties, valid)
    index = {int(n[0]): i for i, n in enumerate(nodes)}
    physical = nodes[:, 2:] * [1.625, .40625, .40625]
    times = nodes[:, 1].astype(int)
    src = np.array([index[int(a)] for a in pairs[:, 0]])
    pair_time = times[src]
    core = np.floor(physical / 52.).astype(int)
    for t in range(int(times.max())):
        lo = max(0, min(t - 1, int(times.max()) - 4))
        hi = min(int(times.max()), lo + 4)
        for key in sorted(set(map(tuple, core[src[pair_time == t]]))):
            lower, upper = np.array(key) * 52. - 16., (np.array(key) + 1) * 52. + 16.
            selected = ((times >= lo) & (times <= hi) & valid
                        & np.all((physical >= lower) & (physical < upper), axis=1))
            ids = [mapping[int(nodes[i, 0])] for i in np.flatnonzero(selected)]
            if len(ids) < 2:
                continue
            batch = item_from_filter(graph.filter(node_ids=ids), ["z", "y", "x"], REGIONPROPS, [], [])
            if batch is None or not len(batch[DataKeys.EDGE_ID]):
                continue
            original = np.array([edge_reverse[int(i)] for i in batch[DataKeys.EDGE_ID]])
            own = (pair_time[original] == t) & np.all(core[src[original]] == key, axis=1)
            if own.any():
                yield batch, original, own, dict(t=t, nodes=len(ids), edges=len(original))


def score_one(name):
    import torch
    setup()
    from hoct._api import _MEAN, _STD
    from hoct.data._batching import DataKeys as K

    source = "6bba" if name.startswith("44b6") else "44b6"
    obs_path = OLD / "observations" / f"{name}.npz"
    bank_path = OLD / "banks" / source / "P0" / f"{name}.npz"
    dest = ROOT / "legacy/scores" / f"{name}.npz"
    inputs = dict(observations=sha(obs_path), bank=sha(bank_path), config=sha(CONFIG), code=sha(__file__))
    if dest.with_suffix(".json").exists():
        prior = read(dest.with_suffix(".json"))
        if prior["inputs"] != inputs or prior["sha256"] != sha(dest):
            raise ValueError("Stale legacy scores")
        return name
    data, bank = arrays(obs_path), arrays(bank_path)
    old = data["oldmask"]
    nodes, properties, valid = data["nodes"][old], data["properties"][old], data["valid_region"][old]
    pairs = bank["pairs"]
    model = load_model()
    mean, std = torch.tensor(_MEAN), torch.tensor(_STD)
    out = {mode: np.full(len(pairs), np.nan, np.float32) for mode in ("physical", "voxel")}
    emitted = np.zeros(len(pairs), bool)
    started = time.monotonic()
    windows = 0
    with torch.inference_mode():
        for batch, original, own, meta in iter_batches(nodes, pairs, properties, valid):
            if emitted[original[own]].any():
                raise AssertionError("Duplicate edge ownership")
            emitted[original[own]] = True
            x = batch[K.NODE_FEATS][None]
            positions, edgepos = batch[K.NODE_POS][None], batch[K.EDGE_POS][None]
            edges = batch[K.EDGE_BATCH_ID][None]
            nm, em = torch.ones(x.shape[:2], dtype=torch.bool), torch.ones(edges.shape[:2], dtype=torch.bool)
            for mode in out:
                xx, pp, ep = x.clone(), positions.clone(), edgepos.clone()
                if mode == "voxel":
                    xx[:, :, 1:5] /= 1.625
                    xx[:, :, 9:18] /= 1.625**2
                    pp /= 1.625
                    ep /= 1.625
                raw, _, _, _ = model((xx - mean) / std, pp, ep, edges, nm, em)
                values = raw[0, own].float().numpy().ravel()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite model scores")
                out[mode][original[own]] = values
            windows += 1
    if not np.array_equal(np.isfinite(out["physical"]), np.isfinite(out["voxel"])):
        raise AssertionError("Unit conversion changed candidate coverage")
    save(dest, **out)
    write(dest.with_suffix(".json"), dict(dataset=name, source=source, inputs=inputs, sha256=sha(dest),
          seconds=time.monotonic()-started, windows=windows, edges=len(pairs), scored=int(emitted.sum()),
          same_context_and_candidate_ownership=True, precision="CPU FP32"))
    print("LEGACY SCORES", name, round(time.monotonic()-started, 1), flush=True)
    return name


def decode_one(task):
    name, mode = task
    from tools.image_native_tracking_v5.common import graph
    from tools.image_native_tracking_v5.calibrate import apply
    from tools.image_native_tracking_v5.temporal_decode import decode
    from tools.cellpose_ultrack.track import validate_graph

    source = "6bba" if name.startswith("44b6") else "44b6"
    output = ROOT / "graphs" / f"legacy_{mode}" / f"{name}.npz"
    paths = {"scores": ROOT / "legacy/scores" / f"{name}.npz",
             "observations": OLD / "observations" / f"{name}.npz",
             "bank": OLD / "banks" / source / "P0" / f"{name}.npz",
             "calibration": OLD / "calibration" / f"{source}_H_20260910.json",
             "joint": OLD / "calibration" / f"{source}_J.json"}
    inputs = {k: sha(p) for k, p in paths.items()}
    inputs.update(code=sha(__file__), decoder=sha(__import__(decode.__module__, fromlist=["x"]).__file__))
    if output.with_suffix(".json").exists():
        receipt = read(output.with_suffix(".json"))
        if receipt["inputs"] != inputs or receipt["sha256"] != sha(output):
            raise ValueError("Stale decoded graph")
        return str(output)
    data, bank, scores = (arrays(paths[k]) for k in ("observations", "bank", "scores"))
    baseline = graph(name)
    mask = data["oldmask"]
    observed = data["nodes"][mask]
    assert np.array_equal(observed[np.argsort(observed[:, 0])],
                          baseline["nodes"][np.argsort(baseline["nodes"][:, 0])])
    calibration = read(paths["calibration"])["models"]["H0"]
    joint = read(paths["joint"])
    started = time.monotonic()
    nodes, edges, receipt = decode(data["nodes"][mask], bank["pairs"], apply(calibration, scores[mode]),
                                  baseline["nodes"], baseline["edges"], data["confidence"][mask], config=joint["config"])
    validate_graph(nodes, edges, [100, 64, 256, 256])
    assert np.array_equal(nodes[np.argsort(nodes[:, 0])],
                          baseline["nodes"][np.argsort(baseline["nodes"][:, 0])])
    save(output, nodes=nodes, edges=edges)
    write(output.with_suffix(".json"), dict(dataset=name, arm=f"legacy_{mode}", inputs=inputs,
          sha256=sha(output), seconds=time.monotonic()-started, decode=receipt, nodes=len(nodes), edges=len(edges)))
    print("LEGACY GRAPH", mode, name, round(time.monotonic()-started, 1), flush=True)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["score", "decode"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    names = args.datasets or read(CONFIG)["pilot_clips"]
    if args.all:
        names = sorted(p.stem for p in (OLD / "observations").glob("*.npz"))
        assert len(names) == 199
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = names if args.stage == "score" else [(n, m) for n in names for m in ("physical", "voxel")]
        list(pool.map(score_one if args.stage == "score" else decode_one, tasks))


if __name__ == "__main__":
    main()
