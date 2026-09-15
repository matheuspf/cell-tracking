"""Source-only convex residual fitting and separate label-blind application."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

from .common import REPO, ROOT, RESULTS, arrays, pilot, read, save, sha, write

PLAN = REPO / "configs/hoct-source-adaptation-20260914.json"


def source_guard(source):
    other = "6bba" if source == "44b6" else "44b6"
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            parts = Path(args[0]).resolve().parts
            if any(part == other or part.startswith(other + "_") for part in parts):
                raise PermissionError("Source-only HOCT fit rejected the other embryo")
    sys.addaudithook(audit)


def posterior(log_prior, log_orphan, targets, residual):
    """Normalize adjusted parent evidence and the frozen orphan alternative."""
    values = np.asarray(log_prior, np.float64) + residual
    maximum = np.asarray(log_orphan, np.float64).copy()
    np.maximum.at(maximum, targets, values)
    edge_exp = np.exp(values - maximum[targets])
    orphan_exp = np.exp(log_orphan - maximum)
    denominator = orphan_exp.copy()
    np.add.at(denominator, targets, edge_exp)
    return edge_exp / denominator[targets], orphan_exp / denominator, maximum + np.log(denominator)


def objective(weights, X, log_prior, log_orphan, groups, y, l2):
    residual = X @ weights[:-1] + weights[-1]
    probability, _, norm = posterior(log_prior, log_orphan, groups, residual)
    n = len(log_orphan)
    loss = (norm.sum() - np.sum((log_prior + residual)[y == 1])) / n + .5*l2*np.dot(weights, weights)
    difference = probability - y
    gradient = np.r_[X.T @ difference / n, difference.sum() / n] + l2*weights
    return float(loss), gradient


def fit(source):
    source_guard(source)
    import zarr
    output = ROOT / "models" / f"residual_{source}.npz"
    if output.exists():
        raise RuntimeError("Preserve the frozen source model; do not silently refit")
    xx, lp, lq, yy, gg, manifests = [], [], [], [], [], []
    total_groups = 0
    for clip in pilot():
        if clip["embryo"] != source:
            continue
        name = clip["dataset"]
        paths = dict(bank=ROOT / "cellpose/features" / f"{name}.npz",
                     embeddings=ROOT / "embeddings" / f"{name}.npz",
                     scores=ROOT / "cellpose/scores_cooperative_cuda/general_v1" / f"{name}.npz",
                     matches=ROOT / "matches/cellpose_general_v1_cuda" / f"{name}.npz")
        data, features, scores, matched = (arrays(paths[k]) for k in ("bank", "embeddings", "scores", "matches"))
        reverse = {int(g): int(p) for p, g in matched["matches"]}
        truth_path = Path(clip["image_path"]).with_suffix(".geff")
        truth = zarr.open_group(truth_path, mode="r")
        positive = {(reverse[int(a)], reverse[int(b)]) for a, b in np.asarray(truth["edges/ids"][:])
                    if int(a) in reverse and int(b) in reverse}
        pairs = data["pairs"]
        covered = positive & set(map(tuple, pairs))
        target_ids = sorted({int(b) for a, b in covered})
        mask = np.isin(pairs[:, 1], target_ids)
        chosen = pairs[mask]
        labels = np.array([tuple(p) in covered for p in chosen], np.float64)
        target_ids, groups = np.unique(chosen[:, 1], return_inverse=True)
        assert np.all(np.bincount(groups, weights=labels) == 1)
        node_index = {int(n[0]): i for i, n in enumerate(data["nodes"])}
        orphan = scores["orphan"][[node_index[int(t)] for t in target_ids]]
        xx.append(features["features"][mask])
        lp.append(np.log(np.clip(scores["similarity"][mask], 1e-12, 1)))
        lq.append(np.log(np.clip(orphan, 1e-12, 1)))
        yy.append(labels)
        gg.append(groups + total_groups)
        total_groups += len(target_ids)
        manifests.append(dict(dataset=name, files={k: sha(p) for k, p in paths.items()},
                              gt_metadata_sha256=sha(truth_path / "zarr.json"),
                              supported_targets=len(target_ids), candidate_rows=len(chosen), positives=int(labels.sum())))
    raw = np.concatenate(xx).astype(np.float64)
    mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), .001)
    X = np.clip((raw-mean)/scale, -6, 6)
    log_prior, log_orphan, y, groups = map(np.concatenate, (lp, lq, yy, gg))
    spec = read(PLAN)["optimizer"]
    zero = np.zeros(X.shape[1] + 1)
    initial, _ = objective(zero, X, log_prior, log_orphan, groups, y, spec["l2"])
    started = time.monotonic()
    result = minimize(objective, zero, args=(X, log_prior, log_orphan, groups, y, spec["l2"]),
                      method="L-BFGS-B", jac=True, options=dict(maxiter=spec["max_iterations"], ftol=1e-12, gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Source optimization failed: {result.message}")
    save(output, weights=result.x, mean=mean, scale=scale)
    receipt = dict(source=source, target="6bba" if source == "44b6" else "44b6", plan_sha256=sha(PLAN),
                   code_sha256=sha(__file__), sha256=sha(output), input_manifests=manifests,
                   targets=total_groups, candidate_rows=len(y), channels=X.shape[1],
                   initial_objective=initial, final_objective=float(result.fun), iterations=int(result.nit),
                   gradient_max=float(np.max(abs(result.jac))), converged=bool(result.success),
                   seconds=time.monotonic()-started, source_guard_enabled=True, target_data_read=False,
                   scope="Training loss is source resubstitution, not held-out performance.")
    write(output.with_suffix(".json"), receipt)
    write(RESULTS / f"source-fit-{source}.json", receipt)
    print("FITTED", source, total_groups, initial, result.fun, flush=True)


def predict():
    from .cellpose import guard
    guard()
    models = {source: ROOT / "models" / f"residual_{source}.npz" for source in ("44b6", "6bba")}
    locked = {}
    for source, path in models.items():
        receipt = read(path.with_suffix(".json"))
        assert receipt["source"] == source and receipt["sha256"] == sha(path)
        assert receipt["plan_sha256"] == sha(PLAN)
        locked[source] = dict(model=sha(path), receipt=sha(path.with_suffix(".json")))
    lock_path = ROOT / "source-model-lock.json"
    lock = dict(models=locked, plan=sha(PLAN))
    if lock_path.exists():
        assert read(lock_path) == lock
    else:
        write(lock_path, lock)
    for clip in pilot():
        name = clip["dataset"]
        source = "6bba" if clip["embryo"] == "44b6" else "44b6"
        paths = dict(features=ROOT / "embeddings" / f"{name}.npz",
                     bank=ROOT / "cellpose/features" / f"{name}.npz",
                     prior=ROOT / "cellpose/scores_cooperative_cuda/general_v1" / f"{name}.npz")
        output = ROOT / "cellpose/scores_cooperative_cuda/source_residual" / f"{name}.npz"
        inputs = dict(files={k:sha(p) for k,p in paths.items()}, model=locked[source], plan=sha(PLAN), code=sha(__file__))
        if output.with_suffix(".json").exists():
            receipt = read(output.with_suffix(".json"))
            assert receipt["inputs"] == inputs and receipt["sha256"] == sha(output)
            continue
        data, features, prior = (arrays(paths[k]) for k in ("bank", "features", "prior"))
        model = arrays(models[source])
        X = np.clip((features["features"].astype(np.float64) - model["mean"])/model["scale"], -6, 6)
        residual = X @ model["weights"][:-1] + model["weights"][-1]
        index = {int(n[0]): i for i,n in enumerate(data["nodes"])}
        targets = np.array([index[int(b)] for b in data["pairs"][:, 1]])
        similarity, orphan, _ = posterior(np.log(np.clip(prior["similarity"],1e-12,1)),
                                           np.log(np.clip(prior["orphan"],1e-12,1)), targets, residual)
        used = np.unique(targets)
        totals = np.bincount(targets, weights=similarity, minlength=len(orphan))
        np.testing.assert_allclose(totals[used] + orphan[used], 1, atol=1e-12)
        save(output, similarity=similarity, orphan=orphan)
        write(output.with_suffix(".json"), dict(dataset=name, source=source, target=clip["embryo"], inputs=inputs,
              sha256=sha(output), source_model_lock_sha256=sha(lock_path), target_annotations_read=False,
              model="Frozen HOCT general_v1 plus source-only convex edge-embedding residual"))
        print("ADAPTED PROBABILITIES", source, "->", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "predict"])
    parser.add_argument("--source", choices=["44b6", "6bba"])
    args = parser.parse_args()
    if args.stage == "fit":
        if not args.source:
            parser.error("fit requires --source")
        fit(args.source)
    else:
        predict()


if __name__ == "__main__":
    main()
