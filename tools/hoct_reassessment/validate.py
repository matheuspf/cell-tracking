"""Real upstream feature and solver parity checks before competition scoring."""
from __future__ import annotations

import inspect
from pathlib import Path
import runpy
import time

import numpy as np

from .common import CP, RESULTS, ROOT, setup, sha, write


def region_parity():
    setup()
    import tifffile
    import tracksdata as td
    from hoct.features import create_graph, REGIONPROPS
    from hoct.features.features import normalize_image
    from .cellpose import frame_features

    name = "44b6_81c256f0"
    labs, images, expected = [], [], []
    for t in (0, 1):
        labels = tifffile.imread(CP / "cellpose/volumes" / f"{name}-t{t:03}-masks.tif")
        image = np.load(CP / "images" / name / f"t{t:03}.npy", allow_pickle=False)
        pos, props, _ = frame_features(labels, image)
        expected.extend(np.column_stack([np.full(len(pos), t), pos, props]).tolist())
        labs.append(labels)
        images.append(normalize_image(image))
    with td.options.Options(n_workers=1, show_progress=False):
        graph = create_graph(np.stack(labs), images=np.stack(images), normalize_images=False,
                             distance_threshold=15., n_neighbors=8, delta_t=1., scale=(1., 1.625, .40625, .40625))
    rows = graph.node_attrs(attr_keys=["t", "z", "y", "x", *REGIONPROPS]).to_dicts()
    actual = np.array([[r["t"], r["z"], r["y"], r["x"], r["equivalent_diameter_area"],
                        r["intensity_min"], r["intensity_max"], r["intensity_mean"], r["intensity_std"],
                        *np.asarray(r["inertia_tensor"]).ravel(), r["border_dist"]] for r in rows])
    expected = np.asarray(expected)
    def ordered(x):
        return x[np.lexsort((x[:, 3], x[:, 2], x[:, 1], x[:, 0]))]
    actual, expected = ordered(actual), ordered(expected)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    return dict(dataset=name, frames=[0, 1], regions=len(actual), maximum_absolute_error=float(np.max(np.abs(actual-expected))),
                source=sha(inspect.getfile(create_graph)), real_masks_and_images=True,
                note="Feature parity only. Production candidates deliberately use physical distances; upstream candidate creation is not used.")


def solver_parity():
    setup()
    import polars as pl
    import tracksdata as td
    from .decode import transition, objective

    rng = np.random.default_rng(20260914)
    results = []
    for repetition in range(6):
        graph = td.graph.IndexedRXGraph()
        graph.add_node_attr_key("orphan", pl.Float64, 0.)
        orphan = rng.uniform(0, 1, 4)
        sources = graph.bulk_add_nodes([dict(t=0, orphan=0.) for _ in range(3)])
        targets = graph.bulk_add_nodes([dict(t=1, orphan=float(p)) for p in orphan])
        pairs = np.array([(a, b) for a in sources for b in targets], np.int64)
        keep = rng.random(len(pairs)) > .2
        pairs = pairs[keep]
        values = rng.uniform(0, 1, len(pairs))
        graph.add_edge_attr_key("similarity", pl.Float64, 0.)
        graph.bulk_add_edges([dict(source_id=int(a), target_id=int(b), similarity=float(p)) for (a, b), p in zip(pairs, values)])
        allow = repetition % 2 == 0
        direct, minimum = transition(sources, targets, pairs, values, orphan, allow_division=allow)
        solved = td.solvers.ILPSolver(edge_weight=.5-td.EdgeAttr("similarity"), node_weight=-10.,
            appearance_weight=.5*(1-td.NodeAttr("orphan"))*(td.NodeAttr("t") != 0),
            disappearance_weight=.25*(td.NodeAttr("t") != 1), division_weight=.25 if allow else 1e6,
            num_threads=1, gap=0., timeout=20.).solve(graph)
        # Read solution flags on the original graph so upstream ID remapping is irrelevant.
        na = graph.node_attrs(attr_keys=["node_id", "solution"])
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id", "solution"])
        chosen = [(int(a), int(b)) for a, b, selected in ea.select("source_id", "target_id", "solution").iter_rows() if selected]
        assert all(na["solution"])
        upstream = objective(chosen, pairs, values, sources, targets, orphan)
        np.testing.assert_allclose(minimum, upstream, atol=1e-8)
        results.append(dict(division=allow, assignment_objective=minimum, upstream_objective=upstream,
                            chosen_edges_equal=set(map(tuple, direct)) == set(chosen)))
    return results


def run():
    started = time.monotonic()
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tests/test_hoct_reassessment.py"))
    passed = []
    for name, fn in namespace.items():
        if name.startswith("test_"):
            fn()
            passed.append(name)
    results = dict(unit_checks=passed, real_region_parity=region_parity(), upstream_ilp_parity=solver_parity(),
                   seconds=time.monotonic()-started, annotation_reads=False)
    write(ROOT / "validation.json", results)
    write(RESULTS / "validation.json", results)
    print("VALIDATED", len(passed), "unit contracts, real-mask feature parity and six upstream ILP comparisons", flush=True)


if __name__ == "__main__":
    run()
